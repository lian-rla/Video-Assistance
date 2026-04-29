# =========================
# easyOCR_timestamp.py
# =========================

import os
import cv2
import easyocr
import pandas as pd
import numpy as np
import multiprocessing as mp
import time
import math
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# =========================
# OCR core
# =========================
def extract_timestamp_easyocr(reader, roi, expected_len=16):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    blur = cv2.GaussianBlur(norm, (3, 3), 0)
    up = cv2.resize(blur, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    result = reader.readtext(up, detail=1, paragraph=True)
    if not result:
        return "", 0.0

    digits, confs = [], []
    for r in result:
        if len(r) == 3:
            _, text, conf = r
        elif len(r) == 2:
            _, text = r
            conf = 0.9
        else:
            continue

        t = ''.join([c for c in text if c.isdigit()])
        if t:
            digits.append(t)
            confs.append(conf)

    text_final = ''.join(digits)
    mean_conf = float(np.mean(confs)) if confs else 0.0

    if len(text_final) > expected_len:
        text_final = text_final[:expected_len]
    if len(text_final) < expected_len:
        return "", mean_conf

    return text_final, mean_conf


def _detect_gpu_count():
    try:
        import torch
        if torch.cuda.is_available():
            return int(torch.cuda.device_count())
        return 0
    except Exception:
        return 0


def _make_workers_by_gpu_count(frame_count: int, gpu_count: int):
    if gpu_count <= 0:
        return [{
            "worker_id": 0, "use_gpu": False, "gpu_id": None, "start": 0, "end": frame_count
        }]

    chunk = math.ceil(frame_count / gpu_count)
    workers = []
    for i in range(gpu_count):
        start = i * chunk
        end = min((i + 1) * chunk, frame_count)
        if start >= frame_count:
            break
        workers.append({
            "worker_id": i, "use_gpu": True, "gpu_id": i, "start": start, "end": end
        })
    return workers


def _expected_steps_for_range(start_frame: int, end_frame: int, dt_frames: int) -> int:

    if end_frame <= start_frame:
        return 0
    length = end_frame - start_frame
    return int(math.ceil(length / dt_frames))


def ocr_process(worker_id, video_path, start_frame, end_frame, fps, roi, shared_list,
                use_gpu, gpu_id, dt_frames,
                progress_counter, done_flag):

    try:
        # 중요: reader 초기화 전에 CUDA_VISIBLE_DEVICES 설정
        if use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            reader = easyocr.Reader(['en'], gpu=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            reader = easyocr.Reader(['en'], gpu=False)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with done_flag.get_lock():
                done_flag.value += 1
            return

        x, y, w_roi, h_roi = roi
        frame_idx = start_frame
        prev_text = ""

        while frame_idx < end_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            roi_img = frame[y:y + h_roi, x:x + w_roi]
            if roi_img.size != 0:
                number, conf = extract_timestamp_easyocr(reader, roi_img, expected_len=16)

                # 백필(이전 값)
                if len(number) != 16 and prev_text:
                    number = prev_text
                if len(number) == 16:
                    prev_text = number

                shared_list.append({
                    "index": int(frame_idx),
                    "time_sec": float(frame_idx / fps),
                    "timestamp": str(number),
                    "conf": float(conf),
                    "worker": int(worker_id),
                    "used_gpu": bool(use_gpu),
                    "gpu_id": int(gpu_id) if (use_gpu and gpu_id is not None) else -1,
                })

            frame_idx += dt_frames

            # progress tick (샘플 1개 처리했다고 가정)
            with progress_counter.get_lock():
                progress_counter.value += 1

        cap.release()

    finally:
        with done_flag.get_lock():
            done_flag.value += 1


# =========================
# GUI
# =========================
class EasyOCRApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Timestamp OCR 변환기 (by LeeLab)")
        self.root.geometry("980x620")

        self.video_path = tk.StringVar(value="")
        self.csv_dir = tk.StringVar(value="")  # ✅ 폴더 선택
        self.status_var = tk.StringVar(value="대기 중")

        self.total_progress_var = tk.DoubleVar(value=0.0)
        self.file_progress_var = tk.DoubleVar(value=0.0)

        self._build_ui()

        # runtime
        self.ctx = None
        self.manager = None
        self.shared_list = None
        self.progress_counter = None
        self.done_flag = None
        self.expected_total_steps = 1
        self.procs = []
        self._polling = False
        self._start_time = None
        self._final_csv_path = None

    def _build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=12, pady=10)

        lf_video = ttk.LabelFrame(top, text="영상 선택")
        lf_video.grid(row=0, column=0, padx=(0, 10), sticky="nsew")

        lf_csv = ttk.LabelFrame(top, text="CSV 저장 폴더 선택")
        lf_csv.grid(row=0, column=1, sticky="nsew")

        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=1)

        # 영상 선택
        ttk.Button(lf_video, text="영상 파일 선택", command=self.select_video)\
            .grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.lbl_video = ttk.Label(lf_video, text="선택 영상: 없음")
        self.lbl_video.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        lf_video.columnconfigure(0, weight=1)

        # CSV 폴더 선택
        ttk.Button(lf_csv, text="CSV 저장 폴더 선택", command=self.select_csv_dir)\
            .grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        self.lbl_csv = ttk.Label(lf_csv, text="저장 폴더: 없음")
        self.lbl_csv.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="w")
        lf_csv.columnconfigure(0, weight=1)

        # 실행 버튼 / 상태
        mid = tk.Frame(self.root)
        mid.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_start = ttk.Button(mid, text="OCR 실행", command=self.start)
        self.btn_start.pack(side="left", padx=(0, 10))
        ttk.Label(mid, textvariable=self.status_var).pack(side="left")

        # 진행 상황
        lf_prog = ttk.LabelFrame(self.root, text="진행 상황")
        lf_prog.pack(fill="x", padx=12, pady=8)

        ttk.Label(lf_prog, text="전체 진행률:").grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        self.lbl_total = ttk.Label(lf_prog, text="0.0%")
        self.lbl_total.grid(row=0, column=1, padx=10, pady=(10, 4), sticky="e")

        self.pb_total = ttk.Progressbar(lf_prog, variable=self.total_progress_var, maximum=100.0)
        self.pb_total.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        ttk.Label(lf_prog, text="현재 파일 진행률:").grid(row=2, column=0, padx=10, pady=(0, 4), sticky="w")
        self.lbl_file = ttk.Label(lf_prog, text="0.0%")
        self.lbl_file.grid(row=2, column=1, padx=10, pady=(0, 4), sticky="e")

        self.pb_file = ttk.Progressbar(lf_prog, variable=self.file_progress_var, maximum=100.0)
        self.pb_file.grid(row=3, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")

        lf_prog.columnconfigure(0, weight=1)

        # 로그창
        lf_log = ttk.LabelFrame(self.root, text="로그")
        lf_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log = scrolledtext.ScrolledText(lf_log, height=18)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("[READY] 영상 선택 + 저장 폴더 선택 후 OCR 실행\n")

    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")
        self.root.update_idletasks()

    def _set_progress(self, pct: float):
        pct = max(0.0, min(100.0, pct))
        self.total_progress_var.set(pct)
        self.file_progress_var.set(pct)  # 단일 파일이므로 동일하게
        self.lbl_total.config(text=f"{pct:.1f}%")
        self.lbl_file.config(text=f"{pct:.1f}%")
        self.root.update_idletasks()

    def select_video(self):
        path = filedialog.askopenfilename(
            title="영상 파일 선택",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            self.lbl_video.config(text=f"선택 영상: {path}")
            self._log(f"[GUI] Video = {path}\n")

    def select_csv_dir(self):
        path = filedialog.askdirectory(title="CSV 저장 폴더 선택")
        if path:
            self.csv_dir.set(path)
            self.lbl_csv.config(text=f"저장 폴더: {path}")
            self._log(f"[GUI] CSV DIR = {path}\n")

    def start(self):
        video_path = self.video_path.get().strip()
        csv_dir = self.csv_dir.get().strip()

        if (not video_path) or (not os.path.exists(video_path)):
            messagebox.showerror("오류", "올바른 영상 파일을 선택하세요.")
            return
        if (not csv_dir) or (not os.path.isdir(csv_dir)):
            messagebox.showerror("오류", "CSV 저장 폴더를 선택하세요.")
            return

        video_stem = os.path.splitext(os.path.basename(video_path))[0]
        csv_path = os.path.join(csv_dir, f"{video_stem}_timestamp.csv")
        self._final_csv_path = csv_path

        self.btn_start.config(state="disabled")
        self.status_var.set("ROI 선택 대기 중...")
        self._set_progress(0.0)

        try:
            # ROI 선택
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {video_path}")

            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise RuntimeError("첫 프레임을 불러오지 못했습니다.")

            self._log("[INFO] ROI 선택 창이 열립니다. 드래그 후 Enter/Space.\n")
            roi = cv2.selectROI("Select ROI for timestamp", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyAllWindows()

            x, y, w, h = map(int, roi)
            if w == 0 or h == 0:
                raise ValueError("ROI가 잘못되었습니다. (폭/높이 0)")
            roi = (x, y, w, h)
            self._log(f"[INFO] ROI = x={x}, y={y}, w={w}, h={h}\n")

            # 메타데이터
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            if fps <= 0 or frame_count <= 0:
                raise RuntimeError("FPS 또는 총 프레임 수가 유효하지 않습니다.")

            dt_frames = max(1, int(round(fps / 30)))  # 30Hz
            self._log(f"[INFO] FPS={fps:.2f}, frames={frame_count}, dt_frames={dt_frames} (30Hz)\n")

            # GPU 감지 및 분할
            gpu_count = _detect_gpu_count()
            workers = _make_workers_by_gpu_count(frame_count, gpu_count)

            self._log(f"[INFO] Detected GPU count = {gpu_count}\n")
            for wkr in workers:
                self._log(f"  - worker={wkr['worker_id']} use_gpu={wkr['use_gpu']} gpu_id={wkr['gpu_id']} "
                          f"range=[{wkr['start']},{wkr['end']})\n")

            # 진행률 총 step 계산
            expected = 0
            for wkr in workers:
                expected += _expected_steps_for_range(wkr["start"], wkr["end"], dt_frames)
            self.expected_total_steps = max(1, expected)

            # 멀티프로세싱 세팅
            self.ctx = mp.get_context("spawn")
            self.manager = self.ctx.Manager()
            self.shared_list = self.manager.list()

            self.progress_counter = self.ctx.Value('i', 0)
            self.done_flag = self.ctx.Value('i', 0)

            # 프로세스 시작
            self.procs = []
            self._start_time = time.time()
            self.status_var.set("OCR 실행 중...")
            self._log(f"[INFO] CSV 저장 예정: {csv_path}\n")
            self._log("[INFO] OCR 시작\n")

            for wkr in workers:
                p = self.ctx.Process(
                    target=ocr_process,
                    args=(
                        wkr["worker_id"],
                        video_path,
                        wkr["start"],
                        wkr["end"],
                        fps,
                        roi,
                        self.shared_list,
                        wkr["use_gpu"],
                        wkr["gpu_id"],
                        dt_frames,
                        self.progress_counter,
                        self.done_flag
                    )
                )
                p.start()
                self.procs.append(p)

            # 폴링 시작
            self._polling = True
            self.root.after(200, self._poll_progress, len(workers))

        except Exception as e:
            self._log("[ERROR]\n" + traceback.format_exc() + "\n")
            self.status_var.set("오류 발생")
            messagebox.showerror("오류", str(e))
            self.btn_start.config(state="normal")

    def _poll_progress(self, num_workers: int):
        if not self._polling:
            return

        try:
            processed = int(self.progress_counter.value)
            done = int(self.done_flag.value)
        except Exception:
            processed, done = 0, 0

        pct = (processed / self.expected_total_steps) * 100.0
        self._set_progress(pct)

        # 아직 실행 중
        if done < num_workers:
            self.root.after(200, self._poll_progress, num_workers)
            return

        # 모두 종료 -> join
        for p in self.procs:
            try:
                p.join(timeout=0.1)
            except Exception:
                pass

        # 결과 저장
        try:
            df = pd.DataFrame(list(self.shared_list))
            if len(df) == 0:
                raise RuntimeError("OCR 결과가 비었습니다. ROI/영상/환경을 확인하세요.")

            df.sort_values(by="index", inplace=True)
            df.reset_index(drop=True, inplace=True)

            out_dir = os.path.dirname(self._final_csv_path) or "."
            os.makedirs(out_dir, exist_ok=True)
            df.to_csv(self._final_csv_path, index=False, encoding="utf-8-sig", quoting=1)

            elapsed_min = (time.time() - self._start_time) / 60.0 if self._start_time else 0.0
            self._set_progress(100.0)
            self._log(f"\n[INFO] CSV 저장 완료: {self._final_csv_path}\n")
            self._log(f"[INFO] rows={len(df)}, elapsed={elapsed_min:.2f} min\n")
            self.status_var.set("완료")
            messagebox.showinfo("완료", "OCR이 완료되었습니다.")

        except Exception as e:
            self._log("[ERROR]\n" + traceback.format_exc() + "\n")
            self.status_var.set("오류 발생")
            messagebox.showerror("오류", str(e))

        finally:
            self._polling = False
            self.btn_start.config(state="normal")


# =========================
# cut_video_GUI_.py
# =========================

import os
import cv2
import pandas as pd
import numpy as np
import re
import time
import traceback
from collections import Counter

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading

OCR_CSV_PATH       = r"C:\Users\leelab\Desktop\11310106_timestamp_dualGPU.csv"
VIDEO_PATH         = r"F:\rat948\SEC_20240504_1219\20240504_mp4\12190106_0200.mp4"
EVENTS_PATH        = r"C:\Users\leelab\Desktop\Events.xlsx"
CLIPPED_VIDEO_PATH = r"C:\Users\leelab\Desktop\test2_.mp4"
UNREAL_LOG_PATH    = r"C:\Users\leelab\Desktop\2024.05.04-12.17.34.639_Circular_memory.csv"

# 기본(레거시) Unreal 컬럼 지정(엑셀 레터 기준)
UNREAL_TRUE_COL_LETTER  = "S"
UNREAL_TRIAL_COL_LETTER = "F"

CSV_TS_COL   = "Column4"
CSV_INFO_COL = "Column18"
TTL_PATTERN  = "TTL Input on AcqSystem1_0 board 0 port 2 value (0x0004)."

OFFSET_SAMPLE_COUNT = 200
OUTPUT_MERGED_CSV_PATH = r"C:\Users\leelab\Desktop\merged_cheetah_trial_ocr_with_fill.csv"

UNREAL_PREVIEW_MAX_ROWS = 5000



def normalize_digits(s: str) -> str:
    s = str(s)
    return re.sub(r"\D", "", s)

def excel_col_letter_to_0based(letter: str) -> int:
    letter = letter.strip().upper()
    n = 0
    for ch in letter:
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"엑셀 컬럼 레터가 아닙니다: {letter}")
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1

def read_table_auto(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    else:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            try:
                return pd.read_csv(path, encoding="utf-8")
            except UnicodeDecodeError:
                return pd.read_csv(path, encoding="cp949")

def estimate_offset(video_ts_int: np.ndarray, csv_ts_int: np.ndarray, sample_count: int = 200) -> int:
    if len(video_ts_int) == 0 or len(csv_ts_int) == 0:
        raise ValueError("offset 추정을 위한 timestamp 데이터가 부족합니다.")

    sample_count = min(sample_count, len(video_ts_int))
    sample_indices = np.linspace(0, len(video_ts_int) - 1, num=sample_count, dtype=int)

    diffs = []
    for idx in sample_indices:
        ts_v = video_ts_int[idx]
        j = np.argmin(np.abs(csv_ts_int - ts_v))
        ts_c = csv_ts_int[j]
        diffs.append(ts_c - ts_v)

    diffs = np.array(diffs, dtype=np.int64)
    offset = int(np.median(diffs))
    return offset

def find_nearest_frame_for_ts(target_ts: int, df_ocr: pd.DataFrame) -> int:
    arr = df_ocr["timestamp_int"].to_numpy()
    idx = int(np.argmin(np.abs(arr - np.int64(target_ts))))
    frame_idx = int(df_ocr.iloc[idx]["index"])
    return frame_idx

def nearest_from_sorted(sorted_arr: np.ndarray, x: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(sorted_arr, x, side="left")
    idx0 = np.clip(idx - 1, 0, len(sorted_arr) - 1)
    idx1 = np.clip(idx,     0, len(sorted_arr) - 1)

    v0 = sorted_arr[idx0]
    v1 = sorted_arr[idx1]
    pick1 = np.abs(v1 - x) < np.abs(v0 - x)
    return np.where(pick1, v1, v0)

def compute_offset_error_jump_positions(df_clip: pd.DataFrame, csv_all_ts_sorted: np.ndarray, offset: int):
    pred_csv = df_clip["timestamp_int"].to_numpy().astype(np.int64) + np.int64(offset)
    nearest_csv = nearest_from_sorted(csv_all_ts_sorted, pred_csv)
    err = (nearest_csv - pred_csv).astype(np.int64)
    jump = np.abs(np.diff(err)).astype(np.int64)  # 길이 N-1
    order_desc = np.argsort(-jump)
    return jump, order_desc

def evenly_spaced_positions(n_positions: int, length: int):
    if length <= 0 or n_positions <= 0:
        return np.array([], dtype=int)
    if length == 1:
        return np.zeros(n_positions, dtype=int)
    xs = np.linspace(0, length - 1, num=n_positions, dtype=float)
    return np.clip(np.round(xs).astype(int), 0, length - 1)

def build_counter_roundrobin(candidate_positions: np.ndarray, need: int) -> Counter:
    c = Counter()
    if need <= 0 or len(candidate_positions) == 0:
        return c
    for i in range(need):
        pos = int(candidate_positions[i % len(candidate_positions)])
        c[pos] += 1
    return c

def plan_insert_drop_for_trial(df_trial_ocr: pd.DataFrame,
                              expected_frames: int,
                              csv_all_ts_sorted: np.ndarray,
                              offset: int) -> tuple[Counter, Counter]:
    actual_frames = len(df_trial_ocr)
    diff = expected_frames - actual_frames

    insert_counter = Counter()
    drop_counter = Counter()

    if diff == 0:
        return insert_counter, drop_counter

    if actual_frames <= 1:
        if diff > 0:
            cand = np.array([0], dtype=int)
            insert_counter = build_counter_roundrobin(cand, diff)
        else:
            for pos in range(min(-diff, actual_frames)):
                drop_counter[pos] = 1
        return insert_counter, drop_counter

    jump, order_desc = compute_offset_error_jump_positions(df_trial_ocr, csv_all_ts_sorted, offset)

    if diff > 0:
        need_insert = diff
        cand = np.array([int(i) for i in order_desc.tolist()], dtype=int)
        if len(cand) == 0:
            cand = evenly_spaced_positions(min(need_insert, actual_frames), actual_frames)
        insert_counter = build_counter_roundrobin(cand, need_insert)
        return insert_counter, drop_counter

    # diff < 0 => drop
    need_drop = -diff
    cand = []
    for i in order_desc.tolist():
        pos = int(i + 1)
        if 0 <= pos <= actual_frames - 1:
            cand.append(pos)

    cand = list(dict.fromkeys(cand))
    if len(cand) == 0:
        cand = evenly_spaced_positions(min(need_drop, actual_frames), actual_frames).tolist()

    for pos in cand[:min(need_drop, len(cand))]:
        drop_counter[int(pos)] = 1

    remaining = need_drop - len(drop_counter)
    if remaining > 0:
        extra = evenly_spaced_positions(remaining, actual_frames).tolist()
        for pos in extra:
            if len(drop_counter) >= need_drop:
                break
            if int(pos) not in drop_counter:
                drop_counter[int(pos)] = 1

    return insert_counter, drop_counter

def apply_insert_drop_to_positions(positions: list[int], insert_counter: Counter, drop_counter: Counter) -> list[int]:
    out = []
    for pos in positions:
        if drop_counter.get(pos, 0) > 0:
            continue
        out.append(pos)
        k = insert_counter.get(pos, 0)
        if k > 0:
            out.extend([pos] * k)
    return out


# =========================================================
# GUI
# =========================================================
class TrialAlignGuiApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Trial Align (OCR + Cheetah TTL + Unreal Log) - GUI")
        self.root.geometry("1200x820")


        self.ocr_csv_path = tk.StringVar(value="")
        self.video_path   = tk.StringVar(value="")
        self.events_path  = tk.StringVar(value="")
        self.unreal_path  = tk.StringVar(value="")
        self.out_dir      = tk.StringVar(value="")


        self.unreal_true_col_name  = tk.StringVar(value="")
        self.unreal_trial_col_name = tk.StringVar(value="")
        self.unreal_row_start = tk.StringVar(value="")
        self.unreal_row_end   = tk.StringVar(value="")

        self._unreal_columns_cache = []
        self._unreal_df_cache = None          
        self._unreal_preview_df = None        
        self._unreal_preview_offset = 0      

        self.status_var   = tk.StringVar(value="대기 중")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._worker_thread = None

        self._build_ui()
        self._log("[READY] 파일 선택 → (Unreal 미리보기로 범위 선택 가능) → 실행\n")

    # ---------- UI ----------
    def _build_ui(self):
        top = ttk.LabelFrame(self.root, text="입력/출력 선택")
        top.pack(fill="x", padx=12, pady=10)

        btn_row = tk.Frame(top)
        btn_row.pack(fill="x", padx=10, pady=(10, 6))

        self.btn_ocr = ttk.Button(btn_row, text="OCR CSV 선택", command=self.pick_ocr_csv)
        self.btn_vid = ttk.Button(btn_row, text="Video 선택", command=self.pick_video)
        self.btn_evt = ttk.Button(btn_row, text="Cheetah Events 선택", command=self.pick_events)
        self.btn_unr = ttk.Button(btn_row, text="Unreal Log 선택", command=self.pick_unreal)
        self.btn_out = ttk.Button(btn_row, text="저장 폴더 선택", command=self.pick_out_dir)

        for w in (self.btn_ocr, self.btn_vid, self.btn_evt, self.btn_unr, self.btn_out):
            w.pack(side="left", padx=(0, 8))

        paths = tk.Frame(top)
        paths.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(paths, text="OCR CSV:").grid(row=0, column=0, sticky="w")
        ttk.Label(paths, textvariable=self.ocr_csv_path).grid(row=0, column=1, sticky="w")

        ttk.Label(paths, text="Video:").grid(row=1, column=0, sticky="w")
        ttk.Label(paths, textvariable=self.video_path).grid(row=1, column=1, sticky="w")

        ttk.Label(paths, text="Events:").grid(row=2, column=0, sticky="w")
        ttk.Label(paths, textvariable=self.events_path).grid(row=2, column=1, sticky="w")

        ttk.Label(paths, text="Unreal Log:").grid(row=3, column=0, sticky="w")
        ttk.Label(paths, textvariable=self.unreal_path).grid(row=3, column=1, sticky="w")

        ttk.Label(paths, text="Output Folder:").grid(row=4, column=0, sticky="w")
        ttk.Label(paths, textvariable=self.out_dir).grid(row=4, column=1, sticky="w")

        paths.columnconfigure(1, weight=1)

        # Unreal Align 옵션
        opt = ttk.LabelFrame(self.root, text="Unreal Align 옵션 (미리보기 표에서 마우스로 범위 선택 가능)")
        opt.pack(fill="x", padx=12, pady=(0, 10))

        opt_row1 = tk.Frame(opt)
        opt_row1.pack(fill="x", padx=10, pady=(8, 4))

        ttk.Label(opt_row1, text="TRUE 컬럼:").pack(side="left")
        self.cmb_true = ttk.Combobox(opt_row1, textvariable=self.unreal_true_col_name, state="readonly", width=32)
        self.cmb_true.pack(side="left", padx=(6, 14))

        ttk.Label(opt_row1, text="TRIAL 컬럼:").pack(side="left")
        self.cmb_trial = ttk.Combobox(opt_row1, textvariable=self.unreal_trial_col_name, state="readonly", width=32)
        self.cmb_trial.pack(side="left", padx=(6, 14))

        ttk.Label(opt_row1, text="행 범위(1부터):").pack(side="left")
        ttk.Label(opt_row1, text="Start").pack(side="left", padx=(6, 4))
        self.ent_rs = ttk.Entry(opt_row1, textvariable=self.unreal_row_start, width=8)
        self.ent_rs.pack(side="left")
        ttk.Label(opt_row1, text="End").pack(side="left", padx=(10, 4))
        self.ent_re = ttk.Entry(opt_row1, textvariable=self.unreal_row_end, width=8)
        self.ent_re.pack(side="left", padx=(0, 10))

        self.btn_preview = ttk.Button(opt_row1, text="미리보기/범위 선택", command=self.open_unreal_preview)
        self.btn_preview.pack(side="left")

        opt_row2 = tk.Frame(opt)
        opt_row2.pack(fill="x", padx=10, pady=(0, 8))
        ttk.Label(
            opt_row2,
            text=f"※ Unreal Log 선택 후 미리보기 가능. 표는 최대 {UNREAL_PREVIEW_MAX_ROWS}행까지만 표시(속도 보호)."
        ).pack(side="left")

        mid = tk.Frame(self.root)
        mid.pack(fill="x", padx=12, pady=(6, 6))

        self.btn_run = ttk.Button(mid, text="실행 (Trial Align + CSV + Video)", command=self.on_run)
        self.btn_run.pack(side="left", padx=(0, 10))

        ttk.Label(mid, textvariable=self.status_var).pack(side="left")

        lf_prog = ttk.LabelFrame(self.root, text="진행률")
        lf_prog.pack(fill="x", padx=12, pady=8)

        self.pb = ttk.Progressbar(lf_prog, variable=self.progress_var, maximum=100.0)
        self.pb.pack(fill="x", padx=10, pady=10)

        lf_log = ttk.LabelFrame(self.root, text="로그")
        lf_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log = scrolledtext.ScrolledText(lf_log, height=24)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

    # ---------- helpers ----------
    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")
        self.root.update_idletasks()

    def _set_progress(self, p: float):
        p = max(0.0, min(100.0, p))
        self.progress_var.set(p)
        self.root.update_idletasks()

    def _refresh_unreal_column_dropdowns(self, unreal_path: str):
        try:
            df = read_table_auto(unreal_path)
            self._unreal_df_cache = df  
            cols = list(df.columns.astype(str))
            self._unreal_columns_cache = cols

            self.cmb_true["values"] = cols
            self.cmb_trial["values"] = cols


            try:
                true_idx = excel_col_letter_to_0based(UNREAL_TRUE_COL_LETTER)
                trial_idx = excel_col_letter_to_0based(UNREAL_TRIAL_COL_LETTER)
                if 0 <= true_idx < len(cols) and not self.unreal_true_col_name.get():
                    self.unreal_true_col_name.set(cols[true_idx])
                if 0 <= trial_idx < len(cols) and not self.unreal_trial_col_name.get():
                    self.unreal_trial_col_name.set(cols[trial_idx])
            except Exception:
                pass

            self._log(f"[GUI] Unreal columns loaded: {len(cols)} columns, rows={len(df)}\n")
        except Exception as e:
            self._unreal_df_cache = None
            self._unreal_columns_cache = []
            self._log(f"[WARN] Unreal 컬럼 로드 실패: {e}\n")

    # ---------- Unreal preview ----------
    def open_unreal_preview(self):
        unreal = self.unreal_path.get().strip()
        if not unreal or not os.path.exists(unreal):
            messagebox.showerror("오류", "먼저 Unreal Log 파일을 선택하세요.")
            return

        if self._unreal_df_cache is None:
            try:
                self._unreal_df_cache = read_table_auto(unreal)
            except Exception as e:
                messagebox.showerror("오류", f"Unreal Log 로드 실패:\n{e}")
                return

        df = self._unreal_df_cache
        n_total = len(df)
        n_show = min(n_total, UNREAL_PREVIEW_MAX_ROWS)


        self._unreal_preview_offset = 0
        self._unreal_preview_df = df.iloc[self._unreal_preview_offset:self._unreal_preview_offset + n_show].copy()

        true_col = self.unreal_true_col_name.get().strip()
        trial_col = self.unreal_trial_col_name.get().strip()

        cols_all = list(self._unreal_preview_df.columns.astype(str))
        show_cols = []
        show_cols.extend(cols_all[:6])

        for c in [true_col, trial_col]:
            if c and c in cols_all and c not in show_cols:
                show_cols.append(c)

        show_cols = show_cols[:10]

        # Toplevel 창
        win = tk.Toplevel(self.root)
        win.title("Unreal Log 미리보기 - 마우스로 행 범위 선택")
        win.geometry("1100x650")

        top_info = tk.Frame(win)
        top_info.pack(fill="x", padx=10, pady=8)
        ttk.Label(top_info, text=f"전체 {n_total}행 중 상단 {n_show}행 표시").pack(side="left")

        sel_var = tk.StringVar(value="선택된 행: 없음")
        ttk.Label(top_info, textvariable=sel_var).pack(side="right")

        # Treeview + Scrollbars
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        yscroll = ttk.Scrollbar(frame, orient="vertical")
        xscroll = ttk.Scrollbar(frame, orient="horizontal")

        columns = ["__row__"] + show_cols
        tv = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            selectmode="extended",
            yscrollcommand=yscroll.set,
            xscrollcommand=xscroll.set
        )
        yscroll.config(command=tv.yview)
        xscroll.config(command=tv.xview)

        yscroll.pack(side="right", fill="y")
        xscroll.pack(side="bottom", fill="x")
        tv.pack(side="left", fill="both", expand=True)

        # headings
        tv.heading("__row__", text="Row(1-based)")
        tv.column("__row__", width=90, anchor="center")

        for c in show_cols:
            tv.heading(c, text=c)
            tv.column(c, width=160, anchor="w")

        # insert rows
        def safe_str(v):
            if pd.isna(v):
                return ""
            s = str(v)
            if len(s) > 200:
                return s[:200] + "…"
            return s

        base_row_1 = self._unreal_preview_offset + 1
        for i in range(n_show):
            row_1based = base_row_1 + i
            r = self._unreal_preview_df.iloc[i]
            values = [row_1based] + [safe_str(r.get(c, "")) for c in show_cols]
            tv.insert("", "end", iid=str(row_1based), values=values)

        # selection update
        def update_sel_label(*_):
            sel = tv.selection()
            if not sel:
                sel_var.set("선택된 행: 없음")
                return
            rows = sorted(int(x) for x in sel)
            sel_var.set(f"선택된 행: {rows[0]} ~ {rows[-1]}  (총 {len(rows)}개)")

        tv.bind("<<TreeviewSelect>>", update_sel_label)

        # buttons
        bottom = tk.Frame(win)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        def apply_selection():
            sel = tv.selection()
            if not sel:
                messagebox.showwarning("선택 없음", "표에서 행 범위를 선택하세요. (마우스로 드래그/Shift/Ctrl 가능)")
                return
            rows = sorted(int(x) for x in sel)

            self.unreal_row_start.set(str(rows[0]))
            self.unreal_row_end.set(str(rows[-1]))
            self._log(f"[GUI] Unreal row range set by preview: {rows[0]} ~ {rows[-1]} (selected {len(rows)} rows)\n")
            win.destroy()

        def clear_selection():
            tv.selection_remove(tv.selection())
            sel_var.set("선택된 행: 없음")

        ttk.Button(bottom, text="선택 적용 (Start/End에 반영)", command=apply_selection).pack(side="right", padx=(6, 0))
        ttk.Button(bottom, text="선택 해제", command=clear_selection).pack(side="right")

    # ---------- pickers ----------
    def pick_ocr_csv(self):
        path = filedialog.askopenfilename(
            title="OCR CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.ocr_csv_path.set(path)
            self._log(f"[GUI] OCR CSV: {path}\n")

    def pick_video(self):
        path = filedialog.askopenfilename(
            title="Video 선택",
            filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            self._log(f"[GUI] Video: {path}\n")

    def pick_events(self):
        path = filedialog.askopenfilename(
            title="Cheetah Events 선택",
            filetypes=[("Excel", "*.xlsx *.xls"), ("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.events_path.set(path)
            self._log(f"[GUI] Events: {path}\n")

    def pick_unreal(self):
        path = filedialog.askopenfilename(
            title="Unreal Log 선택",
            filetypes=[("CSV", "*.csv"), ("Excel", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        if path:
            self.unreal_path.set(path)
            self._log(f"[GUI] Unreal Log: {path}\n")
            self._refresh_unreal_column_dropdowns(path)

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="저장 폴더 선택")
        if path:
            self.out_dir.set(path)
            self._log(f"[GUI] Output folder: {path}\n")

    # ---------- run ----------
    def on_run(self):
        if self._worker_thread and self._worker_thread.is_alive():
            messagebox.showwarning("진행 중", "이미 실행 중입니다.")
            return

        ocr_csv = self.ocr_csv_path.get().strip()
        video   = self.video_path.get().strip()
        events  = self.events_path.get().strip()
        unreal  = self.unreal_path.get().strip()
        out_dir = self.out_dir.get().strip()

        if not (ocr_csv and os.path.exists(ocr_csv)):
            messagebox.showerror("오류", "OCR CSV 파일을 선택하세요.")
            return
        if not (video and os.path.exists(video)):
            messagebox.showerror("오류", "Video 파일을 선택하세요.")
            return
        if not (events and os.path.exists(events)):
            messagebox.showerror("오류", "Cheetah Events 파일을 선택하세요.")
            return
        if not (unreal and os.path.exists(unreal)):
            messagebox.showerror("오류", "Unreal Log 파일을 선택하세요.")
            return
        if not (out_dir and os.path.isdir(out_dir)):
            messagebox.showerror("오류", "저장 폴더를 선택하세요.")
            return

        merged_csv_name = os.path.basename(OUTPUT_MERGED_CSV_PATH)
        clipped_vid_name = os.path.basename(CLIPPED_VIDEO_PATH)

        merged_csv_out = os.path.join(out_dir, merged_csv_name)
        clipped_vid_out = os.path.join(out_dir, clipped_vid_name)

        opt_true_col  = self.unreal_true_col_name.get().strip()
        opt_trial_col = self.unreal_trial_col_name.get().strip()
        opt_rs = self.unreal_row_start.get().strip()
        opt_re = self.unreal_row_end.get().strip()

        self.btn_run.config(state="disabled")
        self.status_var.set("실행 중...")
        self._set_progress(0.0)
        self._log("\n[RUN]\n")
        self._log(f" - merged_csv_out: {merged_csv_out}\n")
        self._log(f" - clipped_video_out: {clipped_vid_out}\n")
        self._log("[Unreal Align 옵션]\n")
        self._log(f" - TRUE col : {opt_true_col if opt_true_col else '(default by letter S)'}\n")
        self._log(f" - TRIAL col: {opt_trial_col if opt_trial_col else '(default by letter F)'}\n")
        self._log(f" - rows     : start={opt_rs if opt_rs else '(all)'} , end={opt_re if opt_re else '(all)'}\n\n")

        self._worker_thread = threading.Thread(
            target=self._run_pipeline_safe,
            args=(ocr_csv, video, events, unreal, merged_csv_out, clipped_vid_out,
                  opt_true_col, opt_trial_col, opt_rs, opt_re),
            daemon=True
        )
        self._worker_thread.start()

    def _run_pipeline_safe(self, ocr_csv, video, events, unreal, merged_csv_out, clipped_vid_out,
                           opt_true_col, opt_trial_col, opt_rs, opt_re):
        try:
            t0 = time.time()
            self._run_pipeline(ocr_csv, video, events, unreal, merged_csv_out, clipped_vid_out,
                               opt_true_col, opt_trial_col, opt_rs, opt_re)
            dt = (time.time() - t0) / 60.0
            self._set_progress(100.0)
            self.status_var.set("완료")
            self._log(f"\n[DONE] 총 소요 시간: {dt:.2f}분\n")
            messagebox.showinfo("완료", "처리가 완료되었습니다.")
        except Exception:
            self.status_var.set("오류 발생")
            self._log("\n[ERROR]\n" + traceback.format_exc() + "\n")
            messagebox.showerror("오류", "오류가 발생했습니다. 로그를 확인하세요.")
        finally:
            self.btn_run.config(state="normal")

    # =========================================================
    # 파이프라인
    # =========================================================
    def _run_pipeline(self, OCR_CSV_PATH, VIDEO_PATH, EVENTS_PATH, UNREAL_LOG_PATH,
                      OUTPUT_MERGED_CSV_PATH, CLIPPED_VIDEO_PATH,
                      opt_true_col, opt_trial_col, opt_rs, opt_re):

        # 1) OCR CSV
        self._log("[STEP 1] OCR CSV 읽는 중...\n")
        self._set_progress(5)

        df_ocr = read_table_auto(OCR_CSV_PATH)

        if "index" not in df_ocr.columns:
            raise ValueError("OCR CSV에 'index' 컬럼이 없습니다.")
        if "timestamp" not in df_ocr.columns and "timestamp_str" not in df_ocr.columns:
            raise ValueError("OCR CSV에 'timestamp' 또는 'timestamp_str' 컬럼이 없습니다.")

        df_ocr.sort_values(by="index", inplace=True)
        df_ocr.reset_index(drop=True, inplace=True)

        base_ts_col = "timestamp" if "timestamp" in df_ocr.columns else "timestamp_str"
        df_ocr["timestamp_str"] = df_ocr[base_ts_col].astype(str)
        df_ocr["timestamp_int"] = df_ocr["timestamp_str"].apply(
            lambda s: int(normalize_digits(s)) if normalize_digits(s) else np.nan
        )
        df_ocr.dropna(subset=["timestamp_int"], inplace=True)
        df_ocr["timestamp_int"] = df_ocr["timestamp_int"].astype(np.int64)

        self._log(f"[INFO] OCR CSV 로드 완료. 유효 프레임 수: {len(df_ocr)}\n")
        self._set_progress(15)

        # 2) Events
        self._log("[STEP 2] Events 파일 읽는 중...\n")
        self._set_progress(18)

        df_events = read_table_auto(EVENTS_PATH)

        if CSV_TS_COL not in df_events.columns or CSV_INFO_COL not in df_events.columns:
            raise ValueError(f"Events 파일에 '{CSV_TS_COL}' 또는 '{CSV_INFO_COL}' 컬럼이 없습니다.")

        df_events[CSV_TS_COL] = df_events[CSV_TS_COL].apply(
            lambda s: int(normalize_digits(s)) if pd.notna(s) and normalize_digits(s) else np.nan
        )
        df_events.dropna(subset=[CSV_TS_COL], inplace=True)
        df_events[CSV_TS_COL] = df_events[CSV_TS_COL].astype(np.int64)

        csv_all_ts_int = df_events[CSV_TS_COL].to_numpy().astype(np.int64)
        csv_all_ts_sorted = np.sort(csv_all_ts_int)

        self._log(f"[INFO] Events 로드 완료. 총 row 수: {len(df_events)}\n")
        self._set_progress(25)

        # 3) TTL 필터
        self._log("[STEP 3] TTL=0x0004 조건(row) 필터링...\n")

        info_series = df_events[CSV_INFO_COL].astype(str).str.strip()
        ttl_target = TTL_PATTERN.strip()
        df_ttl = df_events[info_series == ttl_target].copy()

        if df_ttl.empty:
            self._log("[DEBUG] Column18 unique values (first 20):\n")
            self._log("\n".join(map(str, info_series.unique()[:20])) + "\n")
            raise RuntimeError(
                f"[ERROR] {CSV_INFO_COL} 에서 '{ttl_target}' 과 정확히 일치하는 row가 없습니다."
            )

        ttl_ts_int = df_ttl[CSV_TS_COL].to_numpy().astype(np.int64)
        ttl_start_csv = int(ttl_ts_int[0])
        ttl_end_csv   = int(ttl_ts_int[-1])

        self._log(f"[INFO] TTL(0x0004) row 수: {len(ttl_ts_int)}\n")
        self._log(f"[INFO] TTL 시작 CSV ts: {ttl_start_csv}\n")
        self._log(f"[INFO] TTL 종료 CSV ts: {ttl_end_csv}\n")
        self._set_progress(32)


        self._log("[STEP 3-1] Unreal log에서 trial 추출...\n")

        df_unreal = read_table_auto(UNREAL_LOG_PATH)


        n_total = len(df_unreal)
        rs = None
        re_ = None
        if opt_rs:
            if not opt_rs.isdigit() or int(opt_rs) <= 0:
                raise ValueError("Unreal 행 Start는 1 이상의 정수여야 합니다.")
            rs = int(opt_rs)
        if opt_re:
            if not opt_re.isdigit() or int(opt_re) <= 0:
                raise ValueError("Unreal 행 End는 1 이상의 정수여야 합니다.")
            re_ = int(opt_re)

        if rs is not None or re_ is not None:
            start0 = (rs - 1) if rs is not None else 0
            end0_inclusive = (re_ - 1) if re_ is not None else (n_total - 1)
            if start0 > end0_inclusive:
                raise ValueError("Unreal 행 범위가 잘못되었습니다: Start > End")
            start0 = max(0, min(start0, n_total - 1))
            end0_inclusive = max(0, min(end0_inclusive, n_total - 1))
            df_unreal = df_unreal.iloc[start0:end0_inclusive + 1].copy()
            self._log(f"[INFO] Unreal 행 범위 적용: {start0+1} ~ {end0_inclusive+1} (총 {len(df_unreal)}행)\n")
        else:
            self._log(f"[INFO] Unreal 행 범위: 전체 사용 (총 {len(df_unreal)}행)\n")

        # (B) 컬럼 선택
        cols = list(df_unreal.columns.astype(str))

        if opt_true_col and opt_true_col in cols:
            true_col_name = opt_true_col
        else:
            true_idx = excel_col_letter_to_0based(UNREAL_TRUE_COL_LETTER)
            if true_idx >= len(cols):
                raise ValueError(f"Unreal log TRUE 기본 레터({UNREAL_TRUE_COL_LETTER}) 인덱스가 컬럼 범위를 초과합니다.")
            true_col_name = cols[true_idx]

        if opt_trial_col and opt_trial_col in cols:
            trial_col_name = opt_trial_col
        else:
            trial_idx = excel_col_letter_to_0based(UNREAL_TRIAL_COL_LETTER)
            if trial_idx >= len(cols):
                raise ValueError(f"Unreal log TRIAL 기본 레터({UNREAL_TRIAL_COL_LETTER}) 인덱스가 컬럼 범위를 초과합니다.")
            trial_col_name = cols[trial_idx]

        self._log(f"[INFO] Unreal TRUE 컬럼: {true_col_name}\n")
        self._log(f"[INFO] Unreal TRIAL 컬럼: {trial_col_name}\n")

        # TRUE 필터
        df_unreal_true = df_unreal[df_unreal[true_col_name].astype(str).str.upper().isin(["TRUE", "1", "T", "YES"])].copy()
        if df_unreal_true.empty:
            raise RuntimeError("[ERROR] Unreal log에서 TRUE 조건(row)이 0개입니다. TRUE 컬럼/값 또는 범위를 확인하세요.")

        df_unreal_true["current_trial"] = df_unreal_true[trial_col_name]

        trial_counts = (
            df_unreal_true.groupby("current_trial", sort=False)
            .size()
            .reset_index(name="unreal_count")
        )

        trial_seq_raw = df_unreal_true["current_trial"].tolist()
        trial_order = []
        prev = object()
        for t in trial_seq_raw:
            if t != prev:
                trial_order.append(t)
                prev = t

        count_map = dict(zip(trial_counts["current_trial"], trial_counts["unreal_count"]))
        trial_plan = [(t, int(count_map.get(t, 0))) for t in trial_order]
        trial_plan = [(t, c) for (t, c) in trial_plan if c > 0]

        if not trial_plan:
            raise RuntimeError("[ERROR] Unreal log에서 trial count가 모두 0입니다. TRIAL 컬럼/필터를 확인하세요.")

        total_unreal = sum(c for _, c in trial_plan)
        self._log(f"[INFO] Unreal TRUE rows: {len(df_unreal_true)}\n")
        self._log(f"[INFO] Unreal trial segments: {len(trial_plan)}\n")
        self._log(f"[INFO] Unreal total count(sum per trial): {total_unreal}\n")
        self._set_progress(40)

        # 3-2) TTL row에 trial 매핑
        self._log("[STEP 3-2] TTL rows에 Unreal trial을 순서대로 매핑...\n")

        ttl_n = len(ttl_ts_int)
        trial_assign = np.empty(ttl_n, dtype=object)

        cursor = 0
        for (t, cnt) in trial_plan:
            if cursor >= ttl_n:
                break
            take = min(cnt, ttl_n - cursor)
            trial_assign[cursor:cursor+take] = t
            cursor += take

        if cursor < ttl_n:
            last_trial = trial_plan[-1][0]
            trial_assign[cursor:] = last_trial
            self._log(f"[WARN] Unreal total({total_unreal}) < TTL({ttl_n}) => 남은 {ttl_n-cursor}개를 마지막 trial({last_trial})로 채움\n")
        elif cursor > ttl_n:
            self._log(f"[WARN] Unreal total({total_unreal}) > TTL({ttl_n}) => TTL 길이까지만 사용\n")

        df_ttl = df_ttl.reset_index(drop=True)
        df_ttl["current_trial"] = trial_assign
        self._set_progress(45)

        # 4) offset 자동 추정
        self._log("[STEP 4] offset 자동 추정...\n")

        video_ts_int = df_ocr["timestamp_int"].to_numpy().astype(np.int64)
        offset = estimate_offset(video_ts_int, csv_all_ts_int, sample_count=OFFSET_SAMPLE_COUNT)

        ttl_start_video_ts = int(ttl_start_csv - offset)
        ttl_end_video_ts   = int(ttl_end_csv   - offset)

        self._log(f"[OFFSET] median offset = {offset}\n")
        self._log(f"[INFO] 변환된 TTL 시작 video ts: {ttl_start_video_ts}\n")
        self._log(f"[INFO] 변환된 TTL 종료 video ts: {ttl_end_video_ts}\n")
        self._set_progress(52)

        # 5) OCR start/end frame
        self._log("[STEP 5] 전체 시작/끝 프레임 index 찾기...\n")

        start_frame_idx = find_nearest_frame_for_ts(ttl_start_video_ts, df_ocr)
        end_frame_idx   = find_nearest_frame_for_ts(ttl_end_video_ts, df_ocr)

        self._log(f"[MATCH] start_frame={start_frame_idx}, end_frame={end_frame_idx}\n")

        if end_frame_idx <= start_frame_idx:
            raise RuntimeError(
                f"[ERROR] 종료 프레임({end_frame_idx})이 시작 프레임({start_frame_idx})보다 앞입니다."
            )

        df_clip_all = df_ocr[(df_ocr["index"] >= start_frame_idx) & (df_ocr["index"] <= end_frame_idx)].copy()
        df_clip_all.reset_index(drop=True, inplace=True)

        actual_frames_all = int(end_frame_idx - start_frame_idx + 1)
        expected_frames_all = int(len(ttl_ts_int))
        diff_all = expected_frames_all - actual_frames_all

        self._log(f"[INFO] actual_frames(OCR clip all)   = {actual_frames_all}\n")
        self._log(f"[INFO] expected_frames(TTL rows all) = {expected_frames_all}\n")
        self._log(f"[INFO] diff(TTL - OCR)               = {diff_all}\n")
        self._set_progress(60)

        # 6) trial별 OCR 매칭 + insert/drop 계획
        self._log("[STEP 6] trial 단위 OCR 구간 매칭 + 보강 계획 생성...\n")

        ttl_trials = []
        start_i = 0
        while start_i < ttl_n:
            t = trial_assign[start_i]
            end_i = start_i
            while end_i < ttl_n and trial_assign[end_i] == t:
                end_i += 1
            ttl_trials.append((t, start_i, end_i))
            start_i = end_i

        self._log(f"[INFO] TTL trial segments: {len(ttl_trials)}\n")

        trial_ocr_ranges = []
        for (t, a, b) in ttl_trials:
            ts_start_csv = int(ttl_ts_int[a])
            ts_end_csv   = int(ttl_ts_int[b-1])
            ts_start_vid = ts_start_csv - offset
            ts_end_vid   = ts_end_csv   - offset

            f_start = find_nearest_frame_for_ts(int(ts_start_vid), df_clip_all)
            f_end   = find_nearest_frame_for_ts(int(ts_end_vid),   df_clip_all)

            clip_start_pos = int(f_start - start_frame_idx)
            clip_end_pos   = int(f_end   - start_frame_idx)

            if clip_end_pos < clip_start_pos:
                clip_start_pos, clip_end_pos = clip_end_pos, clip_start_pos

            clip_start_pos = max(0, min(clip_start_pos, len(df_clip_all)-1))
            clip_end_pos   = max(0, min(clip_end_pos,   len(df_clip_all)-1))

            trial_ocr_ranges.append((t, a, b, clip_start_pos, clip_end_pos))

        final_clip_pos_list = []

        for (t, a, b, cs, ce) in trial_ocr_ranges:
            expected = int(b - a)
            df_trial_ocr = df_clip_all.iloc[cs:ce+1].copy()
            df_trial_ocr.reset_index(drop=True, inplace=True)
            actual = len(df_trial_ocr)

            insert_counter, drop_counter = plan_insert_drop_for_trial(
                df_trial_ocr=df_trial_ocr,
                expected_frames=expected,
                csv_all_ts_sorted=csv_all_ts_sorted,
                offset=offset
            )

            base_positions = list(range(actual))
            fixed_positions = apply_insert_drop_to_positions(base_positions, insert_counter, drop_counter)

            # 길이 강제 맞춤(기존 로직 유지)
            if len(fixed_positions) != expected:
                delta = expected - len(fixed_positions)
                if delta > 0:
                    extra = evenly_spaced_positions(delta, len(fixed_positions) if len(fixed_positions) > 0 else 1)
                    if len(fixed_positions) == 0 and actual > 0:
                        fixed_positions = [0] * expected
                    else:
                        for k in extra.tolist():
                            k = int(np.clip(k, 0, len(fixed_positions)-1))
                            fixed_positions.insert(k+1, fixed_positions[k])
                elif delta < 0:
                    dropn = -delta
                    drop_idx = sorted(set(evenly_spaced_positions(dropn, len(fixed_positions)).tolist()), reverse=True)
                    for di in drop_idx:
                        if 0 <= di < len(fixed_positions):
                            fixed_positions.pop(di)

            fixed_clip_pos = [cs + p for p in fixed_positions]
            final_clip_pos_list.extend(fixed_clip_pos)

        if len(final_clip_pos_list) != ttl_n:
            self._log(f"[WARN] final_clip_pos_list({len(final_clip_pos_list)}) != TTL({ttl_n}) => 전체 길이 보정\n")
            if len(final_clip_pos_list) < ttl_n:
                need = ttl_n - len(final_clip_pos_list)
                last = final_clip_pos_list[-1] if final_clip_pos_list else 0
                final_clip_pos_list.extend([last]*need)
            else:
                final_clip_pos_list = final_clip_pos_list[:ttl_n]

        assert len(final_clip_pos_list) == ttl_n
        self._set_progress(72)

        # 7) merged CSV 저장
        self._log("[STEP 7] merged CSV 생성...\n")

        ocr_ts_filled = []
        ocr_frame_filled = []
        trial_filled = []

        for i, clip_pos in enumerate(final_clip_pos_list):
            clip_pos = int(np.clip(clip_pos, 0, len(df_clip_all)-1))
            row = df_clip_all.iloc[clip_pos]
            ocr_ts_filled.append(int(row["timestamp_int"]))
            ocr_frame_filled.append(int(row["index"]))
            trial_filled.append(trial_assign[i])

        df_out = pd.DataFrame({
            "out_idx": np.arange(ttl_n, dtype=int),
            "cheetah_timestamp": ttl_ts_int.astype(np.int64),
            "current_trial": trial_filled,
            "ocr_timestamp_filled": np.array(ocr_ts_filled, dtype=np.int64),
            "video_frame_src": np.array(ocr_frame_filled, dtype=np.int64),
        })

        df_out.to_csv(OUTPUT_MERGED_CSV_PATH, index=False, encoding="utf-8-sig")
        self._log(f"[OK] merged CSV 저장: {os.path.abspath(OUTPUT_MERGED_CSV_PATH)}\n")
        self._set_progress(80)

        # 8) 영상 출력
        self._log("[STEP 8] 영상 출력(30fps) ...\n")

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise FileNotFoundError(f"영상 파일을 열 수 없습니다: {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(CLIPPED_VIDEO_PATH, fourcc, fps, (width, height))

        video_frames_to_write = [
            int(df_clip_all.iloc[int(np.clip(p, 0, len(df_clip_all)-1))]["index"])
            for p in final_clip_pos_list
        ]

        written = 0
        cur_frame_no = None
        cur_frame_img = None

        total = max(1, len(video_frames_to_write))
        base_p = 80.0
        span_p = 19.0

        for k, target_frame_no in enumerate(video_frames_to_write):
            target_frame_no = int(target_frame_no)

            if cur_frame_no is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
                ret, cur_frame_img = cap.read()
                if not ret:
                    self._log("[WARN] 첫 프레임을 읽지 못해 종료합니다.\n")
                    break
                cur_frame_no = target_frame_no

            if target_frame_no < cur_frame_no:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_no)
                ret, cur_frame_img = cap.read()
                if not ret:
                    self._log("[WARN] 점프 후 프레임을 읽지 못해 종료합니다.\n")
                    break
                cur_frame_no = target_frame_no

            while cur_frame_no < target_frame_no:
                ret, cur_frame_img = cap.read()
                if not ret:
                    self._log("[WARN] 비디오 프레임을 더 읽지 못했습니다. 조기 종료합니다.\n")
                    break
                cur_frame_no += 1

            if cur_frame_img is None:
                break

            out.write(cur_frame_img)
            written += 1
            self._set_progress(base_p + span_p * (k + 1) / total)

        cap.release()
        out.release()

        self._log("\n[DONE] 전체 완료\n")
        self._log(f" - merged_csv: {os.path.abspath(OUTPUT_MERGED_CSV_PATH)}\n")
        self._log(f" - output_video: {os.path.abspath(CLIPPED_VIDEO_PATH)}\n")
        self._log(f" - fps: {fps}\n")
        self._log(f" - TTL rows (expected): {ttl_n}\n")
        self._log(f" - written frames: {written}\n")

        if written != ttl_n:
            self._log("[WARN] written != TTL rows\n")
        else:
            self._log("[OK] written == TTL rows (정확히 일치)\n")


def _main_trial_align():
    root = tk.Tk()
    app = TrialAlignGuiApp(root)
    root.mainloop()


# =========================
# stop_run_print.py
# =========================

import os
import threading
import traceback
import time

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


def flatten_col(col):
    if isinstance(col, tuple):
        return "_".join([str(c) for c in col if c != ""])
    return str(col)


def run_pipeline(
    circular_csv: str,
    dlc_csv: str,
    output_csv: str,
    fps: int,
    stop_dist: float,
    stop_min_sec: float,
    window_smooth: int,
    video_start_from_sensor=None,
    log_fn=None,
    progress_fn=None,
):
 
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(p):
        if progress_fn:
            progress_fn(p)

    t0 = time.time()

    prog(5)
    log("[1/6] Circular(Unreal) CSV 로딩...\n")
    circular = pd.read_csv(circular_csv, header=None)

    time_col_idx = 0
    flag_col_idx = 18  
    trial_now_col_idx = 5  
    trial_cum_col_idx = 6  

    circular_valid = circular[circular.iloc[:, flag_col_idx] == True].copy()
    sensor_trials = circular_valid.iloc[:, [time_col_idx, trial_now_col_idx, trial_cum_col_idx]].copy()
    sensor_trials.columns = ["time", "trial_F", "trial_G"]

    sensor_trials["time"] = pd.to_numeric(sensor_trials["time"], errors="coerce")
    sensor_trials = sensor_trials.dropna(subset=["time"])
    sensor_trials = sensor_trials.sort_values("time").reset_index(drop=True)

    if len(sensor_trials) == 0:
        raise RuntimeError("Circular 파일에서 S==True 이면서 time이 유효한 행이 0개입니다.")

    if video_start_from_sensor is None:
        video_start_from_sensor = sensor_trials["time"].iloc[0]

    log(f" - sensor_trials rows (S==True): {len(sensor_trials)}\n")
    log(f" - VIDEO_START_FROM_SENSOR: {video_start_from_sensor}\n")


    prog(20)
    log("[2/6] DLC CSV 로딩...\n")
    dlc = pd.read_csv(dlc_csv, header=[0, 1, 2])

    scorer_name = dlc.columns[1][0]

    LEFT_BP = "left_leg"
    RIGHT_BP = "right_leg"

    # 좌표 추출
    x_left = dlc[(scorer_name, LEFT_BP, "x")].values
    y_left = dlc[(scorer_name, LEFT_BP, "y")].values
    x_right = dlc[(scorer_name, RIGHT_BP, "x")].values
    y_right = dlc[(scorer_name, RIGHT_BP, "y")].values

    x_mean = (x_left + x_right) / 2.0
    y_mean = (y_left + y_right) / 2.0

    n_frames = len(x_mean)
    frames = np.arange(n_frames)

    # 프레임 간 이동거리
    dist = np.sqrt(
        np.diff(x_mean, prepend=x_mean[0]) ** 2 +
        np.diff(y_mean, prepend=y_mean[0]) ** 2
    )

    # 이동평균
    dist_smooth = pd.Series(dist).rolling(window_smooth, min_periods=1).mean()

    # 1차 stop/run
    base_run = dist_smooth >= stop_dist
    state = base_run.astype(int).values  # 1=run, 0=stop 후보

    # 2차: 최소 stop_min_sec 이상 지속되어야 stop 인정
    min_frames = int(stop_min_sec * fps)
    start_idx = None
    for i, is_run in enumerate(base_run):
        if (not is_run) and start_idx is None:
            start_idx = i

        if (is_run or i == len(base_run) - 1) and start_idx is not None:
            end_idx = i if is_run else i + 1
            length = end_idx - start_idx
            if length < min_frames:
                state[start_idx:end_idx] = 1
            start_idx = None

    log(f" - DLC frames: {n_frames}\n")
    log(f" - min_frames(stop confirm): {min_frames}\n")

 
    prog(45)
    log("[3/6] meta 컬럼 추가...\n")
    dlc_time = float(video_start_from_sensor) + frames / float(fps)

    dlc_out = dlc.copy()
    dlc_out[("meta", "leg_center_x", "")] = x_mean
    dlc_out[("meta", "leg_center_y", "")] = y_mean
    dlc_out[("meta", "dist", "")] = dist
    dlc_out[("meta", "dist_smooth", "")] = dist_smooth.values
    dlc_out[("meta", "stop_run", "")] = state
    dlc_out[("meta", "frame_index", "")] = frames
    dlc_out[("meta", "time", "")] = dlc_time


    prog(60)
    log("[4/6] 컬럼 flatten...\n")
    dlc_flat = dlc_out.copy()
    dlc_flat.columns = [flatten_col(c) for c in dlc_flat.columns]

    dlc_flat["meta_time"] = pd.to_numeric(dlc_flat["meta_time"], errors="coerce")
    dlc_flat["meta_frame_index"] = pd.to_numeric(dlc_flat["meta_frame_index"], errors="coerce")
    dlc_flat = dlc_flat.dropna(subset=["meta_time"])


    prog(75)
    log("[5/6] trial merge_asof...\n")
    sensor_trials_sorted = sensor_trials.sort_values("time")
    dlc_sorted = dlc_flat.sort_values("meta_time")

    aligned = pd.merge_asof(
        dlc_sorted,
        sensor_trials_sorted,
        left_on="meta_time",
        right_on="time",
        direction="nearest",
        tolerance=1 / float(fps),
    )

    aligned = aligned.sort_values("meta_frame_index").reset_index(drop=True)


    prog(90)
    log("[6/6] CSV 저장...\n")
    aligned_out = aligned.drop(columns=["meta_time", "time"], errors="ignore")
    aligned_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    prog(100)
    dt = (time.time() - t0)
    log(f"\n[DONE] 저장 완료: {output_csv}\n")
    log(f"[DONE] 소요 시간: {dt:.2f} sec\n")



class StopRunApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DLC + Unreal(trial) Align GUI")
        self.root.geometry("1100x760")

        self.circular_path = tk.StringVar(value="")
        self.dlc_path = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="")

        self.fps_var = tk.StringVar(value="30")
        self.stop_dist_var = tk.StringVar(value="1.0")
        self.stop_min_sec_var = tk.StringVar(value="0.25")
        self.window_smooth_var = tk.StringVar(value="3")

        self.status_var = tk.StringVar(value="대기 중")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.worker = None

        self._build()

    def _build(self):
        top = ttk.LabelFrame(self.root, text="파일 선택")
        top.pack(fill="x", padx=12, pady=10)

        row_btn = tk.Frame(top)
        row_btn.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Button(row_btn, text="Unreal log CSV 선택", command=self.pick_circular).pack(side="left", padx=(0, 8))
        ttk.Button(row_btn, text="DLC CSV 선택", command=self.pick_dlc).pack(side="left", padx=(0, 8))
        ttk.Button(row_btn, text="저장 폴더 선택", command=self.pick_outdir).pack(side="left", padx=(0, 8))

        row_paths = tk.Frame(top)
        row_paths.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(row_paths, text="Unreal:").grid(row=0, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.circular_path).grid(row=0, column=1, sticky="w")

        ttk.Label(row_paths, text="DLC:").grid(row=1, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.dlc_path).grid(row=1, column=1, sticky="w")

        ttk.Label(row_paths, text="Output folder:").grid(row=2, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.out_dir).grid(row=2, column=1, sticky="w")

        row_paths.columnconfigure(1, weight=1)

        opt = ttk.LabelFrame(self.root, text="파라미터")
        opt.pack(fill="x", padx=12, pady=(0, 10))

        row_opt = tk.Frame(opt)
        row_opt.pack(fill="x", padx=10, pady=10)

        def add_labeled_entry(parent, label, var, w=10):
            f = tk.Frame(parent)
            ttk.Label(f, text=label).pack(side="left")
            ttk.Entry(f, textvariable=var, width=w).pack(side="left", padx=(6, 0))
            return f

        add_labeled_entry(row_opt, "FPS", self.fps_var, 8).pack(side="left", padx=(0, 18))
        add_labeled_entry(row_opt, "STOP_DIST", self.stop_dist_var, 8).pack(side="left", padx=(0, 18))
        add_labeled_entry(row_opt, "STOP_MIN_SEC", self.stop_min_sec_var, 8).pack(side="left", padx=(0, 18))
        add_labeled_entry(row_opt, "WINDOW_SMOOTH", self.window_smooth_var, 8).pack(side="left", padx=(0, 18))

        runbar = tk.Frame(self.root)
        runbar.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_run = ttk.Button(runbar, text="실행", command=self.on_run)
        self.btn_run.pack(side="left", padx=(0, 10))

        ttk.Label(runbar, textvariable=self.status_var).pack(side="left")

        prog = ttk.LabelFrame(self.root, text="진행률")
        prog.pack(fill="x", padx=12, pady=8)

        self.pb = ttk.Progressbar(prog, variable=self.progress_var, maximum=100.0)
        self.pb.pack(fill="x", padx=10, pady=10)

        lf_log = ttk.LabelFrame(self.root, text="로그")
        lf_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log = scrolledtext.ScrolledText(lf_log, height=24)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("[READY] 파일 선택 후 실행하세요.\n")

    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")
        self.root.update_idletasks()

    def _set_progress(self, p: float):
        p = max(0.0, min(100.0, float(p)))
        self.progress_var.set(p)
        self.root.update_idletasks()

    def pick_circular(self):
        path = filedialog.askopenfilename(
            title="Circular(Unreal) CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.circular_path.set(path)
            self._log(f"[GUI] Circular: {path}\n")

    def pick_dlc(self):
        path = filedialog.askopenfilename(
            title="DLC CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.dlc_path.set(path)
            self._log(f"[GUI] DLC: {path}\n")

    def pick_outdir(self):
        path = filedialog.askdirectory(title="저장 폴더 선택")
        if path:
            self.out_dir.set(path)
            self._log(f"[GUI] Output folder: {path}\n")

    def on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("진행 중", "이미 실행 중입니다.")
            return

        circular = self.circular_path.get().strip()
        dlc = self.dlc_path.get().strip()
        out_dir = self.out_dir.get().strip()

        if not (circular and os.path.exists(circular)):
            messagebox.showerror("오류", "Unreal log CSV를 선택하세요.")
            return
        if not (dlc and os.path.exists(dlc)):
            messagebox.showerror("오류", "DLC CSV를 선택하세요.")
            return
        if not (out_dir and os.path.isdir(out_dir)):
            messagebox.showerror("오류", "저장 폴더를 선택하세요.")
            return

        # 파라미터 파싱
        try:
            fps = int(float(self.fps_var.get().strip()))
            stop_dist = float(self.stop_dist_var.get().strip())
            stop_min_sec = float(self.stop_min_sec_var.get().strip())
            window_smooth = int(float(self.window_smooth_var.get().strip()))
        except Exception:
            messagebox.showerror("오류", "파라미터 값이 올바르지 않습니다. (숫자만 입력)")
            return

        base_name = os.path.splitext(os.path.basename(dlc))[0]
        out_csv = os.path.join(out_dir, f"{base_name}_with_stop_run_and_trials.csv")

        self.btn_run.config(state="disabled")
        self.status_var.set("실행 중...")
        self._set_progress(0)
        self._log("\n[RUN]\n")
        self._log(f" - circular: {circular}\n")
        self._log(f" - dlc: {dlc}\n")
        self._log(f" - out_csv: {out_csv}\n")
        self._log(f" - fps={fps}, STOP_DIST={stop_dist}, STOP_MIN_SEC={stop_min_sec}, WINDOW_SMOOTH={window_smooth}\n\n")

        def worker_fn():
            try:
                run_pipeline(
                    circular_csv=circular,
                    dlc_csv=dlc,
                    output_csv=out_csv,
                    fps=fps,
                    stop_dist=stop_dist,
                    stop_min_sec=stop_min_sec,
                    window_smooth=window_smooth,
                    video_start_from_sensor=None,
                    log_fn=self._log,
                    progress_fn=self._set_progress,
                )
                self.status_var.set("완료")
                messagebox.showinfo("완료", f"저장 완료:\n{out_csv}")
            except Exception:
                self.status_var.set("오류")
                self._log("\n[ERROR]\n" + traceback.format_exc() + "\n")
                messagebox.showerror("오류", "오류가 발생했습니다. 로그를 확인하세요.")
            finally:
                self.btn_run.config(state="normal")

        self.worker = threading.Thread(target=worker_fn, daemon=True)
        self.worker.start()


def _main_stop_run():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


# =========================
# text_mp4.py
# =========================

import os
import threading
import traceback
import time

import cv2
import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# ==========================
# Core logic 
# ==========================
def overlay_runstop_trial(
    metrics_csv_path: str,
    merged_csv_path: str,
    input_video_path: str,
    output_video_path: str,
    log_fn=None,
    progress_fn=None,
):
    def log(msg: str):
        if log_fn:
            log_fn(msg)

    def prog(p: float):
        if progress_fn:
            progress_fn(p)

    t0 = time.time()

    # --------------------------
    # 1) stop_run CSV 로드
    # --------------------------
    prog(5)
    log("[1/5] stop_run CSV 로딩...\n")
    df_stop = pd.read_csv(metrics_csv_path, low_memory=False)

    if "scorer" in df_stop.columns:
        df_stop = df_stop[~df_stop["scorer"].astype(str).str.lower().isin(["bodyparts", "coords"])].copy()


    stop_col = None
    for cand in ["meta_stop_run", "stop_run"]:
        if cand in df_stop.columns:
            stop_col = cand
            break
    if stop_col is None:
        raise ValueError("1번 CSV에서 'meta_stop_run' 또는 'stop_run' 컬럼을 찾을 수 없습니다.")
    df_stop = df_stop.reset_index(drop=True)


    if "out_idx" in df_stop.columns:
        df_stop["out_idx"] = pd.to_numeric(df_stop["out_idx"], errors="coerce").astype("Int64")
        df_stop = df_stop.dropna(subset=["out_idx"]).copy()
        df_stop["out_idx"] = df_stop["out_idx"].astype(int)
    else:
        df_stop["out_idx"] = np.arange(len(df_stop), dtype=int)

    df_stop["stop_run"] = pd.to_numeric(df_stop[stop_col], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df_stop = df_stop[["out_idx", "stop_run"]].sort_values("out_idx").drop_duplicates("out_idx")

    log(f" - stop_run rows: {len(df_stop)} | out_idx min/max: {df_stop['out_idx'].min()}/{df_stop['out_idx'].max()}\n")

    # --------------------------
    # 2) merged CSV 로드 
    # --------------------------
    prog(20)
    log("[2/5] merged(trial) CSV 로딩...\n")
    df_trial = pd.read_csv(merged_csv_path, low_memory=False)

    if "current_trial" not in df_trial.columns:
        raise ValueError("2번 CSV에서 'current_trial' 컬럼을 찾을 수 없습니다.")

    df_trial = df_trial.reset_index(drop=True)

    if "out_idx" in df_trial.columns:
        df_trial["out_idx"] = pd.to_numeric(df_trial["out_idx"], errors="coerce").astype("Int64")
        df_trial = df_trial.dropna(subset=["out_idx"]).copy()
        df_trial["out_idx"] = df_trial["out_idx"].astype(int)
    else:
        df_trial["out_idx"] = np.arange(len(df_trial), dtype=int)

    df_trial["current_trial"] = pd.to_numeric(df_trial["current_trial"], errors="coerce")
    df_trial["current_trial"] = df_trial["current_trial"].ffill().bfill()

    df_trial = df_trial[["out_idx", "current_trial"]].sort_values("out_idx").drop_duplicates("out_idx")

    log(f" - trial rows: {len(df_trial)} | out_idx min/max: {df_trial['out_idx'].min()}/{df_trial['out_idx'].max()}\n")

    # --------------------------
    # 3) out_idx로 병합
    # --------------------------
    prog(35)
    log("[3/5] out_idx 병합...\n")
    df = df_trial.merge(df_stop, on="out_idx", how="left").sort_values("out_idx").reset_index(drop=True)
    df["stop_run"] = df["stop_run"].ffill().bfill().fillna(0).astype(int).clip(0, 1)

    log(f" - merged rows: {len(df)} | out_idx min/max: {df['out_idx'].min()}/{df['out_idx'].max()}\n")

    # --------------------------
    # 4) 비디오 오픈
    # --------------------------
    prog(45)
    log("[4/5] 비디오 오픈...\n")
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise IOError(f"영상을 열 수 없습니다: {input_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    log(f" - Video: fps={fps}, size={w}x{h}, total_frames={total}\n")

    max_frames = min(total, len(df))
    if total != len(df):
        log(f"[WARN] video frames({total}) != csv rows({len(df)}). 처리 프레임={max_frames} 로 제한합니다.\n")

    # --------------------------
    # 5) 프레임별 오버레이
    # --------------------------
    prog(55)
    log("[5/5] 프레임 오버레이 + 저장...\n")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    x0, y0 = 20, 45

    written = 0
    for frame_no in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        row = df.iloc[frame_no]
        trial_val = row["current_trial"]
        stop_val = int(row["stop_run"])

        trial_text = "-" if pd.isna(trial_val) else str(int(trial_val))
        state_str = "RUN" if stop_val == 1 else "STOP"
        text = f"Trial: {trial_text}  |  {state_str}  |  out_idx: {frame_no}"
        color = (0, 255, 0) if stop_val == 1 else (0, 0, 255)

        (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
        cv2.rectangle(frame, (x0 - 10, y0 - th - 12), (x0 + tw + 10, y0 + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (x0, y0), font, font_scale, color, thickness, cv2.LINE_AA)

        out.write(frame)
        written += 1

        # 진행률(55~100 구간)
        if max_frames > 0:
            prog(55 + 45 * (frame_no + 1) / max_frames)

    cap.release()
    out.release()

    dt = time.time() - t0
    log("\n[DONE]\n")
    log(f" - output: {output_video_path}\n")
    log(f" - written frames: {written}\n")
    log(f" - elapsed: {dt:.2f} sec\n")


# ==========================
# GUI 
# ==========================
class TextMp4App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Overlay (Trial + RUN/STOP) GUI")
        self.root.geometry("1100x760")


        self.metrics_path = tk.StringVar(value="")
        self.merged_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        # StopRunDiscrimination(StopRunApp)처럼 "폴더"를 선택하게 하고,
        # 실제 출력 mp4 경로는 선택된 폴더 + 자동 파일명으로 구성
        self.out_dir = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")  # 표시용(최종 mp4 전체 경로)

        self.status_var = tk.StringVar(value="대기 중")
        self.progress_var = tk.DoubleVar(value=0.0)

        self.worker = None

        self._build()

    def _build(self):
        # ---------- 파일 선택 ----------
        top = ttk.LabelFrame(self.root, text="파일 선택")
        top.pack(fill="x", padx=12, pady=10)

        row_btn = tk.Frame(top)
        row_btn.pack(fill="x", padx=10, pady=(10, 6))

        ttk.Button(row_btn, text="stop_run CSV 선택", command=self.pick_metrics).pack(side="left", padx=(0, 8))
        ttk.Button(row_btn, text="merged(trial) CSV 선택", command=self.pick_merged).pack(side="left", padx=(0, 8))
        ttk.Button(row_btn, text="입력 비디오 선택", command=self.pick_video).pack(side="left", padx=(0, 8))
        ttk.Button(row_btn, text="저장 폴더 선택", command=self.pick_out_dir).pack(side="left", padx=(0, 8))

        row_paths = tk.Frame(top)
        row_paths.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Label(row_paths, text="stop_run:").grid(row=0, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.metrics_path).grid(row=0, column=1, sticky="w")

        ttk.Label(row_paths, text="merged:").grid(row=1, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.merged_path).grid(row=1, column=1, sticky="w")

        ttk.Label(row_paths, text="video:").grid(row=2, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.video_path).grid(row=2, column=1, sticky="w")

        ttk.Label(row_paths, text="output:").grid(row=3, column=0, sticky="w")
        ttk.Label(row_paths, textvariable=self.output_path).grid(row=3, column=1, sticky="w")

        row_paths.columnconfigure(1, weight=1)

        # ---------- 실행 바 ----------
        runbar = tk.Frame(self.root)
        runbar.pack(fill="x", padx=12, pady=(0, 8))

        self.btn_run = ttk.Button(runbar, text="실행", command=self.on_run)
        self.btn_run.pack(side="left", padx=(0, 10))

        ttk.Label(runbar, textvariable=self.status_var).pack(side="left")

        # ---------- 진행률 ----------
        prog = ttk.LabelFrame(self.root, text="진행률")
        prog.pack(fill="x", padx=12, pady=8)

        self.pb = ttk.Progressbar(prog, variable=self.progress_var, maximum=100.0)
        self.pb.pack(fill="x", padx=10, pady=10)

        # ---------- 로그 ----------
        lf_log = ttk.LabelFrame(self.root, text="로그")
        lf_log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.log = scrolledtext.ScrolledText(lf_log, height=24)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)

        self._log("[READY] 파일 선택 후 실행하세요.\n")

    def _log(self, msg: str):
        self.log.insert("end", msg)
        self.log.see("end")
        self.root.update_idletasks()

    def _set_progress(self, p: float):
        p = max(0.0, min(100.0, float(p)))
        self.progress_var.set(p)
        self.root.update_idletasks()

    # ----- pickers -----
    def pick_metrics(self):
        path = filedialog.askopenfilename(
            title="stop_run CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.metrics_path.set(path)
            self._log(f"[GUI] stop_run CSV: {path}\n")

    def pick_merged(self):
        path = filedialog.askopenfilename(
            title="merged(trial) CSV 선택",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")]
        )
        if path:
            self.merged_path.set(path)
            self._log(f"[GUI] merged CSV: {path}\n")

    def pick_video(self):
        path = filedialog.askopenfilename(
            title="입력 비디오 선택",
            filetypes=[("MP4", "*.mp4"), ("All files", "*.*")]
        )
        if path:
            self.video_path.set(path)
            self._log(f"[GUI] video: {path}\n")

            # 저장 폴더를 이미 선택했다면, 새 비디오명 기준으로 자동 output mp4 갱신
            out_dir = self.out_dir.get().strip()
            if out_dir and os.path.isdir(out_dir):
                stem = os.path.splitext(os.path.basename(path))[0] if path else "output"
                base_name = f"{stem}_overlay.mp4"
                out_mp4 = os.path.join(out_dir, base_name)
                if os.path.exists(out_mp4):
                    i = 1
                    while True:
                        cand = os.path.join(out_dir, f"{stem}_overlay_{i}.mp4")
                        if not os.path.exists(cand):
                            out_mp4 = cand
                            break
                        i += 1
                self.output_path.set(out_mp4)
                self._log(f"[GUI] output(auto): {out_mp4}\n")

    def pick_output(self):
        path = filedialog.asksaveasfilename(
            title="출력 MP4 저장경로",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")]
        )
        if path:
            self.output_path.set(path)
            self._log(f"[GUI] output: {path}\n")

    def pick_out_dir(self):
        path = filedialog.askdirectory(title="저장 폴더 선택")
        if path:
            self.out_dir.set(path)
            # 입력 비디오명을 기반으로 자동 파일명 생성
            vid = self.video_path.get().strip()
            stem = os.path.splitext(os.path.basename(vid))[0] if vid else "output"
            base_name = f"{stem}_overlay.mp4"
            out_mp4 = os.path.join(path, base_name)
            # 덮어쓰기 방지: 동일 파일이 있으면 _1, _2 ...
            if os.path.exists(out_mp4):
                i = 1
                while True:
                    cand = os.path.join(path, f"{stem}_overlay_{i}.mp4")
                    if not os.path.exists(cand):
                        out_mp4 = cand
                        break
                    i += 1

            self.output_path.set(out_mp4)
            self._log(f"[GUI] Output folder: {path}\n")
            self._log(f"[GUI] output mp4     : {out_mp4}\n")

    # ----- run -----
    def on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("진행 중", "이미 실행 중입니다.")
            return

        metrics = self.metrics_path.get().strip()
        merged = self.merged_path.get().strip()
        video = self.video_path.get().strip()
        out_dir = self.out_dir.get().strip()
        out_mp4 = self.output_path.get().strip()

        if not (metrics and os.path.exists(metrics)):
            messagebox.showerror("오류", "stop_run CSV를 선택하세요.")
            return
        if not (merged and os.path.exists(merged)):
            messagebox.showerror("오류", "merged(trial) CSV를 선택하세요.")
            return
        if not (video and os.path.exists(video)):
            messagebox.showerror("오류", "입력 비디오(mp4)를 선택하세요.")
            return
        if (not out_dir) or (not os.path.isdir(out_dir)):
            messagebox.showerror("오류", "저장 폴더를 선택하세요.")
            return

        # 폴더만 선택된 경우를 대비해, 실행 시점에 최종 mp4 경로를 한번 더 보정
        if not out_mp4:
            stem = os.path.splitext(os.path.basename(video))[0]
            out_mp4 = os.path.join(out_dir, f"{stem}_overlay.mp4")
            self.output_path.set(out_mp4)

        self.btn_run.config(state="disabled")
        self.status_var.set("실행 중...")
        self._set_progress(0)
        self._log("\n[RUN]\n")
        self._log(f" - stop_run csv: {metrics}\n")
        self._log(f" - merged csv  : {merged}\n")
        self._log(f" - input video : {video}\n")
        self._log(f" - output mp4  : {out_mp4}\n\n")

        def worker_fn():
            try:
                overlay_runstop_trial(
                    metrics_csv_path=metrics,
                    merged_csv_path=merged,
                    input_video_path=video,
                    output_video_path=out_mp4,
                    log_fn=self._log,
                    progress_fn=self._set_progress,
                )
                self.status_var.set("완료")
                messagebox.showinfo("완료", f"저장 완료:\n{out_mp4}")
            except Exception:
                self.status_var.set("오류")
                self._log("\n[ERROR]\n" + traceback.format_exc() + "\n")
                messagebox.showerror("오류", "오류가 발생했습니다. 로그를 확인하세요.")
            finally:
                self.btn_run.config(state="normal")

        self.worker = threading.Thread(target=worker_fn, daemon=True)
        self.worker.start()


def _main_text_mp4():
    root = tk.Tk()
    App(root)
    root.mainloop()

# =========================
# Combined GUI
# =========================
import tkinter as tk
from tkinter import ttk
import multiprocessing as mp


def make_embed_root(frame: tk.Widget):
    noop = lambda *args, **kwargs: None

    for name in [
        "title", "geometry", "resizable", "iconbitmap", "protocol",
        "attributes", "state", "wm_attributes", "wm_title", "wm_geometry"
    ]:
        if not hasattr(frame, name):
            setattr(frame, name, noop)

    return frame


STEPS = [
    ("Timestamp Extraction", EasyOCRApp),
    ("Frame Reinforcement", TrialAlignGuiApp),
    ("Stop Run Discrimination", StopRunApp),
    ("Text Video", TextMp4App),
]


class CombinedGUIApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Assistance (by Leelab)")
        self.root.geometry("1200x780")

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=10, pady=10)

        for tab_name, cls in STEPS:
            tab = ttk.Frame(self.nb)
            tab.pack(fill="both", expand=True)
            self.nb.add(tab, text=tab_name)
            self._build_tab(tab, tab_name, cls)

    def _build_tab(self, tab: ttk.Frame, tab_name: str, cls):
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        container = ttk.Frame(tab)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(0, weight=1)
        container.rowconfigure(0, weight=1)

        embed_root = make_embed_root(container)

        try:
            cls(embed_root)
        except Exception as e:
            msg = f"[EMBED ERROR]\n{tab_name}\n\n{e}"
            lbl = tk.Label(container, text=msg, fg="red", justify="left")
            lbl.pack(anchor="nw", padx=12, pady=12)


def main():
    mp.freeze_support()
    root = tk.Tk()
    CombinedGUIApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
