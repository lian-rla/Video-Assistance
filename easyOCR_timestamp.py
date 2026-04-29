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
class App:
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

        # 모두 종료 
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

            # 저장 폴더는 이미 선택한 폴더라서 보통 존재하지만 안전하게
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


if __name__ == "__main__":
    mp.freeze_support()

    root = tk.Tk()


    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass

    app = App(root)
    root.mainloop()
