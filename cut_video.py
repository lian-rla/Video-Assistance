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

# =========================================================
# 기본 설정 (기존 값/조건 유지)
# =========================================================
OCR_CSV_PATH       = r"C:\Users\leelab\Desktop\11310106_timestamp_dualGPU.csv"
VIDEO_PATH         = r"F:\rat948\SEC_20240504_1219\20240504_mp4\12190106_0200.mp4"
EVENTS_PATH        = r"C:\Users\leelab\Desktop\Events.xlsx"
CLIPPED_VIDEO_PATH = r"C:\Users\leelab\Desktop\test2_.mp4"
UNREAL_LOG_PATH    = r"C:\Users\leelab\Desktop\2024.05.04-12.17.34.639_Circular_memory.csv"


UNREAL_TRUE_COL_LETTER  = "S"
UNREAL_TRIAL_COL_LETTER = "F"

CSV_TS_COL   = "Column4"
CSV_INFO_COL = "Column18"
TTL_PATTERN  = "TTL Input on AcqSystem1_0 board 0 port 2 value (0x0004)."

OFFSET_SAMPLE_COUNT = 200
OUTPUT_MERGED_CSV_PATH = r"C:\Users\leelab\Desktop\merged_cheetah_trial_ocr_with_fill.csv"


UNREAL_PREVIEW_MAX_ROWS = 5000


# =========================================================
# 유틸
# =========================================================
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
        self._unreal_df_cache = None          # 전체 DF 캐시(미리보기용)
        self._unreal_preview_df = None        # 미리보기 표에 실제 표시되는 DF(최대 N행)
        self._unreal_preview_offset = 0       # 현재 미리보기 DF가 원본에서 시작하는 row offset(지금은 0만 사용)

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
        # 앞쪽 최대 6개
        show_cols.extend(cols_all[:6])


        for c in [true_col, trial_col]:
            if c and c in cols_all and c not in show_cols:
                show_cols.append(c)


        show_cols = show_cols[:10]


        win = tk.Toplevel(self.root)
        win.title("Unreal Log 미리보기 - 마우스로 행 범위 선택")
        win.geometry("1100x650")

        top_info = tk.Frame(win)
        top_info.pack(fill="x", padx=10, pady=8)
        ttk.Label(top_info, text=f"전체 {n_total}행 중 상단 {n_show}행 표시").pack(side="left")

        sel_var = tk.StringVar(value="선택된 행: 없음")
        ttk.Label(top_info, textvariable=sel_var).pack(side="right")


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

        def update_sel_label(*_):
            sel = tv.selection()
            if not sel:
                sel_var.set("선택된 행: 없음")
                return
            rows = sorted(int(x) for x in sel)
            sel_var.set(f"선택된 행: {rows[0]} ~ {rows[-1]}  (총 {len(rows)}개)")

        tv.bind("<<TreeviewSelect>>", update_sel_label)


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


        self._log("[STEP 4] offset 자동 추정...\n")

        video_ts_int = df_ocr["timestamp_int"].to_numpy().astype(np.int64)
        offset = estimate_offset(video_ts_int, csv_all_ts_int, sample_count=OFFSET_SAMPLE_COUNT)

        ttl_start_video_ts = int(ttl_start_csv - offset)
        ttl_end_video_ts   = int(ttl_end_csv   - offset)

        self._log(f"[OFFSET] median offset = {offset}\n")
        self._log(f"[INFO] 변환된 TTL 시작 video ts: {ttl_start_video_ts}\n")
        self._log(f"[INFO] 변환된 TTL 종료 video ts: {ttl_end_video_ts}\n")
        self._set_progress(52)


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


def main():
    root = tk.Tk()
    app = TrialAlignGuiApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
