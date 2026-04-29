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
# Core logic (기존 mp4_text.py 로직 유지)
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
    # 1) stop_run CSV 로드 (out_idx 기준)
    # --------------------------
    prog(5)
    log("[1/5] stop_run CSV 로딩...\n")
    df_stop = pd.read_csv(metrics_csv_path, low_memory=False)

    # DLC 스타일이면 bodyparts/coords 행 제거
    if "scorer" in df_stop.columns:
        df_stop = df_stop[~df_stop["scorer"].astype(str).str.lower().isin(["bodyparts", "coords"])].copy()

    if "stop_run" not in df_stop.columns:
        raise ValueError("1번 CSV에서 'stop_run' 컬럼을 찾을 수 없습니다.")

    df_stop = df_stop.reset_index(drop=True)

    # out_idx 컬럼이 있으면 그걸, 없으면 행번호를 out_idx로
    if "out_idx" in df_stop.columns:
        df_stop["out_idx"] = pd.to_numeric(df_stop["out_idx"], errors="coerce").astype("Int64")
        df_stop = df_stop.dropna(subset=["out_idx"]).copy()
        df_stop["out_idx"] = df_stop["out_idx"].astype(int)
    else:
        df_stop["out_idx"] = np.arange(len(df_stop), dtype=int)

    df_stop["stop_run"] = pd.to_numeric(df_stop["stop_run"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    df_stop = df_stop[["out_idx", "stop_run"]].sort_values("out_idx").drop_duplicates("out_idx")

    log(f" - stop_run rows: {len(df_stop)} | out_idx min/max: {df_stop['out_idx'].min()}/{df_stop['out_idx'].max()}\n")

    # --------------------------
    # 2) merged CSV 로드 (out_idx 기준)
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
# GUI (stop_run_print.py 스타일로 구성)
# ==========================
class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Overlay (Trial + RUN/STOP) GUI")
        self.root.geometry("1100x760")

        # 파일 경로: 선택 전엔 빈 문자열(= 화면에 안 뜸)
        self.metrics_path = tk.StringVar(value="")
        self.merged_path = tk.StringVar(value="")
        self.video_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")

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
        ttk.Button(row_btn, text="출력 MP4 저장경로", command=self.pick_output).pack(side="left", padx=(0, 8))

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

    def pick_output(self):
        path = filedialog.asksaveasfilename(
            title="출력 MP4 저장경로",
            defaultextension=".mp4",
            filetypes=[("MP4", "*.mp4")]
        )
        if path:
            self.output_path.set(path)
            self._log(f"[GUI] output: {path}\n")

    # ----- run -----
    def on_run(self):
        if self.worker and self.worker.is_alive():
            messagebox.showwarning("진행 중", "이미 실행 중입니다.")
            return

        metrics = self.metrics_path.get().strip()
        merged = self.merged_path.get().strip()
        video = self.video_path.get().strip()
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
        if not out_mp4:
            messagebox.showerror("오류", "출력 MP4 저장경로를 선택하세요.")
            return

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


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
