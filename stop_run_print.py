import os
import threading
import traceback
import time

import numpy as np
import pandas as pd

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext


# ==========================
# Core logic 
# ==========================
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
    """
    기존 코드 로직 그대로 수행.
    최종 CSV에서는 meta_time, time 컬럼 제거.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)

    def prog(p):
        if progress_fn:
            progress_fn(p)

    t0 = time.time()

    # ==========================
    # 2. Circular_memory.csv 읽어서 trial 정보 정리
    # ==========================
    prog(5)
    log("[1/6] Circular(Unreal) CSV 로딩...\n")
    circular = pd.read_csv(circular_csv, header=None)

    time_col_idx = 0
    flag_col_idx = 18  # S열 (True/False)
    trial_now_col_idx = 5  # F열
    trial_cum_col_idx = 6  # G열

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

    # ==========================
    # 3. DLC CSV 읽기 + dist_smooth + stop_run 계산
    # ==========================
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

    # ==========================
    # 4. DLC 프레임에 sensor 시간축 붙이기
    # ==========================
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

    # ==========================
    # 5. flatten
    # ==========================
    prog(60)
    log("[4/6] 컬럼 flatten...\n")
    dlc_flat = dlc_out.copy()
    dlc_flat.columns = [flatten_col(c) for c in dlc_flat.columns]

    dlc_flat["meta_time"] = pd.to_numeric(dlc_flat["meta_time"], errors="coerce")
    dlc_flat["meta_frame_index"] = pd.to_numeric(dlc_flat["meta_frame_index"], errors="coerce")
    dlc_flat = dlc_flat.dropna(subset=["meta_time"])

    # ==========================
    # 6. merge_asof
    # ==========================
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

    # ==========================
    # 7. 저장 (meta_time, time 제거)
    # ==========================
    prog(90)
    log("[6/6] CSV 저장...\n")
    aligned_out = aligned.drop(columns=["meta_time", "time"], errors="ignore")
    aligned_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    prog(100)
    dt = (time.time() - t0)
    log(f"\n[DONE] 저장 완료: {output_csv}\n")
    log(f"[DONE] 소요 시간: {dt:.2f} sec\n")


# ==========================
# GUI
# ==========================
class App:
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


def main():
    root = tk.Tk()
    app = App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
