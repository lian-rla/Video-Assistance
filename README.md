# Video Assistance

<a href="https://github.com/opencv/opencv"><img src="https://github.com/opencv.png" width="50" height="50" alt="OpenCV"/></a>&nbsp;
<a href="https://github.com/JaidedAI/EasyOCR"><img src="https://github.com/JaidedAI.png" width="50" height="50" alt="EasyOCR"/></a>&nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)&nbsp;
<a href="https://github.com/DeepLabCut/DeepLabCut"><img src="https://github.com/DeepLabCut.png" width="50" height="50" alt="DeepLabCut"/></a>

An integrated GUI tool for behavioral neuroscience video analysis.  
Covers the full pipeline from EasyOCR-based timestamp extraction to frame reinforcement, stop/run behavior discrimination, and text-overlay video generation — all in a single interface.

---

## Features

| Feature | Description |
|---------|-------------|
| **1. Timestamp Extraction** | Extract timestamps overlaid on video using EasyOCR → save as CSV |
| **2. Frame Reinforcement** | Reinforce frames based on Unreal Log trials → output reinforced video and aligned CSV |
| **3. Stop Run Discrimination** | Classify stop/run behavior using DLC (DeepLabCut) coordinate data |
| **4. Text Video** | Overlay analysis results as text onto the output video |

---

## Requirements

### Python
Python 3.8 or higher is recommended.

### Python Packages

| Package | Purpose | Install |
|---------|---------|---------|
| `opencv-python` | Video I/O, ROI selection, frame processing | `pip install opencv-python` |
| `easyocr` | Timestamp OCR extraction | `pip install easyocr` |
| `torch` | GPU acceleration for EasyOCR (PyTorch) | See CUDA section below |
| `pandas` | CSV/Excel data processing | `pip install pandas` |
| `numpy` | Array operations | `pip install numpy` |
| `openpyxl` | Read `.xlsx` files (pandas dependency) | `pip install openpyxl` |

Install all at once:
```bash
pip install opencv-python easyocr pandas numpy openpyxl
```

> `tkinter` is included in the Python standard library and does not require a separate installation.  
> It is bundled by default with the official Python installer on Windows.

### PyTorch (CUDA version)

Install PyTorch matching your CUDA version for GPU acceleration.

**Step 1: Check your installed CUDA version**
```bash
nvidia-smi
```

**Step 2: Install PyTorch for your CUDA version**

| CUDA Version | Install Command |
|--------------|-----------------|
| CUDA 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| CUDA 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| CUDA 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| No GPU (CPU only) | `pip install torch` |

> For the exact version, refer to the [PyTorch official site](https://pytorch.org/get-started/locally/).

### CUDA Toolkit (for GPU use)
- GPU is strongly recommended for Timestamp Extraction.
- Install the CUDA Toolkit matching your GPU from the [NVIDIA official site](https://developer.nvidia.com/cuda-downloads).
- Without a GPU, the tool runs in CPU mode but processing speed will be significantly slower.

### DeepLabCut (required for Stop Run Discrimination)

Stop Run Discrimination takes a joint coordinate CSV exported by DeepLabCut (DLC) as input.  
DLC is not bundled with this tool — run your DLC analysis separately and use the output CSV.

**Install**
```bash
pip install deeplabcut
```

> For a full installation with GUI:
> ```bash
> pip install "deeplabcut[gui]"
> ```

**DLC analysis workflow**
1. Launch DLC and load the project for your experiment (`Load project`)
2. Run `analyze videos` on the video output from Frame Reinforcement
3. Use the exported label coordinate CSV as input `B` in Stop Run Discrimination

**Setting bodyparts (when using on a different machine)**  
If no pre-trained DLC model is available, train a model with bodyparts set to `left_leg` / `right_leg` and use it.

> For detailed DLC usage, refer to the [DeepLabCut official documentation](https://deeplabcut.github.io/DeepLabCut).

---

## Usage

### Run
```bash
python video_assistance_GUI.py
```

---

### 1. Timestamp Extraction

Extracts timestamps from video frames using OCR and saves them as a CSV file.

**Input**
- `A` : Video file to extract timestamps from (`.mp4`, `.avi`, `.mov`, `.mkv`)
- `B` : Folder to save the extracted timestamp CSV

**Steps**
1. Select a video file → select a CSV save folder
2. Click the **Run OCR** button
3. In the ROI selection window, drag over the timestamp area and press `Enter` or `Space`
4. Monitor progress — CSV is saved automatically upon completion

**Output**
- `{video_filename}_timestamp.csv`

> If multiple GPUs are detected, processing is automatically parallelized across all GPUs.

---

### 2. Frame Reinforcement

Reinforces frames based on TTL signals from the Cheetah event log and trial information from the Unreal Log.

#### Prerequisite: Convert Cheetah Event Log CSV → XLSX

Before using Frame Reinforcement, convert the Cheetah event log `.csv` file to `.xlsx` format.

1. In Excel, go to `Data` → `From Text/CSV`
2. Import the Cheetah event log `.csv` file
3. Select `Transform Data`
4. Select the timestamp column and set the data type to **Text**
5. Click `Close & Load`
6. Delete the original `.csv` and save as `.xlsx` with a new name

**Input**
| Item | Description |
|------|-------------|
| `A` | Timestamp Extraction output CSV |
| `B` | Source video to reinforce |
| `C` | Cheetah event log (use after converting to `.xlsx`) |
| `D` | Unreal Log |
| `E` | Output save path |

**Unreal Align Options**
- `TRUE column` / `TRIAL column` : Select columns to use from the Unreal Log
- `Row range (Start / End)` : Unreal Log row range to use for analysis (1-based)
- Click **Preview / Select Range** to view the table and select a row range interactively with the mouse

**Output**
- Reinforced video
- CSV with trial timestamps aligned

---

### 3. Stop Run Discrimination

Analyzes leg coordinate data from DLC to classify each frame as stop or run.

**Prerequisite: DLC Analysis**
1. Launch DLC and load the appropriate project (`Load project`)
   - Side-view video: `test14` project
   - Front-view video: `test15` project
2. Run `analyze videos` on the Frame Reinforcement output video
3. Export the DLC label coordinate CSV

> When using DLC on a different machine, use a model trained with bodyparts set to `left_leg` / `right_leg`.

**Input**
| Item | Description |
|------|-------------|
| `A` | Unreal Log CSV |
| `B` | DLC output label coordinate CSV |
| `C` | Output save path |

**Parameters**
| Parameter | Default | Description |
|-----------|---------|-------------|
| FPS | 30 | Video FPS (set according to your video) |
| Label Jitter Tolerance | 1.0 | Allowed jitter range in pixels |
| Stop Duration | 0.25 s | Classified as stop if jitter stays within tolerance for this duration |
| Smoothing | 3 | Smoothing strength applied to the center coordinate |

> Example: For a 30 fps video, apply smoothing=3 to DLC coordinates; if movement stays within 1 pixel for 0.25 seconds or more, the frame is classified as stop.

**Output CSV Columns**

| Column | Description |
|--------|-------------|
| `meta_leg_center_x/y` | Center x, y coordinates |
| `meta_dist` | Average center coordinate displacement |
| `meta_dist_smooth` | Smoothed average center coordinate |
| `meta_stop` | `1` = run, `0` = stop |
| `trial_F` | Current trial |
| `trial_G` | Cumulative trial |

---

### 4. Text Video

Overlays analysis results as text onto the video.

**Input**
| Item | Description |
|------|-------------|
| `A` | Stop Run Discrimination output CSV |
| `B` | Timestamp Extraction output align CSV |
| `C` | Input video to overlay text on (Frame Reinforcement output recommended) |
| `D` | Output save path |

---

## Pipeline Overview

```
Video
 └─► [1. Timestamp Extraction]  ──► timestamp CSV
                                          │
                                          ▼
     Cheetah Events (.xlsx)  ──► [2. Frame Reinforcement] ──► reinforced video + align CSV
     Unreal Log              ──►
                                          │
                                          ▼
     DLC label CSV  ──────────► [3. Stop Run Discrimination] ──► stop/run CSV
                                          │
                                          ▼
                               [4. Text Video] ──► text-overlay video
```
