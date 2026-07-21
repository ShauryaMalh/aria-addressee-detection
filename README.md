# 🎧 Multimodal Addressee Detection with Aria Everyday Activities Dataset
 
[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![ProjectAria](https://img.shields.io/badge/Meta-ProjectAria-purple)](https://projectaria.com/)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-orange)](https://github.com/ultralytics/ultralytics)
[![GroundingDINO](https://img.shields.io/badge/IDEA--Research-GroundingDINO-green)](https://github.com/IDEA-Research/GroundingDINO)
[![Gemini](https://img.shields.io/badge/Google-Gemini%202.5%20Flash-blue?logo=google)](https://ai.google.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
 
A multimodal AI system that fuses **7-channel spatial audio**, **egocentric RGB vision**, **eye gaze**, and **6DoF head pose** from Meta's Project Aria glasses to detect who a wearer is speaking to and ground verbal object references to precise 3D locations in household scenes.
 
Built on the [Aria Everyday Activities (AEA) Dataset](https://www.projectaria.com/datasets/aea/), this project compares three approaches to the addressee detection and spatial grounding problem.
 
---
 
## 📌 Overview
 
When a person wearing AR glasses says *"Can you grab that?"*, a robot assistant needs to know:
1. **Who** is being spoken to (the robot, another person, or no one)?
2. **What** is the referred object?
3. **Where** is it in 3D space?
This system answers all three questions by combining acoustic, visual, and spatial signals across three increasingly capable methods.
 
---
 
## 🧠 Methods
 
### Method A — Structured Signal Processing Pipeline
- **SRP-PHAT** direction-of-arrival estimation on a 7-mic array at 1° resolution
- Microphone geometry projected onto the device x-z plane, filtered to the 300–3500 Hz speech band
- Spectrum-shape forward-mask (extending the SAVVY paper, UW ECE 2025) to separate the wearer's voice from other speakers
- Gaze vectors projected into world frame via rotation matrices, intersected with a semi-dense 3D point cloud (838K points) for sub-meter object localization
- **GroundingDINO** (open-vocabulary) projects 2D detections to 3D via camera ray–point cloud intersection
- **Result: 100% decision accuracy** with exact 3D grounding and navigation route rendering
### Method B — Gemini 2.5 Flash VLM Baseline
- Sends egocentric RGB frame + transcript to Gemini for holistic multimodal reasoning
- No explicit spatial computation (no DoA, no gaze projection, no point cloud)
- **Result:** Strong deictic language understanding, but no spatial output and occasional hallucinations
### Method C — Combined VLM + Spatial Grounding
- Gemini handles language understanding (what object? is there a task?)
- Method A's spatial pipeline handles grounding (where is it? navigation route?)
- DoA and gaze used to validate VLM decisions before acting
- **Result:** Preserves exact 3D coordinates while gaining discourse-level reasoning that rules alone cannot handle
---
 
## 📊 Results
 
| Method | Addressee Accuracy | 3D Grounding | Navigation Route | Deictic Resolution |
|---|---|---|---|---|
| A (Signal Processing) | 100% | ✅ Exact | ✅ Yes | ✅ Gaze-based |
| B (Gemini VLM) | Strong | ❌ None | ❌ No | ✅ Language-based |
| C (Combined) | Strong | ✅ Exact | ✅ Yes | ✅ Both |
 
### Sample Visualizations
 
Case 1 (Task-Directed Speech) and Case 2 (Non-Addressed Speech) output maps are in [`project/results/`](project/results/).
 
---
 
## 🛠 Tech Stack
 
| Component | Technology |
|---|---|
| Spatial Audio (DoA) | `pyroomacoustics` — SRP-PHAT |
| Object Detection | GroundingDINO (open-vocabulary), YOLOv8 |
| Vision-Language Model | Gemini 2.5 Flash (`google-genai`) |
| Head Pose & Gaze | `projectaria-tools` MPS loader |
| Point Cloud | Semi-dense SLAM (838K points) |
| Visualization | `matplotlib`, `rerun-sdk` |
| Data Format | VRS (Meta's sensor data format) |
 
---
 
## 📂 Project Structure
 
```
aria-addressee-detection/
├── project/
│   ├── perception.py        # Data loading: VRS, trajectory, gaze, speech, audio
│   ├── doa.py               # SRP-PHAT direction-of-arrival + speaker classification
│   ├── object_finder.py     # GroundingDINO open-vocabulary detection + 3D projection
│   ├── method_a.py          # Structured signal processing pipeline
│   ├── method_b.py          # Gemini VLM baseline
│   ├── method_c.py          # Combined VLM + spatial grounding
│   ├── compare_all.py       # Side-by-side comparison of all three methods
│   ├── visualization.py     # Top-down map + egocentric view rendering
│   └── results/             # Output visualizations (case1_*.png, case2_*.png)
├── aea_download_urls.json   # Dataset download manifest
├── test_load.py             # Sanity check for data loading
├── test_speech.py           # Speech transcript validation
├── requirements.txt
└── .gitignore
```
 
---
 
## ⚡ Installation & Setup
 
### 1. Clone the repo
```bash
git clone https://github.com/ShauryaMalh/aria-addressee-detection.git
cd aria-addressee-detection
```
 
### 2. Install PyTorch (GPU recommended)
Install the correct version for your CUDA setup from [pytorch.org](https://pytorch.org/get-started/locally/) before proceeding.
 
```bash
# Example for CUDA 12.x:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
 
### 3. Install dependencies
```bash
python -m venv aria_env
source aria_env/bin/activate   # Windows: aria_env\Scripts\activate
pip install -r requirements.txt
```
 
### 4. Download the AEA dataset
```bash
aria_dataset_downloader \
  -c aea_download_urls.json \
  -o ./aea_data \
  -l loc2_script2_seq4_rec2
```
 
### 5. Download YOLOv8 weights (optional)
```bash
# yolov8n.pt is downloaded automatically by Ultralytics on first run,
# or manually from: https://github.com/ultralytics/assets/releases
```
 
### 6. Set your Gemini API key (required for Methods B and C)
```bash
export GEMINI_API_KEY=your_key_here
```
Get a free API key at [Google AI Studio](https://aistudio.google.com).
 
---
 
## 🚀 Usage
 
**Update the sequence path** in `project/perception.py`:
```python
SEQ_DIR = Path("/path/to/your/aea_data/loc2_script2_seq4_rec2")
```
 
**Run each method individually:**
```bash
cd project
python method_a.py      # Structured pipeline
python method_b.py      # Gemini VLM baseline
python method_c.py      # Combined approach
```
 
**Compare all three methods side-by-side:**
```bash
python compare_all.py
```
 
**Generate visualizations:**
```bash
python visualization.py   # Outputs top-down maps to project/results/
```
 
---
 
## 🔬 Technical Details
 
### Direction-of-Arrival (DoA)
SRP-PHAT is run on Aria's 7-microphone array. Microphone positions are projected onto the device x-z plane (horizontal plane) for 2D azimuth estimation. A forward-mask classifier then determines whether the dominant sound source is in front of (wearer) or off-axis (other speaker) based on spatial spectrum shape.
 
### 3D Object Grounding
GroundingDINO processes the egocentric RGB frame with an arbitrary text query (e.g., `"creamer. spoon. bottle."`). Each 2D detection center is unprojected to a 3D ray using the Aria camera calibration, then intersected with the filtered semi-dense point cloud to produce a world-frame position.
 
### Gaze-Based Deictic Resolution
When speech contains deictic references ("that", "this", "it"), the system resolves the referent by projecting the eye gaze vector into world frame and finding the detected object with highest cosine alignment to the gaze ray within a 2–5m range.
 
---
 
## 🚀 Future Improvements
 
- Extend to multi-sequence evaluation for quantitative benchmarking
- Replace hardcoded robot position assumption with actual robot localization
- Add speaker diarization to handle overlapping speech
- Test on additional AEA sequences with different household layouts
- Fine-tune GroundingDINO on household object vocabulary
---
 
## 📜 License
 
MIT License — free to use, modify, and distribute.
 
---
 
## 👨‍💻 Author
 
Developed by **Shaurya Malhotra**  
University of Washington — ECE/CSE  
Project built on [Meta's ProjectAria Tools](https://github.com/facebookresearch/projectaria_tools) and the [Aria Everyday Activities Dataset](https://www.projectaria.com/datasets/aea/).
 
