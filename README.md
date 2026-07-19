# Saliency-Aware Autonomous Driving Simulation using cmSalGAN & Fusion Network

A deep learning–powered vision system combining **saliency detection**, **saliency-guided fusion**, **object detection**, and **autonomous driving simulation** into one unified pipeline.  
Built using **PyTorch**, **OpenCV**, and the **cmSalGAN saliency model**, this project demonstrates how attention maps can improve object detection and guide simulated vehicle movement across visual scenes.

---

## 🚀 Features

* **cmSalGAN-based heatmap generation** (RGB + Depth)
* **Custom Fusion Model** (jointly trained with the detector, saved and reloaded with it)
* **Faster R-CNN object detection** trained on boxes derived from GT saliency masks (connected components → bounding boxes, class `salient_object`)
* **Real evaluation metrics** — saliency: MAE, adaptive F-measure, S-measure vs. GT masks; detection: AP@0.5, precision/recall/F1; measured inference time
* **Autonomous car simulation**: potential-field steering (goal attraction + saliency-gradient and detection-box repulsion) over the frame stream
* **Frame-by-frame visualization** (car, path trace, goal, detections, heatmap overlay) + final `.avi` simulation video
* **End-to-end modular pipeline** via `main.py`
* **Supports NJUD, NLPR, and STEREO datasets**

---

## 📦 Tech Stack

* **Framework** → PyTorch  
* **Computer Vision** → OpenCV, NumPy  
* **Visualization** → OpenCV Rendering  
* **Automation** → tqdm, argparse, JSON  
* **Language** → Python 3.12  
* **Environment** → Google Colab / Linux / Windows  

---

## 🛠️ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/guru-bharadwaj20/saliency-aware-autonomous-driving.git
cd saliency-aware-autonomous-driving
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```
> 💡 `Ensure PyTorch and OpenCV are installed properly before running.`

---

## 🧠 Pipeline Overview

The updated pipeline consists of the following stages:

### **1. Saliency Generation (cmSalGAN)**
- Takes RGB + Depth (ImageNet-normalized RGB input)  
- Produces heatmaps at the original image resolution for both train and test splits  

### **2. Fusion Module**
- Inputs: RGB + Heatmap → 4-channel attention-weighted fusion  
- Trained jointly with the detector; weights saved in the same checkpoint  

### **3. Object Detection (Faster R-CNN)**
- Supervision: bounding boxes derived from GT saliency masks via connected components (2 classes: background / salient_object)  
- Produces bounding boxes, labels, and confidence scores in original image coordinates  

### **4. Testing & Evaluation**
- Saliency: MAE, adaptive F-measure, S-measure against GT masks  
- Detection: AP@0.5, precision/recall/F1 against GT-derived boxes  
- Measures real per-image inference time  
- Saves predictions to `results/detection_outputs/<DATASET>_detections.json`  

### **5. Simulation**
- `CarSimulator` steers a point-mass car per frame using `obstacle_avoidance` (goal attraction + saliency-gradient repulsion + detection repulsion)  

### **6. Visualization**
- Draws detections, saliency overlays, car, path trace, heading arrow, and goal marker  
- Produces frame images + final `.avi` simulation video  

---


### Run Full Pipeline (All Stages)

```bash
python main.py --all --dataset NJUD
```

### Run Individual Stages

```bash
# 1. Generate saliency maps (requires models/cmSalGAN.ckpt — see downloads below)
python main.py --saliency --dataset NJUD

# 2. Train the detector + fusion module (uses the saliency maps from step 1)
python main.py --train --dataset NJUD

# 3. Test detector: saves detections JSON and prints real metrics
python main.py --test --dataset NJUD

# 4. Run the car simulation and render the video
python main.py --simulate --dataset NJUD

# Render saliency/detection overlays without the car
python main.py --visualize --dataset NJUD
```

> Stages depend on each other in that order: `--train`/`--test` expect saliency maps,
> and `--simulate` gives best results after `--test` has produced detections.

---

## 📦 Dataset & Model Downloads

| Type | Description | Download Link |
|------|--------------|----------------|
| 📁 Dataset | Training Set (contains `GT`, `RGB`, and `depth` folders) | [Download from Google Drive](https://drive.google.com/file/d/1YENRxUxAcFQhxcesxaWHEM3BipcIayX1/view?usp=sharing) |
| 📁 Dataset | Edge Dataset (for training with edge maps) | [Download from Google Drive](https://drive.google.com/file/d/1J8z_LH2KvHYZEXApcwLgV8KoqBrxAJ5d/view?usp=sharing) |
| 🧠 Model | Pretrained cmSalGAN Model Checkpoint (`models/cmSalGAN.ckpt`) | [Download from Google Drive](https://drive.google.com/file/d/1j18BvmGEUip1NSlK3N4t66jU_WeV2tCF/view?usp=sharing) |

> ⚙️ After downloading, extract and place them into the respective directories:
> - `Dataset/train/` → for datasets  
> - `models/` → for pretrained model checkpoint

---

## 📂 Project Structure

```bash
saliency-aware-autonomous-driving/
│
├── data/                        # Input datasets (NJUD, NLPR, STEREO)
├── models/                      # cmSalGAN + detector model checkpoints
│   └── checkpoints/             # Best detector models
├── results/                     # Auto-generated outputs
│   ├── saliency_maps/           # Generated heatmaps
│   ├── detection_outputs/       # Inference predictions
│   └── simulation_outputs/      # Frames + final simulation video
├── detection/                   # Fusion model + detector training/testing/metrics
├── saliency_generation/         # cmSalGAN saliency generation scripts
├── simulation/                  # Custom car simulator + visualizer
├── main.py                      # Pipeline orchestrator
├── requirements.txt             # Package dependencies
└── README.md                    # Documentation
```

---

## 🎞️ Output

The system produces:

- Saliency heatmaps  
- Fusion-enhanced images  
- Object detection outputs  
- Car simulation frames 

---

## ☁️ Deployment

This project runs efficiently on:

- **Google Colab**
- **Linux / Ubuntu**
- **Windows (via WSL or native Python)**

To export the project and save simulation results:
```bash
!zip -r saliency_aware_autonomous_driving.zip /content/saliency
!cp saliency_aware_autonomous_driving.zip /content/drive/MyDrive/
```

---

## 🤝 Contributing

Contributions are always welcome!

1. Fork the repository  
2. Create a new branch (`feature/new-feature`)  
3. Commit your changes  
4. Push to your branch  
5. Open a Pull Request 🚀

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software with proper attribution.