# Saliency-Aware Autonomous Driving Simulation using cmSalGAN & Fusion Network

A deep learning–powered vision system combining **saliency detection**, **saliency-guided fusion**, **object detection**, and **autonomous driving simulation** into one unified pipeline.  
Built using **PyTorch**, **OpenCV**, and the **cmSalGAN saliency model**, this project demonstrates how attention maps can improve object detection and guide simulated vehicle movement across visual scenes.

---

## 🚀 Features

* **cmSalGAN-based heatmap generation** (RGB + Depth)
* **Custom Fusion Model** to refine saliency and boost detector performance
* **Faster R-CNN object detection** on saliency-guided images
* **Detector training with best-model checkpointing**
* **Evaluation metrics**: training loss, accuracy, inference time
* **Autonomous car simulation** using saliency-based focus regions
* **Frame-by-frame visualization** + final `.avi` simulation video
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
- Takes RGB + Depth  
- Produces heatmaps representing visual attention  

### **2. Fusion Module**
- Inputs: RGB + Heatmap  
- Outputs: enhanced saliency-guided image  
- Improves detector accuracy and reduces background noise  

### **3. Object Detection (Faster R-CNN)**
- Trained on saliency-guided images  
- Produces bounding boxes, labels, and confidence scores  

### **4. Testing & Evaluation**
- Computes accuracy  
- Measures inference time  
- Selects best model based on loss  

### **5. Simulation**
- Uses saliency + detections to simulate car movement  
- Generates frame-by-frame outputs  

### **6. Visualization**
- Draws objects, saliency overlays, and car path  
- Produces final simulation video  

---


### Run Full Pipeline (All Stages)

```bash
python main.py --all --dataset NJUD
```

### Run Individual Stages

```bash
# Generate saliency maps
python main.py --saliency --dataset NJUD

# Train the detector
python main.py --train

# Test detector performance
python main.py --test

# Run simulation (auto mode)
python main.py --simulate --dataset NJUD --auto

# Visualize results
python main.py --visualize --dataset NJUD
```

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