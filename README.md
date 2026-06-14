# 🌿 Crop Disease AI — Multimodal Agricultural Diagnostic System

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.12-red)
![FastAPI](https://img.shields.io/badge/FastAPI-0.136-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-orange)
![Accuracy](https://img.shields.io/badge/Accuracy-99.74%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

**An AI-powered diagnostic system that identifies crop diseases from leaf images and field conditions with 99.74% accuracy.**

---

## 📌 Overview

Crop Disease AI is a multimodal deep learning system that combines:
- **Computer vision** (ResNet50) to analyse leaf images
- **Machine learning** (XGBoost) to process soil and weather data
- **Fusion model** (MLP) to combine both for higher accuracy
- **Explainability** (GradCAM++) to show which leaf regions triggered the prediction
- **REST API** (FastAPI) to serve predictions
- **Web UI** (Streamlit) for farmers to interact with

The system supports **15 disease classes** across **3 crops** (tomato, potato, pepper) and provides disease identification, confidence scores, severity ratings, treatment recommendations, and visual explanations.

---

## 🎯 Results

| Model | Approach | Test Accuracy |
|-------|----------|--------------|
| ResNet50 | Transfer learning only | 91.9% |
| ResNet50 | Full fine-tuning | 99.52% |
| XGBoost | Tabular features only | 99.50% |
| Fusion MLP | Image + tabular | 99.80% |
| ResNet50 | Balanced sampling (final) | **99.74%** |

**Final metrics on 3,096 test images:**
- Accuracy: **99.74%**
- F1 Score (macro): **0.9972**
- Precision (macro): **0.9969**
- Recall (macro): **0.9976**
- Error rate: **0.26% (only 8 misclassified images)**

---

## 🏗️ Architecture

```
Input: Leaf Image + Field Conditions
         │                    │
         ▼                    ▼
   ResNet50 (CNN)        XGBoost
   2048-d embedding    Tabular prediction
         │                    │
         └──────────┬─────────┘
                    ▼
              Fusion MLP
           (2057 → 512 → 256 → 15)
                    │
                    ▼
         Disease + Confidence
         Treatment + GradCAM++
```

**Key design decisions:**
- Transfer learning from ImageNet weights — faster convergence, better accuracy
- WeightedRandomSampler — handles class imbalance (106 to 2247 images per class)
- Layer-wise learning rates — protects pretrained features while adapting to new domain
- Late fusion — easier to debug, train independently, swap components
- GradCAM++ over GradCAM — sharper localization of disease spots

---

## 🌱 Supported Diseases

| Crop | Diseases |
|------|----------|
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Bacterial Spot, Healthy |
| 🥔 Potato | Early Blight, Late Blight, Healthy |
| 🫑 Pepper | Bacterial Spot, Healthy |

---

## 🚀 How to Run

### Prerequisites
- Python 3.10
- Conda
- ~2GB disk space for models

### 1. Clone the repository
```bash
git clone https://github.com/AryanLovesCoding/Crop-Disease-AI.git
cd Crop-Disease-AI
```

### 2. Create environment
```bash
conda create -n cropai python=3.10 -y
conda activate cropai
pip install -r requirements.txt
```

### 3. Download and prepare data
```bash
kaggle datasets download -d emmarex/plantdisease -p data/raw
unzip data/raw/plantdisease.zip -d data/raw/
```

### 4. Train models (run notebooks in order)
```
day3_data_loader.ipynb
day4_model_training.ipynb
day5_fine_tuning.ipynb
day6_tabular_model.ipynb
day7_fusion_model.ipynb
day8_data_pipeline.ipynb
day12_class_imbalance.ipynb
```

### 5. Start the API
```bash
uvicorn src.api:app --port 8000
```

### 6. Start the frontend
```bash
streamlit run app.py
```

### 7. Or run with Docker
```bash
docker build -t cropai .
docker run -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  cropai
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API status |
| `/health` | GET | Health check |
| `/classes` | GET | List all 15 disease classes |
| `/predict` | POST | Predict disease from image + tabular data |

**Example prediction request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf.jpg" \
  -F "soil_pH=6.5" \
  -F "nitrogen=80" \
  -F "temperature=28" \
  -F "humidity=65" \
  -F "rainfall=120" \
  -F "crop_age_days=45" \
  -F "sunlight_hours=7"
```

**Example response:**
```json
{
  "disease": "Tomato_Early_blight",
  "confidence": 99.74,
  "severity": "Moderate",
  "description": "Fungal disease causing dark spots with yellow halos",
  "treatment": "Apply mancozeb fungicide every 7-14 days",
  "prevention": "Rotate crops, avoid overhead irrigation",
  "urgency": "Act within 3-5 days",
  "gradcam_image": "<base64 encoded heatmap>"
}
```

---

## 📁 Project Structure

```
Crop-Disease-AI/
├── src/
│   ├── api.py                # FastAPI backend
│   ├── models.py             # Model loading utilities
│   └── preprocessing.py      # Data preprocessing utilities
├── data/
│   ├── treatments.json       # Disease treatment database
│   ├── tabular_features_v2.csv
│   └── processed/            # Processed splits and encoders
├── outputs/
│   ├── gradcam/              # GradCAM++ visualizations
│   ├── final_confusion_matrix.png
│   ├── per_class_performance.png
│   ├── model_progression.png
│   └── dataset_analysis.png
├── day1.ipynb
├── day2_data_exploration.ipynb
├── day3_data_loader.ipynb
├── day4_model_training.ipynb
├── day5_fine_tuning.ipynb
├── day6_tabular_model.ipynb
├── day7_fusion_model.ipynb
├── day8_data_pipeline.ipynb
├── day9_api.ipynb
├── day10_gradcam.ipynb
├── day11_streamlit_frontend.ipynb
├── day12_class_imbalance.ipynb
├── day13_github.ipynb
├── day14_evaluation.ipynb
├── day15_hyperparameter_tuning.ipynb
├── day16_realworld_robustness.ipynb
├── day17_docker.ipynb
├── day18_deployment.ipynb
├── day19_ui.ipynb
├── day20_optimization.ipynb
├── day21_checkpoint.ipynb
├── day22_gradcam_improvements.ipynb
├── day23_ui_polish.ipynb
├── day24_evaluation_report.ipynb
├── day25_error_analysis.ipynb
├── day26_deployment_prep.ipynb
├── day27_project_report.ipynb
├── app.py                    # Streamlit frontend
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚠️ Limitations

- Trained on controlled PlantVillage images — accuracy may drop to ~75-80% on real field photos
- Tabular data is synthetic — real IoT soil sensor data would improve accuracy
- Supports 3 crops only — tomato, potato, pepper
- CPU inference only — GPU would reduce response time from 170ms to ~20ms

---

## 🔮 Future Work

- Collect real-world field images and retrain
- Integrate with IoT soil sensors for real tabular data
- Build Flutter mobile app for farmers
- Deploy to cloud with Hugging Face Hub for model storage
- Add multi-language support
- Expand to more crop types

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Image model | PyTorch, ResNet50 |
| Tabular model | XGBoost |
| Fusion model | PyTorch MLP |
| Explainability | GradCAM++ |
| API | FastAPI, Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker |
| Data processing | scikit-learn, pandas, numpy |
| Visualization | matplotlib, seaborn |

---

## 👨‍💻 Developer

**Aryan Ajmera**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/aryan-ajmera7)
[![Website](https://img.shields.io/badge/Website-Visit-green)](https://aryanlovescoding.github.io/AryanWebsite/)

---

## 📄 License

MIT License — feel free to use this project for learning and research.
