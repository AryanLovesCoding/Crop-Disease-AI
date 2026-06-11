# 🌿 Crop Disease AI — Multimodal Agricultural Diagnostic System

A deep learning system that diagnoses crop diseases from leaf images and field conditions with 99.74% accuracy.

## 🔍 Overview

This system combines computer vision and tabular machine learning to identify 15 crop disease classes across tomatoes, potatoes, and peppers. It uses a multimodal fusion approach — combining a fine-tuned ResNet50 image model with an XGBoost tabular model — to achieve higher accuracy than either model alone.

## 🏗️ Architecture

- **Image Model**: ResNet50 pretrained on ImageNet, fine-tuned on PlantVillage dataset
- **Tabular Model**: XGBoost trained on soil pH, NPK levels, temperature, humidity, rainfall, crop age, and sunlight hours
- **Fusion Model**: MLP that combines 2048-d image embeddings + 9 tabular features
- **Explainability**: Grad-CAM heatmaps showing which leaf regions triggered the prediction
- **API**: FastAPI backend serving predictions via REST endpoints
- **Frontend**: Streamlit web app for farmers to upload images and get diagnoses

## 📊 Results

| Model | Test Accuracy |
|-------|--------------|
| ResNet50 (image only) | 99.52% |
| XGBoost (tabular only) | 99.50% |
| Fusion (image + tabular) | 99.80% |
| Balanced fusion (final) | **99.74%** |

## 🌱 Supported Crops & Diseases

- **Tomato**: Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Bacterial Spot, Healthy
- **Potato**: Early Blight, Late Blight, Healthy
- **Pepper**: Bacterial Spot, Healthy

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/AryanLovesCoding/Crop-Disease-AI.git
cd Crop-Disease-AI
```

### 2. Create environment
```bash
conda create -n cropai python=3.10 -y
conda activate cropai
pip install torch torchvision scikit-learn pandas numpy matplotlib pillow jupyter xgboost joblib requests fastapi uvicorn python-multipart streamlit grad-cam seaborn
```

### 3. Download dataset
```bash
kaggle datasets download -d emmarex/plantdisease -p data/raw
unzip data/raw/plantdisease.zip -d data/raw/
```

### 4. Train models
Run notebooks in order:
- `notebooks/day3_data_loader.ipynb`
- `notebooks/day4_model_training.ipynb`
- `notebooks/day5_fine_tuning.ipynb`
- `notebooks/day6_tabular_model.ipynb`
- `notebooks/day7_fusion_model.ipynb`

### 5. Start the API
```bash
uvicorn src.api:app --port 8000
```

### 6. Start the frontend
```bash
streamlit run app.py
```

## 📁 Project Structure

crop-disease-ai/
├── data/
│   └── treatments.json       # Disease treatment recommendations
├── notebooks/                # Daily Jupyter notebooks
├── src/
│   ├── api.py               # FastAPI backend
│   ├── models.py            # Model loading utilities
│   └── preprocessing.py     # Data preprocessing utilities
├── outputs/
│   └── gradcam/             # Grad-CAM visualizations
├── app.py                   # Streamlit frontend
└── README.md

## ⚠️ Limitations

- Trained on PlantVillage controlled images — performance may degrade on real-world field photos
- Currently supports 3 crops (tomato, potato, pepper)
- Tabular data is synthetic — real soil sensor data would improve accuracy

## 🔮 Future Work

- Collect real-world field images for training
- Add mobile app (Flutter)
- Integrate with IoT soil sensors
- Expand to more crop types
- Multi-language support for farmers

## 👨‍💻 Developer

**Aryan Ajmera**
- [LinkedIn](https://www.linkedin.com/in/aryan-ajmera7)
- [Website](https://aryanlovescoding.github.io/AryanWebsite/)

## 📄 License

MIT License
