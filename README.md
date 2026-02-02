# 🚗 Smart Driver Monitoring - AI & Analytics

A comprehensive AI-powered system for driver monitoring, including:
- **Driver Rating Prediction** - ML models to predict driver ratings from telematics
- **Feedback Analytics (NLP)** - Sentiment analysis of passenger feedback
- **Violation Detection** - Classify driving violations from sensor data
- **Forgery Detection** - OCR + image analysis for document verification

## Project Structure

```
Week4/
├── data/                    # Data files
│   ├── telematics.csv       # Synthetic telematics data
│   ├── ratings.csv          # Driver ratings
│   ├── feedback.csv         # Passenger feedback text
│   ├── driver_features.csv  # Engineered features
│   └── licenses/            # Sample license images
├── notebooks/               # Jupyter notebooks (modules)
│   ├── 01_data_prep.ipynb
│   ├── 02_eda_feature_engineering.ipynb
│   ├── 03_driver_ratings.ipynb
│   ├── 04_feedback_nlp.ipynb
│   ├── 05_violations_detection.ipynb
│   └── 06_forgery_detection.ipynb
├── src/                     # Source code & models
│   ├── data_generator.py    # Generate synthetic data
│   ├── forgery_check.py     # Forgery detection module
│   └── *.joblib             # Trained models (after running notebooks)
├── demos/
│   └── streamlit_app.py     # Interactive demo dashboard
├── requirements.txt         # Python dependencies
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Synthetic Data

```bash
python src/data_generator.py
```

This creates sample telematics, ratings, and feedback data in the `data/` folder.

### 3. Run Jupyter Notebooks

```bash
jupyter lab
```

Run notebooks in order (01 → 06) to:
- Prepare and explore data
- Train ML models
- Analyze feedback sentiment
- Build violation detection
- Create forgery detection pipeline

### 4. Launch Streamlit Demo

```bash
streamlit run demos/streamlit_app.py
```

Visit http://localhost:8501 to interact with the demo dashboard.

## 📊 Modules Overview

### Module 1: Data Preparation
- Generate synthetic telematics data
- Create driver ratings based on behavior
- Generate passenger feedback text

### Module 2: EDA & Feature Engineering
- Explore data distributions
- Visualize correlations
- Create aggregated driver features

### Module 3: Driver Rating Prediction
- Train Linear Regression, Random Forest, XGBoost
- Evaluate with RMSE, MAE, R²
- Feature importance analysis (SHAP)

### Module 4: Feedback NLP
- Text preprocessing
- TF-IDF + Logistic Regression baseline
- Hugging Face transformer sentiment
- Topic modeling (LDA)

### Module 5: Violation Detection
- Binary classification for driving violations
- Handle class imbalance with weights
- ROC/AUC evaluation

### Module 6: Forgery Detection
- OCR text extraction (Tesseract)
- Format validation (regex)
- Image quality analysis (OpenCV)
- CNN classification (MobileNetV2)

## 🛠️ Dependencies

Core libraries:
- **Data**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **ML**: scikit-learn, xgboost
- **Deep Learning**: tensorflow
- **NLP**: transformers, spacy
- **Vision/OCR**: opencv-python, pytesseract
- **Demo**: streamlit

## 📝 Notes

- **Tesseract OCR**: Install separately from https://github.com/tesseract-ocr/tesseract
- **Privacy**: All data is synthetic. In production, use anonymized data.
- **Models**: Trained models are saved as `.joblib` files in `src/`

## 📈 Metrics & Evaluation

| Model | Task | Primary Metric |
|-------|------|----------------|
| Rating Prediction | Regression | RMSE, R² |
| Sentiment Analysis | Classification | F1, Accuracy |
| Violation Detection | Classification | Precision, Recall, AUC |
| Forgery Detection | Classification | Precision (minimize false accusations) |

## 🎯 Learning Objectives

By completing this project, you will:
1. Build end-to-end ML pipelines for real-world problems
2. Work with telematics, text, and image data
3. Handle class imbalance and feature engineering
4. Deploy models in a simple web interface
5. Understand explainability (SHAP, feature importance)

---

*Made for Week 4 Training - AI & Analytics for Smart Driver Monitoring*
