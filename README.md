# ⚙️ PredictIQ — Machine Health Monitor

> Real-time machine failure prediction dashboard powered by XGBoost + Streamlit.

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd ml_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
ml_dashboard/
├── app.py                        # Main Streamlit application
├── xgboost_binary_model.joblib   # Trained XGBoost classifier
├── scaler.joblib                 # Fitted StandardScaler
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎯 Features

| Feature | Description |
|---|---|
| **Single Prediction** | Slider-based form for real-time prediction |
| **Batch Predictions** | Upload CSV → get predictions for all rows |
| **Health Gauge** | Plotly gauge chart for health score |
| **Sensor Radar** | Radar chart comparing inputs to training means |
| **Deviation Chart** | Bar chart showing standard-deviation outliers |
| **Dark Mode UI** | Professional SaaS-style dark dashboard |
| **Model Info Page** | Scaler stats, encoding reference, architecture |

---

## 🔢 Input Parameters

| Parameter | Type | Typical Range |
|---|---|---|
| Machine Type | Categorical (M/L/H) | M = Medium, L = Low, H = High |
| Air Temperature | Float (K) | 295 – 305 K |
| Process Temperature | Float (K) | 305 – 315 K |
| Rotational Speed | Int (RPM) | 1000 – 2800 RPM |
| Torque | Float (Nm) | 3 – 80 Nm |
| Tool Wear | Int (min) | 0 – 250 min |

---

## 📤 Output

- **Prediction**: Normal (0) or Machine Failure (1)
- **Failure Probability**: Probability score from `predict_proba`
- **Health Score**: `(1 − failure_probability) × 100`
- **Recommendation**: Automated maintenance guidance

---

## ☁️ Deployment

### Streamlit Cloud
1. Push the project to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file

### HuggingFace Spaces
1. Create a new Space with **Streamlit** SDK
2. Upload all project files
3. The Space will auto-build and deploy

### Render
Add a `render.yaml`:
```yaml
services:
  - type: web
    name: predictiq
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

---

## 🔬 Model Details

- **Algorithm**: XGBoost Binary Classifier
- **Preprocessing**: StandardScaler (z-score normalisation on 5 numerical features)
- **Categorical encoding**: One-hot encoding for Machine Type → `[type_H, type_L, type_M]`
- **Total feature dimensions**: 8 (5 scaled + 3 OHE)
- **Dataset**: AI4I 2020 Predictive Maintenance Dataset

---

## 📄 License

MIT — free to use and modify.
