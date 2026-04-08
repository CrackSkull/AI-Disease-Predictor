# 🧠 AI-Powered Disease Risk Prediction System

### (Dengue • Malaria • Typhoid)

This project is an intelligent, location-aware disease risk prediction system that estimates the relative likelihood of infectious diseases such as dengue, malaria, and typhoid based on user-specific inputs.

It combines **machine learning (XGBoost)**, **spatial analysis (Kernel Density Estimation)**, and **generative AI (Gemini)** to provide both quantitative risk predictions and personalized health guidance.

---

## 🚀 Key Features

* 📍 **Location-Aware Risk Prediction**
  Uses geographic coordinates to estimate disease risk based on nearby case density.

* 🤖 **Hybrid AI System**
  Combines:

  * XGBoost → structured risk prediction
  * KDE → spatial clustering of cases
  * Gemini → human-friendly medical advice

* 🌐 **End-to-End Web Application**
  Includes:

  * Backend API server
  * Interactive frontend UI
  * Real-time prediction + advice

* ⚖️ **Balanced Risk Estimation**
  Adjusts predictions to account for dataset imbalance.

---

## ⚠️ Important Limitation

The current dataset primarily contains **confirmed positive cases**.

➡️ Therefore, this system estimates **relative risk**, not absolute probability of infection.

### To improve accuracy in future:

* Include **negative test records**
* Use **population-level data (denominator)**
* Add **temporal and environmental features**

---

## 🏗️ Project Structure

```
AI-Public-Health-Assistant/
│
├── src/disease_risk/
│   ├── engine.py              # Data processing + training + inference
│   ├── xgb_predictor.py       # XGBoost-based prediction logic
│
├── frontend/
│   ├── index.html             # User interface
│   ├── assets/
│       ├── styles.css         # UI styling
│       ├── app.js             # Frontend logic
│
├── api_server.py              # Backend API + frontend serving + Gemini integration
├── train_ml_model.py          # Model training script
├── requirements.txt           # Dependencies
└── README.md
```

---

## ⚙️ Installation

```bash
python -m pip install -r requirements.txt
```

---

## 🧪 Model Training

```bash
python train_ml_model.py --data-dir . --output model_artifact.json
```

This step:

* Loads epidemiological data
* Trains XGBoost model
* Builds spatial KDE model
* Saves trained model artifact

---

## 🔐 Configure Gemini API

Set your API key as an environment variable:

### macOS / Linux:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

### Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

Or create a `.env` file:

```
GEMINI_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run the Application

```bash
python api_server.py --model model_artifact.json --host 127.0.0.1 --port 8000
```

Open in browser:

```
http://127.0.0.1:8000
```

---

## 🔍 API Endpoints

### 🔹 Health Check

```
GET /health
```

### 🔹 Risk Prediction

```
POST /predict
```

#### Example Request:

```json
{
  "name": "Aman",
  "age": 28,
  "gender": "male",
  "latitude": 13.0827,
  "longitude": 80.2707
}
```

---

### 🔹 AI-Based Advice

```
POST /advice
```

#### Example Request:

```json
{
  "input": {
    "name": "Aman",
    "age": 28,
    "gender": "male",
    "latitude": 13.0827,
    "longitude": 80.2707
  },
  "prediction": {
    "balanced_risk": {
      "dengue": 0.44,
      "malaria": 0.39,
      "typhoid": 0.17
    }
  }
}
```

---

## 📊 Output Description

* **balanced_risk** → Adjusted disease risk scores
* **raw_case_distribution** → Local case proportions
* **location_density_level** → low / medium / high / very_high
* **disclaimer** → Context for AI-generated advice

---

## 🤖 AI Advice System

The `/advice` endpoint uses Gemini to:

* Explain risk in simple language
* Suggest preventive measures
* Highlight early symptoms
* Recommend medical testing
* Provide emergency precautions

If Gemini fails:

* System falls back to predefined medical advice
* Returns `llm_error.hint` for debugging

---

## 🧠 System Workflow

1. User inputs data via frontend
2. Backend processes request
3. XGBoost + KDE generate risk scores
4. Results sent to Gemini
5. Personalized advice returned to user

---

## 🚀 Future Improvements

* Integration of **real-time weather data (rainfall, temperature)**
* Inclusion of **negative case data**
* Temporal modeling for **seasonal trends**
* Deployment as a **scalable cloud-based application**
* Mobile app integration

---

## 📌 Summary

This project demonstrates how **machine learning**, **spatial analytics**, and **generative AI** can be integrated to build an intelligent public health assistant capable of delivering personalized disease risk insights and actionable guidance.

---

## 👨‍💻 Author

Developed as part of an academic project focused on AI-driven public health systems.

