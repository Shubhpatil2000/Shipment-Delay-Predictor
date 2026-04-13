
# 🌿 **AI-Powered Shipment Delay & Risk Predictor — Nashik → Dubai**

A **Streamlit** app that predicts **On-Time vs Delayed**, estimates **Delay Days**, and assigns a **Risk Score (0–100)** for vegetable shipments from **Nashik → Dubai**, with **weather-aware insights** (Monsoon, Cyclone, Heat, Fog).

---

## ✨ **What this project does**

* Takes manual inputs for one or more shipments (mode, equipment, setpoint °C, quantity, month/quarter, costs, and weather flags).
* Loads pre-trained ML models (`.joblib`) to predict **On-Time/Delayed**, **Delay Days**, and **Risk Score**.
* Adds domain heuristics for **weather-aware outputs** (e.g., Monsoon delay probability, Cyclone extra days, Fog impact for Air, Temp risk).
* Displays interactive **Plotly** visuals for instant analysis.

---

## 🧠 **Models & Files**

| File                                                | Purpose                                                |
| --------------------------------------------------- | ------------------------------------------------------ |
| `best_delay_classifier.joblib`                      | Classifies **On-Time** vs **Delayed**                  |
| `best_delay_regressor.joblib`                       | Predicts **Delay Days** *(optional but recommended)*   |
| `best_risk_regressor.joblib`                        | Predicts **Risk Score** *(optional; app has fallback)* |
| `streamlit_app.py`                                  | Streamlit UI and inference logic                       |
| `nashik_dubai_shipments_with_predictions_2024.xlsx` | Sample dataset with example predictions                |

> 💡 **Tip:** Keep all `.joblib` files in the same folder as `streamlit_app.py` or inside `./models/`.

---

## 🛠️ **Setup & Run (Local)**

### 1️⃣ Clone the repository

```bash
git clone <your-repo-url>.git
cd <your-repo-folder>
```

### 2️⃣ Create & activate virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add models

Place your `.joblib` files in `./models/` or the same folder as `streamlit_app.py`.

### 5️⃣ Run the app

```bash
streamlit run streamlit_app.py
```

---

## 🧾 **Input Fields (Quick Guide)**

| Field                                                   | Description                                                                        |
| ------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| **mode**                                                | `Sea` or `Air`                                                                     |
| **equipment_type**                                      | `Reefer 20'`, `Reefer 40'` (Sea) / `AKE ULD` (Air)                                 |
| **commodity**                                           | `Coriander`, `Mint`, `Green Chili`, `Okra`, `Curry Leaves`                         |
| **setpoint_c (°C)**                                     | Typical range **2–10**                                                             |
| **quantity_kg (kg)**                                    | Air: **200–3500** Sea: **3000–28000**                                              |
| **month**, **quarter**                                  | Shipment month/quarter                                                             |
| **freight_usd**, **insurance_usd**, **inland_cost_inr** | Cost details                                                                       |
| **season flags**                                        | `season_monsoon_IN`, `season_cyclone_ARABIAN`, `season_heat_UAE`, `season_fog_UAE` |

---

## 📈 **Outputs**

* **Predicted_Status** → `On-Time` / `Delayed`
* **Predicted_DelayDays** → Numeric delay forecast
* **Predicted_RiskScore** → 0 – 100

### 🌦️ Weather-Aware Insights

* `Weather_Flag` → Monsoon / Cyclone / Heat / Fog / Normal
* `Monsoon_Delay_Prob` → 0–1
* `Cyclone_Delay_Days`
* `Fog_Delay_Prob` → 0–1
* `Expected_Temp_Risk` → High / Medium / Low

---

## 🖼️ **Screenshots**

### 🏠 Home & Overview

<img width="1873" height="615" alt="image" src="https://github.com/user-attachments/assets/997f5311-119b-40f4-af8b-7dab5a90a7ad" />

### 📋 Input Form

<img width="1870" height="681" alt="image" src="https://github.com/user-attachments/assets/d9287469-f482-45e6-8ac3-fba2176a4f5f" />

### 📊 Predictions Table

<img width="1847" height="277" alt="image" src="https://github.com/user-attachments/assets/abbb7d8e-c5bd-4ef4-a91b-7cdb506fe418" />

### 📉 Charts & Insights

<img width="1879" height="830" alt="image" src="https://github.com/user-attachments/assets/3b1952e1-2341-445e-adfc-1d9811240d9b" />

---

## ❓ **FAQ**

**1️⃣ Do I need all three models?**
No. Only the **classifier** is required. If the delay/risk regressors are missing, the app still runs (risk uses a safe fallback).

**2️⃣ Where should I keep the model files?**
In the same folder as `streamlit_app.py` or inside `./models/`. The app checks both locations.

**3️⃣ Can I change commodities or equipment types?**
Yes, edit the dropdown lists in `streamlit_app.py`.

**4️⃣ Can I upload Excel/CSV instead of manual inputs?**
Currently built for manual entry, but can be extended to support batch predictions.

---

## 🗺️ **Roadmap (Next Steps)**

* ⛅ Real-time **weather API** integration for live flags
* 📦 Batch uploads (CSV/Excel) & export of predictions
* 🌍 Generalisation beyond Nashik → Dubai route
* 🥦 Spoilage/quality prediction & carbon/cost impact scores

---

## 📝 **License**

This project is for **educational and demo purposes**.
Use at your own discretion.
You may add a licence of your choice (MIT / Apache-2.0).

---

## 🙌 **Acknowledgements**

Thanks to the domain context and presentation materials developed for the **Nashik → Dubai exports use case**, which inspired this AI-powered prediction system.

---
