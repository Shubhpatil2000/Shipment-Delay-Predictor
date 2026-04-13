

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Manual Shipment Predictor", page_icon="🌿", layout="wide")

st.title("🌿 Manual Shipment Predictor — Nashik → Dubai")
st.caption("Enter shipment details with guided limits. The app predicts on-time status, delay days, and risk score, plus weather-aware outputs.")

# Model loading (paths below)

CLASSIFIER_PATH = "best_delay_classifier.joblib"      # required
DELAY_MODEL_PATH = "best_delay_regressor.joblib"      # optional
RISK_MODEL_PATH  = "best_risk_regressor.joblib"       # optional

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    p = Path(path)
    p_alt = Path("/mnt/data") / path
    if p.exists():
        return joblib.load(p)
    if p_alt.exists():
        return joblib.load(p_alt)
    return None

cls_model   = load_model(CLASSIFIER_PATH)
delay_model = load_model(DELAY_MODEL_PATH)
risk_model  = load_model(RISK_MODEL_PATH)

if cls_model is None:
    st.error("❌ Delay classifier not found. Make sure best_delay_classifier.joblib is in the same folder.")
    st.stop()


# Feature meanings & limits

with st.expander("ℹ️ Input features & meanings (please read)"):
    st.markdown("""
**Numeric**
- **setpoint_c (°C)**: Reefer temperature setpoint for the commodity. Typical: **2–10**.
- **quantity_kg (kg)**: Shipment weight. Typical air: **200–3500**, sea: **3000–28000**.
- **month (1–12)**: Month of departure (from ETD).
- **quarter (1–4)**: Quarter of the year.
- **freight_usd (USD)**: International freight cost (shipment-level).
- **insurance_usd (USD)**: Insurance cost (shipment-level).
- **inland_cost_inr (₹)**: Inland logistics cost in India (trucking, handling).

**Categorical / Boolean**
- **mode**: `"Sea"` or `"Air"`.
- **equipment_type**: `"Reefer 20'"`, `"Reefer 40'"` for Sea; `"AKE ULD"` for Air.
- **commodity**: Vegetable/herb shipped.
- **season_monsoon_IN**: Is Indian monsoon affecting ex-India leg? (Jun–Sep often **True**)
- **season_cyclone_ARABIAN**: Arabian Sea cyclone season? (mainly May–Jun, Oct–Nov)
- **season_heat_UAE**: UAE peak summer? (May–Sep)
- **season_fog_UAE**: UAE winter fog risk? (Dec–Feb)
""")

# Allowed options
COMMODITIES = ["Coriander", "Mint", "Green Chili", "Okra", "Curry Leaves"]
EQUIP_SEA   = ["Reefer 20'", "Reefer 40'"]
EQUIP_AIR   = ["AKE ULD"]


# Utility: Weather-aware funcs

def expected_temp_risk(row):
    score = 0
    if row["season_heat_UAE"]: score += 2
    if row["setpoint_c"] <= 5: score += 2
    if "Reefer 20" in str(row["equipment_type"]): score += 1
    if row["mode"] == "Air": score += 1
    if row["quantity_kg"] > 15000: score += 1
    return "High" if score >= 4 else ("Medium" if score >= 2 else "Low")

def monsoon_delay_prob(row):
    if not row["season_monsoon_IN"]:
        return 0.05
    base = 0.35 if row["mode"] == "Sea" else 0.25
    if row["quantity_kg"] > 10000: base += 0.10
    return min(base, 0.80)

def cyclone_delay_days(row):
    return 1.0 if (row["season_cyclone_ARABIAN"] and row["mode"] == "Sea") else 0.0

def fog_delay_prob(row):
    return 0.35 if (row["season_fog_UAE"] and row["mode"] == "Air") else 0.03

def weather_flag(row):
    if row["season_cyclone_ARABIAN"]: return "Cyclone"
    if row["season_monsoon_IN"]:      return "Monsoon"
    if row["season_heat_UAE"]:        return "Heat"
    if row["season_fog_UAE"]:         return "Fog"
    return "Normal"


# Manual input form

st.subheader("✍️ Enter 1 or more shipments")
n_rows = st.number_input("How many shipments to enter?", min_value=1, max_value=50, value=3, step=1)
rows = []

for i in range(int(n_rows)):
    with st.expander(f"Shipment #{i+1}", expanded=(i == 0)):
        # Left / Right columns for neat layout
        c1, c2, c3, c4 = st.columns(4)

        with c1:
            mode = st.selectbox("mode", ["Sea", "Air"], key=f"mode_{i}",
                                help="Transport mode. Sea for containers; Air for ULD.")
            # Equipment options depend on mode
            eq_options = EQUIP_SEA if mode == "Sea" else EQUIP_AIR
            equipment_type = st.selectbox("equipment_type", eq_options, key=f"equip_{i}",
                                          help="Container/ULD type")
            commodity = st.selectbox("commodity", COMMODITIES, key=f"comm_{i}",
                                     help="Type of vegetable/herb shipped")

        with c2:
            setpoint_c = st.number_input("setpoint_c (°C)", min_value=2.0, max_value=10.0, value=4.0, step=0.5, key=f"sp_{i}",
                                         help="Reefer temperature setpoint (typical 2–10°C)")
            quantity_kg = st.number_input("quantity_kg (kg)", min_value=200, max_value=28000, value=12000 if mode=="Sea" else 800, step=100, key=f"qty_{i}",
                                          help="Shipment weight (Air 200–3500; Sea 3000–28000)")

        with c3:
            month = st.number_input("month (1–12)", min_value=1, max_value=12, value=7, step=1, key=f"month_{i}",
                                    help="Month of ETD")
            quarter = st.number_input("quarter (1–4)", min_value=1, max_value=4, value=3, step=1, key=f"q_{i}",
                                      help="Quarter of ETD")

        with c4:
            freight_usd  = st.number_input("freight_usd (USD)", min_value=50.0, max_value=50000.0, value=1000.0, step=50.0, key=f"fr_{i}",
                                           help="International freight cost")
            insurance_usd = st.number_input("insurance_usd (USD)", min_value=0.0, max_value=2000.0, value=5.0, step=1.0, key=f"ins_{i}",
                                            help="Insurance cost")
            inland_cost_inr = st.number_input("inland_cost_inr (₹)", min_value=5000, max_value=200000, value=25000, step=500, key=f"inr_{i}",
                                              help="Inland India cost (trucking/handling)")

        # Season flags
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            season_monsoon_IN = st.checkbox("season_monsoon_IN", value=(month in [6,7,8,9]), key=f"mon_{i}",
                                            help="Heavy rain risk ex-India (Jun–Sep)")
        with s2:
            season_cyclone_ARABIAN = st.checkbox("season_cyclone_ARABIAN", value=(month in [5,6,10,11]), key=f"cyc_{i}",
                                                 help="Arabian Sea cyclone windows (May–Jun, Oct–Nov)")
        with s3:
            season_heat_UAE = st.checkbox("season_heat_UAE", value=(month in [5,6,7,8,9]), key=f"heat_{i}",
                                          help="UAE peak summer (May–Sep)")
        with s4:
            season_fog_UAE = st.checkbox("season_fog_UAE", value=(month in [12,1,2]), key=f"fog_{i}",
                                         help="UAE winter fog risk (Dec–Feb; affects Air more)")

        rows.append({
            "mode": mode,
            "equipment_type": equipment_type,
            "commodity": commodity,
            "setpoint_c": float(setpoint_c),
            "quantity_kg": int(quantity_kg),
            "month": int(month),
            "quarter": int(quarter),
            "freight_usd": float(freight_usd),
            "insurance_usd": float(insurance_usd),
            "inland_cost_inr": int(inland_cost_inr),
            "season_monsoon_IN": bool(season_monsoon_IN),
            "season_cyclone_ARABIAN": bool(season_cyclone_ARABIAN),
            "season_heat_UAE": bool(season_heat_UAE),
            "season_fog_UAE": bool(season_fog_UAE)
        })

df_in = pd.DataFrame(rows)

st.markdown("#### Your Inputs")
st.dataframe(df_in, use_container_width=True)

# Predict

if st.button("🔮 Predict"):
    with st.spinner("Scoring..."):
        # 1) Classification
        pred_cls = cls_model.predict(df_in)
        out = df_in.copy()
        out["Predicted_Status"] = np.where(pred_cls == 1, "On-Time", "Delayed")

        # 2) Regression (if available)
        if delay_model is not None:
            out["Predicted_DelayDays"] = np.round(delay_model.predict(df_in), 1)
        else:
            out["Predicted_DelayDays"] = np.nan

        if risk_model is not None:
            out["Predicted_RiskScore"] = np.round(risk_model.predict(df_in), 1)
        else:
            # Simple fallback if risk regressor isn't provided
            scores = []
            for _, r in out.iterrows():
                s = 20
                if r["season_monsoon_IN"]: s += 15
                if r["season_cyclone_ARABIAN"] and r["mode"] == "Sea": s += 10
                if r["season_heat_UAE"]: s += 20
                if r["season_fog_UAE"] and r["mode"] == "Air": s += 10
                scores.append(np.clip(s, 0, 100))
            out["Predicted_RiskScore"] = scores

        # 3) Weather-aware outputs
        out["Weather_Flag"]       = out.apply(weather_flag, axis=1)
        out["Monsoon_Delay_Prob"] = np.round(out.apply(monsoon_delay_prob, axis=1), 2)
        out["Cyclone_Delay_Days"] = np.round(out.apply(cyclone_delay_days, axis=1), 1)
        out["Fog_Delay_Prob"]     = np.round(out.apply(fog_delay_prob, axis=1), 2)
        out["Expected_Temp_Risk"] = out.apply(expected_temp_risk, axis=1)

    st.subheader("✅ Predictions")
    show_cols = [
        "commodity","mode","equipment_type","quantity_kg","setpoint_c",
        "month","quarter",
        "season_monsoon_IN","season_cyclone_ARABIAN","season_heat_UAE","season_fog_UAE",
        "Predicted_Status","Predicted_DelayDays","Predicted_RiskScore",
        "Weather_Flag","Monsoon_Delay_Prob","Cyclone_Delay_Days","Fog_Delay_Prob","Expected_Temp_Risk"
    ]
    st.dataframe(out[show_cols], use_container_width=True)

    # Charts
    st.markdown("### 📊 Visual Insights")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.pie(out, names="Predicted_Status", title="On-Time vs Delayed", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(out, x="Predicted_DelayDays", nbins=20, title="Predicted Delay Days")
        st.plotly_chart(fig, use_container_width=True)
    with c3:
        fig = px.histogram(out, x="Predicted_RiskScore", nbins=20, title="Predicted Risk Score (0–100)")
        st.plotly_chart(fig, use_container_width=True)

    c4, c5 = st.columns(2)
    with c4:
        fig = px.box(out, x="mode", y="Predicted_DelayDays", color="mode", title="Delay Days by Mode")
        st.plotly_chart(fig, use_container_width=True)
    with c5:
        fig = px.box(out, x="commodity", y="Predicted_RiskScore", color="commodity", title="Risk Score by Commodity")
        st.plotly_chart(fig, use_container_width=True)

    c6, c7 = st.columns(2)
    with c6:
        fig = px.bar(out.groupby("Weather_Flag")["Predicted_DelayDays"].mean().reset_index(),
                     x="Weather_Flag", y="Predicted_DelayDays", title="Avg Predicted Delay by Weather Flag")
        st.plotly_chart(fig, use_container_width=True)
    with c7:
        fig = px.bar(out.groupby("Weather_Flag")["Predicted_RiskScore"].mean().reset_index(),
                     x="Weather_Flag", y="Predicted_RiskScore", title="Avg Predicted Risk by Weather Flag")
        st.plotly_chart(fig, use_container_width=True)

    st.success("Done! Adjust inputs above to simulate different scenarios.")
else:
    st.info("Set inputs above and click **Predict**.")
