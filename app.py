
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ── Load Model & Preprocessor ──────────────────────
@st.cache_resource
def load_model():
    with open("final_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor

model, preprocessor = load_model()

REFERENCE_YEAR = 2020

def add_features(data):
    data = data.copy()
    data["car_age"]         = REFERENCE_YEAR - data["year"]
    data["mileage_per_year"]= data["mileage"] / data["car_age"].replace(0, 1)
    data["power_proxy"]     = data["enginesize"] / (data["mpg"] + 1)
    data["tax_per_mpg"]     = data["tax"] / (data["mpg"] + 1)
    return data

# ── Page Config ────────────────────────────────────
st.set_page_config(
    page_title = "AutoWorth AI",
    page_icon  = "🚗",
    layout     = "centered"
)

# ── Header ─────────────────────────────────────────
st.title("🚗 AutoWorth AI")
st.markdown("### Used Car Price & Deal Advisor")
st.markdown("Enter the car details below to get the estimated market price and deal rating.")
st.divider()

# ── Input Form ─────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    make = st.selectbox("Make", [
        "audi","bmw","ford","hyundai",
        "mercedes","skoda","toyota","vauxhall","volkswagen"
    ])
    year        = st.number_input("Year",         min_value=1990, max_value=2024, value=2018)
    mileage     = st.number_input("Mileage",      min_value=0,    max_value=300000, value=25000, step=1000)
    transmission= st.selectbox("Transmission",    ["Manual","Automatic","Semi-Auto"])
    fueltype    = st.selectbox("Fuel Type",       ["Petrol","Diesel","Hybrid","Electric"])

with col2:
    model_name  = st.text_input("Model (e.g. A3, Focus)", value="A3")
    enginesize  = st.number_input("Engine Size (L)", min_value=0.5, max_value=6.5, value=1.4, step=0.1)
    tax         = st.number_input("Road Tax (£)",    min_value=0,   max_value=600,  value=145, step=5)
    mpg         = st.number_input("MPG",             min_value=10.0,max_value=200.0,value=55.4,step=0.1)
    seller_price= st.number_input("Seller Price (£)",min_value=500, max_value=150000,value=14000,step=500)

st.divider()

# ── Predict Button ─────────────────────────────────
if st.button("🔍 Analyse Deal", use_container_width=True, type="primary"):

    input_data = pd.DataFrame([{
        "model":        model_name,
        "year":         year,
        "transmission": transmission,
        "mileage":      mileage,
        "fueltype":     fueltype,
        "tax":          float(tax),
        "mpg":          float(mpg),
        "enginesize":   float(enginesize),
        "make":         make.lower()
    }])

    input_data      = add_features(input_data)
    input_processed = preprocessor.transform(input_data)
    predicted_price = model.predict(input_processed)[0]

    diff     = predicted_price - seller_price
    diff_pct = (diff / predicted_price) * 100

    if diff_pct > 10:
        rating = "🟢 GREAT DEAL"
        color  = "green"
    elif diff_pct > 5:
        rating = "🟢 GOOD DEAL"
        color  = "green"
    elif diff_pct >= -5:
        rating = "🟡 FAIR PRICE"
        color  = "orange"
    elif diff_pct >= -10:
        rating = "🟠 SLIGHTLY OVERPRICED"
        color  = "orange"
    else:
        rating = "🔴 OVERPRICED"
        color  = "red"

    # ── Results ────────────────────────────────────
    st.subheader("📊 Results")

    c1, c2, c3 = st.columns(3)
    c1.metric("💰 Market Price",  f"£{predicted_price:,.0f}")
    c2.metric("🏷️ Seller Price",  f"£{seller_price:,.0f}")
    c3.metric("💵 Difference",
              f"£{abs(diff):,.0f}",
              f"{'cheaper ✅' if diff > 0 else 'more expensive ❌'}")

    st.divider()
    st.markdown(f"## 🚦 Deal Rating")
    st.markdown(f"# :{color}[{rating}]")
    st.markdown(f"**{abs(diff_pct):.1f}% {'below' if diff > 0 else 'above'} estimated market price**")

    # ── Progress bar ───────────────────────────────
    score = max(0, min(100, int(50 + diff_pct * 2)))
    st.progress(score, text=f"Deal Score: {score}/100")
