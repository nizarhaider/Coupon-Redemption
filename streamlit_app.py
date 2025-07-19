import streamlit as st
import pandas as pd
import pickle
import os

# --------- Default values ---------
default_values = {
    'campaign_type': 0,
    'num_items': 6,
    'mode_brand': 782,
    'mode_brand_type': 0,
    'mode_category': 3,
    'age_range': 1,
    'rented': 0.0,
    'family_size': 3,
    'income_bracket': 4,
    'no_bought_items': 15973.0,
    'total_cost': 119287.65,
    'total_discount': -32217.11,
    'total_coupon_discount': -1321.50,
    'campaign_duration': 47,
    'day_of_year': 139,
    'week_of_year': 20,
    'month_of_year': 5
}

st.set_page_config(page_title="Customer Coupon Redemption Predictor", layout="wide")
st.title("🎯 Customer Coupon Redemption Predictor")

st.sidebar.header("Edit Customer & Campaign Features (Most Impactful at Top)")

# Sorted by importance
family_size = st.sidebar.slider("Family Size", 1, 10, int(default_values['family_size']))
total_coupon_discount = st.sidebar.slider("Total Coupon Discount", -5000, 0, int(default_values['total_coupon_discount']))
num_items = st.sidebar.slider("Num Items", 1, 1000, int(default_values['num_items']))
campaign_type_label = st.sidebar.selectbox(
    "Campaign Type",
    options=["X", "Y"],
    index=default_values['campaign_type']
)
campaign_type = 0 if campaign_type_label == "X" else 1
rented = st.sidebar.select_slider("Rented (0=No, 1=Yes)", options=[0.0, 1.0], value=float(default_values['rented']))
total_discount = st.sidebar.slider("Total Discount", -100000, 0, int(default_values['total_discount']))
mode_brand = st.sidebar.slider("Mode Brand", 0, 2000, int(default_values['mode_brand']))
month_of_year = st.sidebar.slider("Month of Year", 1, 12, int(default_values['month_of_year']))

with st.sidebar.expander("Advanced Transaction Features (Low Impact)", expanded=False):
    mode_brand_type = st.slider("Mode Brand Type", 0, 5, int(default_values['mode_brand_type']))
    mode_category = st.slider("Mode Category", 0, 10, int(default_values['mode_category']))
    age_range = st.slider("Age Range", 1, 5, int(default_values['age_range']))
    income_bracket = st.slider("Income Bracket", 1, 5, int(default_values['income_bracket']))
    no_bought_items = st.slider("No Bought Items", 0, 50000, int(default_values['no_bought_items']))
    total_cost = st.slider("Total Cost", 0, 500000, int(default_values['total_cost']))
    campaign_duration = st.slider("Campaign Duration", 1, 365, int(default_values['campaign_duration']))
    day_of_year = st.slider("Day of Year", 1, 366, int(default_values['day_of_year']))
    week_of_year = st.slider("Week of Year", 1, 53, int(default_values['week_of_year']))

# --------- Collect input ---------
input_dict = {
    'campaign_type': campaign_type,
    'num_items': num_items,
    'mode_brand': mode_brand,
    'mode_brand_type': mode_brand_type,
    'mode_category': mode_category,
    'age_range': age_range,
    'rented': rented,
    'family_size': family_size,
    'income_bracket': income_bracket,
    'no_bought_items': no_bought_items,
    'total_cost': total_cost,
    'total_discount': total_discount,
    'total_coupon_discount': total_coupon_discount,
    'campaign_duration': campaign_duration,
    'day_of_year': day_of_year,
    'week_of_year': week_of_year,
    'month_of_year': month_of_year
}
input_df = pd.DataFrame([input_dict])

st.write("#### Input Data Summary")
st.dataframe(input_df, use_container_width=True)

st.markdown("---")

# --------- Prediction & Probability ---------
model_path = os.path.join("model", "best_dt_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(input_df)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0][1]  # Probability of class '1'
    else:
        proba = None

    result = (
        "🎉 <span style='color:green'>Customer WILL redeem the coupon!</span>"
        if prediction == 1 else
        "⚠️ <span style='color:red'>Customer will NOT redeem the coupon.</span>"
    )
    st.markdown(f"<h3>Prediction Result:</h3>{result}", unsafe_allow_html=True)
    if proba is not None:
        st.progress(proba)
        st.markdown(
            f"**Probability of Redemption:** <span style='font-size:20px'>{proba:.2%}</span>",
            unsafe_allow_html=True
        )
else:
    st.error("Model file not found. Please check 'model/best_dt_model.pkl'.")

# --------- Feature Importance Advice ---------
st.markdown("### 🔑 Features that most influence redemption:")

feature_advices = {
    "family_size": "Increasing family size increases the chance.",
    "total_coupon_discount": "Higher coupon discount makes redemption more likely.",
    "num_items": "Adding more items to the cart increases likelihood.",
    "campaign_type": "Switching campaign type may help if possible.",
    "rented": "Rented customers tend to redeem more."
}

for feat in ["family_size", "total_coupon_discount", "num_items", "campaign_type", "rented"]:
    val = input_dict[feat]
    st.write(f"**{feat.replace('_', ' ').title()}**: {val} — {feature_advices[feat]}")

st.info("Try increasing the above values and watch the probability update!")
