# ======================================================================================
# OmniFlow-D2D : Industry-Level Supply Chain Intelligence
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="OmniFlow-D2D", layout="wide")

st.markdown("""
<style>
body {background:#f9fafb; color:#111827; font-family:Segoe UI;}
section[data-testid="stSidebar"] {background:#ffffff;}
</style>
""", unsafe_allow_html=True)

st.title("📦 OmniFlow-D2D — Supply Chain Intelligence")

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['weekday'] = df['Date'].dt.weekday

# Lag features (IMPORTANT 🔥)
df['lag_1'] = df['Demand'].shift(1)
df['lag_7'] = df['Demand'].shift(7)

# Rolling features
df['rolling_mean_7'] = df['Demand'].rolling(7).mean()
df['rolling_std_7'] = df['Demand'].rolling(7).std()

df = df.dropna()

# Encoding
le_p = LabelEncoder()
le_r = LabelEncoder()

df['product_enc'] = le_p.fit_transform(df['Product'])
df['region_enc'] = le_r.fit_transform(df['Region'])

# -----------------------------
# MODEL FEATURES
# -----------------------------
features = [
    'product_enc','region_enc','day','month','year','weekday',
    'lag_1','lag_7','rolling_mean_7','rolling_std_7'
]

# -----------------------------
# TRAIN / TEST SPLIT (TIME BASED)
# -----------------------------
train = df[df['Date'] < df['Date'].max() - pd.Timedelta(days=30)]
test = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=30)]

X_train = train[features]
y_train = train['Demand']

X_test = test[features]
y_test = test['Demand']

model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

# Evaluation
pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

# -----------------------------
# SIDEBAR
# -----------------------------
module = st.sidebar.selectbox("Module", [
    "Overview",
    "Forecasting",
    "Inventory",
    "Production",
    "Logistics",
    "Decision AI"
])

# ======================================================================================
# OVERVIEW
# ======================================================================================
if module == "Overview":

    st.metric("RMSE (Model Accuracy)", round(rmse,2))

    fig = px.line(df, x='Date', y='Demand')
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# FORECASTING (ITERATIVE 🔥)
# ======================================================================================
elif module == "Forecasting":

    st.subheader("📈 Demand Forecasting (Real ML)")

    product = st.selectbox("Product", df['Product'].unique())
    region = st.selectbox("Region", df['Region'].unique())

    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")

    future_dates = pd.date_range(start=start_date, end=end_date)

    history = df.copy()

    preds = []

    for d in future_dates:

        last = history.iloc[-1]

        row = {
            'product_enc': le_p.transform([product])[0],
            'region_enc': le_r.transform([region])[0],
            'day': d.day,
            'month': d.month,
            'year': d.year,
            'weekday': d.weekday(),
            'lag_1': last['Demand'],
            'lag_7': history.iloc[-7]['Demand'] if len(history)>=7 else last['Demand'],
            'rolling_mean_7': history['Demand'].tail(7).mean(),
            'rolling_std_7': history['Demand'].tail(7).std()
        }

        row_df = pd.DataFrame([row])
        pred = model.predict(row_df)[0]

        preds.append(pred)

        # Append for next step
        new_row = row.copy()
        new_row['Demand'] = pred
        history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)

    result = pd.DataFrame({
        "Date": future_dates,
        "Forecast": preds
    })

    st.dataframe(result)

    fig = px.line(result, x='Date', y='Forecast')
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# INVENTORY (REAL FORMULA)
# ======================================================================================
elif module == "Inventory":

    st.subheader("📦 Inventory Optimization")

    demand_std = df['Demand'].std()
    lead_time = 7

    df['Safety_Stock'] = 1.65 * demand_std * np.sqrt(lead_time)
    df['Reorder_Point'] = df['Demand'] * lead_time + df['Safety_Stock']

    st.dataframe(df[['Product','Demand','Safety_Stock','Reorder_Point']].head())

# ======================================================================================
# PRODUCTION
# ======================================================================================
elif module == "Production":

    st.subheader("🏭 Production Planning")

    df['Production'] = df['Demand'] * 1.15

    st.dataframe(df[['Product','Demand','Production']].head())

# ======================================================================================
# LOGISTICS
# ======================================================================================
elif module == "Logistics":

    st.subheader("🚚 Logistics")

    df['Cost'] = df['Logistics'] * 1.2

    fig = px.scatter(df, x='Region', y='Cost', size='Demand')
    st.plotly_chart(fig)

# ======================================================================================
# DECISION AI (SMART 🔥)
# ======================================================================================
elif module == "Decision AI":

    st.subheader("🤖 Decision Intelligence Engine")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user = st.text_input("Ask a business question")

    if user:

        q = user.lower()

        if "forecast" in q:
            response = f"Model RMSE is {round(rmse,2)}. Forecast is reliable for short-term planning."

        elif "inventory" in q:
            high = df.sort_values('Reorder_Point', ascending=False).iloc[0]
            response = f"Increase inventory for {high['Product']} (high reorder point)."

        elif "risk" in q:
            vol = df['Demand'].std()
            response = f"Demand volatility is {round(vol,2)}. Maintain buffer stock."

        elif "production" in q:
            response = "Production should be increased by ~15% above forecast to avoid shortages."

        else:
            response = "Ask about forecast, inventory, production, or risk."

        st.session_state.chat.append(("You", user))
        st.session_state.chat.append(("AI", response))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
