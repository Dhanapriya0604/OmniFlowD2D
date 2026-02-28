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

st.title("📦 OmniFlow-D2D — Supply Chain Intelligence")

# ======================================================================================
# LOAD DATA (FIXED 🔥)
# ======================================================================================
@st.cache_data
def load_data():

    df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv")
    df.columns = [c.strip() for c in df.columns]

    # Convert date
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    # Rename for model compatibility
    df = df.rename(columns={
        "Order_Date": "Date",
        "Product_Name": "Product",
        "Quantity": "Demand"
    })

    # Feature Engineering
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    df["Week"] = df["Date"].dt.to_period("W").astype(str)
    df["Day_of_Week"] = df["Date"].dt.day_name()

    df["day"] = df["Date"].dt.day
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["weekday"] = df["Date"].dt.weekday

    # Ensure required columns
    if "Revenue_INR" not in df.columns:
        df["Revenue_INR"] = df["Demand"] * df.get("Sell_Price", 100)

    if "Shipping_Cost_INR" not in df.columns:
        df["Shipping_Cost_INR"] = np.random.uniform(50,120,len(df))

    if "Region" not in df.columns:
        df["Region"] = "India"

    if "Delivery_Days" not in df.columns:
        df["Delivery_Days"] = np.random.randint(3,7,len(df))

    if "Order_Status" not in df.columns:
        df["Order_Status"] = "Delivered"

    if "Return_Flag" not in df.columns:
        df["Return_Flag"] = 0

    if "Courier_Partner" not in df.columns:
        df["Courier_Partner"] = "Delhivery"

    if "Warehouse" not in df.columns:
        df["Warehouse"] = "Default WH"

    if "Category" not in df.columns:
        df["Category"] = "General"

    # Logistics mapping
    df["Logistics"] = df["Shipping_Cost_INR"]

    return df.sort_values("Date").reset_index(drop=True)

# -----------------------------
# LOAD
# -----------------------------
df = load_data()

# ======================================================================================
# FEATURE ENGINEERING (TIME SERIES)
# ======================================================================================
df['lag_1'] = df['Demand'].shift(1)
df['lag_7'] = df['Demand'].shift(7)
df['rolling_mean_7'] = df['Demand'].rolling(7).mean()
df['rolling_std_7'] = df['Demand'].rolling(7).std()

df = df.dropna()

# Encoding
le_p = LabelEncoder()
le_r = LabelEncoder()

df['product_enc'] = le_p.fit_transform(df['Product'])
df['region_enc'] = le_r.fit_transform(df['Region'])

# ======================================================================================
# MODEL
# ======================================================================================
features = [
    'product_enc','region_enc','day','month','year','weekday',
    'lag_1','lag_7','rolling_mean_7','rolling_std_7'
]

train = df[df['Date'] < df['Date'].max() - pd.Timedelta(days=30)]
test = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=30)]

X_train = train[features]
y_train = train['Demand']

X_test = test[features]
y_test = test['Demand']

model = RandomForestRegressor(n_estimators=200, max_depth=10)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred_test))

# ======================================================================================
# SIDEBAR
# ======================================================================================
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

    st.metric("RMSE", round(rmse,2))

    fig = px.line(df, x='Date', y='Demand')
    st.plotly_chart(fig, use_container_width=True)

# ======================================================================================
# FORECASTING
# ======================================================================================
elif module == "Forecasting":

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

        pred = model.predict(pd.DataFrame([row]))[0]
        preds.append(pred)

        row['Demand'] = pred
        history = pd.concat([history, pd.DataFrame([row])], ignore_index=True)

    result = pd.DataFrame({
        "Date": future_dates,
        "Forecast": preds
    })

    st.dataframe(result)
    st.line_chart(result.set_index("Date"))

# ======================================================================================
# INVENTORY
# ======================================================================================
elif module == "Inventory":

    demand_std = df['Demand'].std()
    lead_time = 7

    df['Safety_Stock'] = 1.65 * demand_std * np.sqrt(lead_time)
    df['Reorder_Point'] = df['Demand'] * lead_time + df['Safety_Stock']

    st.dataframe(df[['Product','Demand','Safety_Stock','Reorder_Point']])

# ======================================================================================
# PRODUCTION
# ======================================================================================
elif module == "Production":

    df['Production'] = df['Demand'] * 1.15
    st.dataframe(df[['Product','Demand','Production']])

# ======================================================================================
# LOGISTICS
# ======================================================================================
elif module == "Logistics":

    df['Cost'] = df['Logistics'] * 1.2

    fig = px.scatter(df, x='Region', y='Cost',
                     size='Demand', color='Product')
    st.plotly_chart(fig)

# ======================================================================================
# DECISION AI (CHATBOT)
# ======================================================================================
elif module == "Decision AI":

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user = st.text_input("Ask business question")

    if user:

        q = user.lower()

        if "forecast" in q:
            response = f"RMSE: {round(rmse,2)}. Forecast is reliable short-term."

        elif "inventory" in q:
            p = df.sort_values('Reorder_Point', ascending=False).iloc[0]['Product']
            response = f"Increase stock for {p}"

        elif "risk" in q:
            vol = df['Demand'].std()
            response = f"Demand volatility: {round(vol,2)}"

        elif "production" in q:
            response = "Increase production by 15% buffer"

        elif "logistics" in q:
            r = df.groupby('Region')['Cost'].mean().idxmax()
            response = f"High cost region: {r}"

        else:
            response = "Ask about forecast, inventory, logistics, or production"

        st.session_state.chat.append(("You", user))
        st.session_state.chat.append(("AI", response))

    for r, m in st.session_state.chat:
        st.write(f"**{r}:** {m}")
