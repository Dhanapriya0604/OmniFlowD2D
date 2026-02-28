# ==========================================================
# OmniFlow FINAL INDUSTRY VERSION 🚀
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="OmniFlow AI", layout="wide")

# ==========================================================
# LOAD DATA
# ==========================================================
@st.cache_data
def load_data():
    df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv")
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    return df

# ==========================================================
# PREPROCESSING
# ==========================================================
def preprocess(df):
    df = df.drop_duplicates()
    df = df[df["Quantity"] > 0]
    df = df[df["Revenue_INR"] > 0]
    df.fillna(method="ffill", inplace=True)
    return df

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================
def feature_engineering(df):
    df["Month"] = df["Order_Date"].dt.month
    df["Year"] = df["Order_Date"].dt.year
    df["Weekday"] = df["Order_Date"].dt.weekday

    df["Price_per_Unit"] = df["Revenue_INR"] / df["Quantity"]

    df["Daily_Demand"] = df.groupby("Order_Date")["Quantity"].transform("sum")

    return df

# ==========================================================
# FORECAST FUNCTION (DYNAMIC)
# ==========================================================
def forecast(df, product, start, end):

    df_p = df[df["Product_Name"] == product]

    ts = df_p.set_index("Order_Date")["Quantity"].resample("M").sum()
    ts = ts.asfreq("M", fill_value=0)

    steps = ((end.year - ts.index.max().year)*12 +
             (end.month - ts.index.max().month))

    if steps <= 0:
        return None

    try:
        model = SARIMAX(ts, order=(1,1,1), seasonal_order=(1,1,1,12))
        res = model.fit(disp=False)
        pred = res.get_forecast(steps=steps).predicted_mean
    except:
        x = np.arange(len(ts))
        coef = np.polyfit(x, ts.values, 1)
        pred = np.poly1d(coef)(np.arange(len(ts), len(ts)+steps))

    dates = pd.date_range(start=ts.index.max(), periods=steps+1, freq="M")[1:]

    fc = pd.DataFrame({"Date": dates, "Forecast": pred})
    fc = fc[(fc["Date"] >= start) & (fc["Date"] <= end)]

    return fc

# ==========================================================
# INVENTORY MODEL
# ==========================================================
def inventory(df):

    g = df.groupby("Product_Name")["Quantity"]

    inv = pd.DataFrame({
        "Avg Demand": g.mean(),
        "Std Dev": g.std()
    })

    inv["Safety Stock"] = 1.65 * inv["Std Dev"]
    inv["Reorder Point"] = inv["Avg Demand"] + inv["Safety Stock"]

    return inv.reset_index()

# ==========================================================
# ML MODELS
# ==========================================================
def train_ml(df):

    features = ["Month", "Weekday", "Price_per_Unit"]
    X = df[features]
    y = df["Quantity"]

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, X, y

# ==========================================================
# EVALUATION
# ==========================================================
def evaluate(model, X, y):

    pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, pred))
    r2 = r2_score(y, pred)

    return rmse, r2

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================
def importance(model, features):

    return pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

# ==========================================================
# MAIN APP
# ==========================================================
df = load_data()
df = preprocess(df)
df = feature_engineering(df)

page = st.sidebar.radio("Navigation",
    ["Overview","Forecasting","Inventory","Production","Logistics","Model Insights"])

# ==========================================================
# OVERVIEW
# ==========================================================
if page == "Overview":

    st.title("📊 Overview")

    st.metric("Revenue", f"₹{df['Revenue_INR'].sum():,.0f}")
    st.metric("Orders", len(df))

    fig = px.bar(df.groupby("Category")["Revenue_INR"].sum().reset_index(),
                 x="Category", y="Revenue_INR")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# FORECASTING
# ==========================================================
elif page == "Forecasting":

    st.title("📈 Forecasting")

    product = st.selectbox("Product", df["Product_Name"].unique())
    start = st.date_input("Start Date")
    end = st.date_input("End Date")

    if st.button("Forecast"):
        fc = forecast(df, product, start, end)

        if fc is None:
            st.warning("Invalid range")
        else:
            st.dataframe(fc)

            fig = px.line(fc, x="Date", y="Forecast")
            st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# INVENTORY
# ==========================================================
elif page == "Inventory":

    st.title("📦 Inventory")

    inv = inventory(df)
    st.dataframe(inv)

# ==========================================================
# PRODUCTION (ML)
# ==========================================================
elif page == "Production":

    st.title("🏭 Production ML")

    model, X, y = train_ml(df)

    rmse, r2 = evaluate(model, X, y)

    st.metric("RMSE", round(rmse,2))
    st.metric("R2 Score", round(r2,2))

    future = pd.DataFrame({
        "Month":[6],
        "Weekday":[2],
        "Price_per_Unit":[500]
    })

    pred = model.predict(future)

    st.metric("Predicted Production", int(pred[0]))

# ==========================================================
# LOGISTICS (ML)
# ==========================================================
elif page == "Logistics":

    st.title("🚚 Logistics ML")

    df_enc = pd.get_dummies(df[["Region","Courier_Partner"]])

    X = df_enc
    y = df["Delivery_Days"]

    model = RandomForestRegressor()
    model.fit(X,y)

    st.success("Model trained to predict delivery days")

# ==========================================================
# MODEL INSIGHTS
# ==========================================================
elif page == "Model Insights":

    st.title("📊 Model Insights")

    model, X, y = train_ml(df)

    imp = importance(model, X.columns)

    st.dataframe(imp)

    fig = px.bar(imp, x="Feature", y="Importance")
    st.plotly_chart(fig, use_container_width=True)
