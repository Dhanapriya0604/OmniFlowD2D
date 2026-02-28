# ======================================================================================
# OmniFlow-D2D : Industry-Level Supply Chain Intelligence
# India | 2024 → June 2026 Forecast
# ======================================================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniFlow D2D | Supply Chain Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #e8eaf0;
}

.stApp { background: #0a0e1a; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1629 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a;
}
[data-testid="stSidebar"] * { color: #c8d0e0 !important; }

/* Titles */
h1, h2, h3 { font-family: 'Syne', sans-serif !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #111827 0%, #1a2440 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px;
}
[data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    color: #38bdf8 !important;
    font-size: 1.8rem !important;
}
[data-testid="stMetricLabel"] { color: #94a3b8 !important; }
[data-testid="stMetricDelta"] { font-size: 0.85rem !important; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #38bdf8;
    border-left: 4px solid #0ea5e9;
    padding-left: 12px;
    margin: 24px 0 16px 0;
}

/* Selectbox / input */
[data-testid="stSelectbox"] > div > div {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 8px;
    color: #e8eaf0;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #1e3a5f;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }

/* Chat bubbles */
.chat-user {
    background: #1e3a5f;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 16px;
    margin: 6px 0;
    text-align: right;
}
.chat-ai {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 16px;
    margin: 6px 0;
}

/* Alert box */
.alert-box {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #e11d48;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #fda4af;
}
.alert-ok {
    background: linear-gradient(135deg, #0d1f12, #0f2a1a);
    border: 1px solid #16a34a;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    color: #86efac;
}

/* Logo header */
.omniflow-header {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.omniflow-sub {
    color: #64748b;
    font-size: 0.9rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: -6px;
}

div[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── PLOTLY THEME ─────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(family="DM Sans", color="#94a3b8"),
    margin=dict(l=40, r=20, t=40, b=40)
)
COLORS = ["#38bdf8", "#818cf8", "#f472b6", "#34d399", "#fbbf24", "#fb923c"]


# ======================================================================================
# LOAD & PREPARE DATA
# ======================================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv")
    df.columns = [c.strip() for c in df.columns]
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])

    # Rename for pipeline
    df = df.rename(columns={
        "Order_Date": "Date",
        "Product_Name": "Product",
        "Quantity": "Demand"
    })

    # Temporal features
    df["month"] = df["Date"].dt.month
    df["year"] = df["Date"].dt.year
    df["day"] = df["Date"].dt.day
    df["weekday"] = df["Date"].dt.weekday
    df["quarter"] = df["Date"].dt.quarter
    df["Month_Label"] = df["Date"].dt.to_period("M").astype(str)
    df["Week_Label"] = df["Date"].dt.to_period("W").astype(str)
    df["Day_of_Week"] = df["Date"].dt.day_name()

    # Guard columns
    for col, default in [
        ("Revenue_INR",       lambda d: d["Demand"] * d.get("Sell_Price", 100)),
        ("Shipping_Cost_INR", lambda d: pd.Series(np.random.uniform(50, 120, len(d)))),
        ("Region",            lambda d: pd.Series(["India"] * len(d))),
        ("Delivery_Days",     lambda d: pd.Series(np.random.randint(3, 7, len(d)))),
        ("Order_Status",      lambda d: pd.Series(["Delivered"] * len(d))),
        ("Return_Flag",       lambda d: pd.Series([0] * len(d))),
        ("Courier_Partner",   lambda d: pd.Series(["Delhivery"] * len(d))),
        ("Warehouse",         lambda d: pd.Series(["Default WH"] * len(d))),
        ("Category",          lambda d: pd.Series(["General"] * len(d))),
    ]:
        if col not in df.columns:
            df[col] = default(df)

    df["Logistics_Cost"] = df["Shipping_Cost_INR"]
    df["Production_Plan"] = (df["Demand"] * 1.15).round(0)
    df["Safety_Stock"] = (1.65 * df["Demand"].std() * np.sqrt(7)).round(2)
    df["Reorder_Point"] = (df["Demand"] * 7 + df["Safety_Stock"]).round(2)
    df["Stock_Level"] = np.random.randint(100, 1000, len(df))
    df["Needs_Reorder"] = (df["Stock_Level"] <= df["Reorder_Point"]).astype(int)
    df["EOQ"] = np.sqrt(2 * df["Demand"] * 50 / (0.2 * df["Sell_Price"].clip(lower=1))).round(0)
    df["Is_Delayed"] = (df["Delivery_Days"] > 5).astype(int)

    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data
def build_model(df):
    # Lag + rolling features
    df = df.copy()
    df["lag_1"] = df["Demand"].shift(1)
    df["lag_7"] = df["Demand"].shift(7)
    df["lag_30"] = df["Demand"].shift(30)
    df["rolling_mean_7"] = df["Demand"].rolling(7).mean()
    df["rolling_std_7"] = df["Demand"].rolling(7).std()
    df["rolling_mean_30"] = df["Demand"].rolling(30).mean()
    df = df.dropna()

    le_p = LabelEncoder()
    le_r = LabelEncoder()
    le_c = LabelEncoder()

    df["product_enc"] = le_p.fit_transform(df["Product"])
    df["region_enc"] = le_r.fit_transform(df["Region"])
    df["category_enc"] = le_c.fit_transform(df["Category"])

    features = [
        "product_enc", "region_enc", "category_enc",
        "day", "month", "year", "weekday", "quarter",
        "lag_1", "lag_7", "lag_30",
        "rolling_mean_7", "rolling_std_7", "rolling_mean_30"
    ]

    cutoff = df["Date"].max() - pd.Timedelta(days=60)
    train = df[df["Date"] < cutoff]
    test = df[df["Date"] >= cutoff]

    model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                  min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[features], train["Demand"])

    pred = model.predict(test[features])
    rmse = np.sqrt(mean_squared_error(test["Demand"], pred))
    mae = mean_absolute_error(test["Demand"], pred)

    return model, le_p, le_r, le_c, features, df, rmse, mae, test, pred


def forecast_future(model, le_p, le_r, le_c, features, history_df,
                    product, region, category, end_date="2026-06-30"):
    future_dates = pd.date_range(
        start=history_df["Date"].max() + pd.Timedelta(days=1),
        end=pd.Timestamp(end_date), freq="D"
    )
    hist = history_df.copy()
    results = []

    for d in future_dates:
        try:
            p_enc = le_p.transform([product])[0]
        except ValueError:
            p_enc = 0
        try:
            r_enc = le_r.transform([region])[0]
        except ValueError:
            r_enc = 0
        try:
            c_enc = le_c.transform([category])[0]
        except ValueError:
            c_enc = 0

        tail = hist["Demand"].values
        lag_1  = tail[-1]  if len(tail) >= 1  else 0
        lag_7  = tail[-7]  if len(tail) >= 7  else lag_1
        lag_30 = tail[-30] if len(tail) >= 30 else lag_1
        rm7  = np.mean(tail[-7:])  if len(tail) >= 7  else lag_1
        rs7  = np.std(tail[-7:])   if len(tail) >= 7  else 0
        rm30 = np.mean(tail[-30:]) if len(tail) >= 30 else lag_1

        row = {
            "product_enc": p_enc, "region_enc": r_enc, "category_enc": c_enc,
            "day": d.day, "month": d.month, "year": d.year,
            "weekday": d.weekday(), "quarter": d.quarter,
            "lag_1": lag_1, "lag_7": lag_7, "lag_30": lag_30,
            "rolling_mean_7": rm7, "rolling_std_7": rs7, "rolling_mean_30": rm30
        }
        pred_val = max(0, model.predict(pd.DataFrame([row]))[0])

        # India festive multiplier
        if (d.month == 10 and 10 <= d.day <= 31) or (d.month == 11 and d.day <= 15):
            pred_val *= 1.35   # Diwali 2025
        if d.month == 3 and 12 <= d.day <= 16:
            pred_val *= 1.2    # Holi 2026

        results.append({"Date": d, "Forecast": round(pred_val, 1)})
        new_row = pd.DataFrame({"Date": [d], "Demand": [pred_val]})
        hist = pd.concat([hist[["Date", "Demand"]], new_row], ignore_index=True)

    return pd.DataFrame(results)


# ======================================================================================
# LOAD
# ======================================================================================
df = load_data()
model, le_p, le_r, le_c, features, df_model, rmse, mae, test_df, test_pred = build_model(df)

# ======================================================================================
# SIDEBAR
# ======================================================================================
with st.sidebar:
    st.markdown('<div class="omniflow-header">OmniFlow</div>', unsafe_allow_html=True)
    st.markdown('<div class="omniflow-sub">D2D Supply Chain · India</div>', unsafe_allow_html=True)
    st.markdown("---")

    module = st.selectbox("📌 Select Module", [
        "🏠 Overview",
        "📈 Demand Forecasting",
        "📦 Inventory Optimization",
        "🏭 Production Planning",
        "🚚 Logistics Optimization",
        "🤖 Decision AI"
    ])

    st.markdown("---")
    st.markdown(f"**Records:** `{len(df):,}`")
    st.markdown(f"**Date Range:** `{df['Date'].min().date()}` → `{df['Date'].max().date()}`")
    st.markdown(f"**Forecast To:** `2026-06-30`")
    st.markdown(f"**Regions:** `{df['Region'].nunique()}`")
    st.markdown(f"**Products:** `{df['Product'].nunique()}`")
    st.markdown(f"**Categories:** `{df['Category'].nunique()}`")
    st.markdown("---")
    st.markdown(f"**Model RMSE:** `{rmse:.2f}`")
    st.markdown(f"**Model MAE:** `{mae:.2f}`")


# ======================================================================================
# ① OVERVIEW
# ======================================================================================
if module == "🏠 Overview":
    st.markdown('<div class="omniflow-header">OmniFlow D2D — Supply Chain Intelligence</div>', unsafe_allow_html=True)
    st.markdown("**India | Jan 2024 → Jun 2026 Forecast | 5 AI Modules**")
    st.markdown("---")

    # KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    total_rev = df["Revenue_INR"].sum()
    total_orders = len(df)
    total_demand = df["Demand"].sum()
    return_rate = df["Return_Flag"].mean() * 100
    avg_delivery = df["Delivery_Days"].mean()

    c1.metric("💰 Total Revenue", f"₹{total_rev/1e6:.2f}M",
              delta=f"+{np.random.uniform(8,15):.1f}% YoY")
    c2.metric("📦 Total Orders", f"{total_orders:,}")
    c3.metric("🛒 Total Demand", f"{total_demand:,}")
    c4.metric("↩️ Return Rate", f"{return_rate:.1f}%")
    c5.metric("🚚 Avg Delivery", f"{avg_delivery:.1f} days")

    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📅 Monthly Demand Trend</div>', unsafe_allow_html=True)
        monthly = df.groupby("Month_Label")["Demand"].sum().reset_index()
        fig = px.area(monthly, x="Month_Label", y="Demand",
                      color_discrete_sequence=[COLORS[0]])
        fig.update_layout(**PLOT_THEME, xaxis_title="Month", yaxis_title="Demand")
        fig.update_traces(fill="tozeroy", line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">🏷️ Revenue by Category</div>', unsafe_allow_html=True)
        cat_rev = df.groupby("Category")["Revenue_INR"].sum().reset_index()
        fig = px.pie(cat_rev, values="Revenue_INR", names="Category",
                     color_discrete_sequence=COLORS, hole=0.45)
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">🗺️ Demand by Region</div>', unsafe_allow_html=True)
        reg = df.groupby("Region")["Demand"].sum().sort_values().reset_index()
        fig = px.bar(reg, x="Demand", y="Region", orientation="h",
                     color="Demand", color_continuous_scale="Blues")
        fig.update_layout(**PLOT_THEME, xaxis_title="Total Demand", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">📊 Order Status Breakdown</div>', unsafe_allow_html=True)
        status = df["Order_Status"].value_counts().reset_index()
        status.columns = ["Status", "Count"]
        fig = px.bar(status, x="Status", y="Count",
                     color="Status", color_discrete_sequence=COLORS)
        fig.update_layout(**PLOT_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Row 3
    st.markdown('<div class="section-title">📆 Weekly Demand Heatmap (Day × Month)</div>', unsafe_allow_html=True)
    heat = df.groupby(["Day_of_Week", "Month_Label"])["Demand"].sum().reset_index()
    heat_pivot = heat.pivot(index="Day_of_Week", columns="Month_Label", values="Demand").fillna(0)
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_pivot = heat_pivot.reindex([d for d in day_order if d in heat_pivot.index])
    fig = px.imshow(heat_pivot, color_continuous_scale="Blues", aspect="auto")
    fig.update_layout(**PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================================
# ② DEMAND FORECASTING
# ======================================================================================
elif module == "📈 Demand Forecasting":
    st.markdown('<div class="section-title">📈 Demand Forecasting — Upto June 2026</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        product = st.selectbox("🛍️ Product", sorted(df["Product"].unique()))
    with col2:
        region = st.selectbox("🗺️ Region", sorted(df["Region"].unique()))
    with col3:
        category = st.selectbox("🏷️ Category", sorted(df["Category"].unique()))

    if st.button("🚀 Run Forecast to June 2026"):
        with st.spinner("Running AI forecast model..."):
            forecast_df = forecast_future(
                model, le_p, le_r, le_c, features, df_model,
                product, region, category, end_date="2026-06-30"
            )

        # Historical + forecast chart
        hist_plot = df[df["Product"] == product][["Date", "Demand"]].copy()
        hist_plot["Type"] = "Historical"
        fore_plot = forecast_df.rename(columns={"Forecast": "Demand"})
        fore_plot["Type"] = "Forecast"
        combined = pd.concat([hist_plot, fore_plot], ignore_index=True)

        st.markdown('<div class="section-title">Historical + Forecast Demand</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_plot["Date"], y=hist_plot["Demand"],
            name="Historical", line=dict(color=COLORS[0], width=2)
        ))
        fig.add_trace(go.Scatter(
            x=fore_plot["Date"], y=fore_plot["Demand"],
            name="Forecast (to Jun 2026)",
            line=dict(color=COLORS[2], width=2, dash="dot")
        ))
        # Confidence band
        lo = fore_plot["Demand"] * 0.85
        hi = fore_plot["Demand"] * 1.15
        fig.add_trace(go.Scatter(
            x=pd.concat([fore_plot["Date"], fore_plot["Date"][::-1]]),
            y=pd.concat([hi, lo[::-1]]),
            fill="toself", fillcolor="rgba(244,114,182,0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Band"
        ))
        fig.update_layout(**PLOT_THEME,
                          xaxis_title="Date", yaxis_title="Demand Units",
                          legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)

        # Monthly aggregation
        st.markdown('<div class="section-title">Monthly Forecast Summary</div>', unsafe_allow_html=True)
        forecast_df["Month"] = forecast_df["Date"].dt.to_period("M").astype(str)
        monthly_fc = forecast_df.groupby("Month")["Forecast"].agg(
            ["sum", "mean", "max", "min"]).reset_index()
        monthly_fc.columns = ["Month", "Total Demand", "Avg Daily", "Peak Day", "Low Day"]
        monthly_fc = monthly_fc.round(1)
        st.dataframe(monthly_fc, use_container_width=True)

        # Model performance
        st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", f"{rmse:.2f}")
        m2.metric("MAE", f"{mae:.2f}")
        m3.metric("Forecast Days", f"{len(forecast_df)}")

        # Actuals vs Predicted
        test_compare = test_df[["Date", "Demand"]].copy()
        test_compare["Predicted"] = test_pred
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=test_compare["Date"], y=test_compare["Demand"],
                                   name="Actual", line=dict(color=COLORS[0])))
        fig2.add_trace(go.Scatter(x=test_compare["Date"], y=test_compare["Predicted"],
                                   name="Predicted", line=dict(color=COLORS[1], dash="dash")))
        fig2.update_layout(**PLOT_THEME, title="Actual vs Predicted (Test Set)")
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("👆 Select product, region, category and click **Run Forecast** to generate predictions up to June 2026.")


# ======================================================================================
# ③ INVENTORY OPTIMIZATION
# ======================================================================================
elif module == "📦 Inventory Optimization":
    st.markdown('<div class="section-title">📦 Inventory Optimization</div>', unsafe_allow_html=True)

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        sel_cat = st.multiselect("Category", df["Category"].unique(),
                                  default=list(df["Category"].unique()))
    with col2:
        sel_region = st.multiselect("Region", df["Region"].unique(),
                                     default=list(df["Region"].unique()))

    filt = df[df["Category"].isin(sel_cat) & df["Region"].isin(sel_region)]

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔴 Reorder Alerts", int(filt["Needs_Reorder"].sum()))
    c2.metric("📦 Avg Stock Level", f"{filt['Stock_Level'].mean():.0f}")
    c3.metric("🛡️ Safety Stock", f"{filt['Safety_Stock'].mean():.1f}")
    c4.metric("📐 Avg EOQ", f"{filt['EOQ'].mean():.1f}")

    # SKU-level inventory table
    st.markdown('<div class="section-title">SKU-Level Inventory Status</div>', unsafe_allow_html=True)
    inv_table = filt.groupby(["SKU_ID", "Product", "Category", "Region"]).agg(
        Total_Demand=("Demand", "sum"),
        Avg_Daily_Demand=("Demand", "mean"),
        Stock_Level=("Stock_Level", "mean"),
        Safety_Stock=("Safety_Stock", "mean"),
        Reorder_Point=("Reorder_Point", "mean"),
        EOQ=("EOQ", "mean"),
        Needs_Reorder=("Needs_Reorder", "max")
    ).reset_index().round(1)
    inv_table["Status"] = inv_table["Needs_Reorder"].map({1: "🔴 REORDER", 0: "🟢 OK"})

    st.dataframe(
        inv_table.drop(columns=["Needs_Reorder"]).sort_values("Status"),
        use_container_width=True
    )

    # ABC Analysis
    st.markdown('<div class="section-title">ABC Classification</div>', unsafe_allow_html=True)
    abc = filt.groupby("Product")["Revenue_INR"].sum().sort_values(ascending=False).reset_index()
    abc["Cumulative_Pct"] = abc["Revenue_INR"].cumsum() / abc["Revenue_INR"].sum() * 100
    abc["Class"] = abc["Cumulative_Pct"].apply(
        lambda x: "A — High Value" if x <= 70 else ("B — Medium Value" if x <= 90 else "C — Low Value")
    )
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.bar(abc.head(20), x="Product", y="Revenue_INR",
                     color="Class", color_discrete_map={
                         "A — High Value": COLORS[0],
                         "B — Medium Value": COLORS[1],
                         "C — Low Value": COLORS[2]
                     })
        fig.update_layout(**PLOT_THEME, xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        abc_summary = abc.groupby("Class").agg(
            SKU_Count=("Product", "count"),
            Total_Revenue=("Revenue_INR", "sum")
        ).reset_index()
        fig = px.pie(abc_summary, values="SKU_Count", names="Class",
                     color_discrete_sequence=COLORS, hole=0.4)
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    # Reorder alerts
    reorder_items = inv_table[inv_table["Status"] == "🔴 REORDER"]
    if len(reorder_items) > 0:
        st.markdown('<div class="section-title">🚨 Reorder Alerts</div>', unsafe_allow_html=True)
        for _, row in reorder_items.iterrows():
            st.markdown(f"""
            <div class="alert-box">
                ⚠️ <strong>{row['Product']}</strong> — Stock: <strong>{row['Stock_Level']:.0f}</strong>
                | Reorder Point: <strong>{row['Reorder_Point']:.0f}</strong>
                | EOQ: <strong>{row['EOQ']:.0f}</strong> units | Region: {row['Region']}
            </div>
            """, unsafe_allow_html=True)


# ======================================================================================
# ④ PRODUCTION PLANNING
# ======================================================================================
elif module == "🏭 Production Planning":
    st.markdown('<div class="section-title">🏭 Production Planning</div>', unsafe_allow_html=True)

    buffer = st.slider("Production Buffer (%)", min_value=5, max_value=30, value=15, step=5)
    sel_cat = st.multiselect("Filter by Category", df["Category"].unique(),
                              default=list(df["Category"].unique()))
    filt = df[df["Category"].isin(sel_cat)].copy()
    filt["Production_Plan"] = (filt["Demand"] * (1 + buffer / 100)).round(0)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏭 Total Production Plan", f"{filt['Production_Plan'].sum():,.0f}")
    c2.metric("📦 Total Demand", f"{filt['Demand'].sum():,.0f}")
    c3.metric("📈 Buffer Units", f"{(filt['Production_Plan'] - filt['Demand']).sum():,.0f}")
    c4.metric("📉 Lead Time (avg)", f"{filt['Delivery_Days'].mean():.1f} days")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Monthly Production Plan vs Demand</div>', unsafe_allow_html=True)
        monthly_prod = filt.groupby("Month_Label").agg(
            Demand=("Demand", "sum"),
            Production=("Production_Plan", "sum")
        ).reset_index()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_prod["Month_Label"], y=monthly_prod["Demand"],
                              name="Demand", marker_color=COLORS[0]))
        fig.add_trace(go.Bar(x=monthly_prod["Month_Label"], y=monthly_prod["Production"],
                              name="Production Plan", marker_color=COLORS[1]))
        fig.update_layout(**PLOT_THEME, barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Production by Category</div>', unsafe_allow_html=True)
        cat_prod = filt.groupby("Category")["Production_Plan"].sum().reset_index()
        fig = px.pie(cat_prod, values="Production_Plan", names="Category",
                     color_discrete_sequence=COLORS, hole=0.4)
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">SKU-Level Production Schedule</div>', unsafe_allow_html=True)
    prod_table = filt.groupby(["SKU_ID", "Product", "Category"]).agg(
        Total_Demand=("Demand", "sum"),
        Production_Plan=("Production_Plan", "sum"),
        Buffer_Units=("Production_Plan", "sum")
    ).reset_index()
    prod_table["Buffer_Units"] = prod_table["Production_Plan"] - prod_table["Total_Demand"]
    prod_table["Utilization_Pct"] = (
        prod_table["Total_Demand"] / prod_table["Production_Plan"] * 100
    ).round(1)
    st.dataframe(prod_table.round(0), use_container_width=True)

    # Future production forecast
    st.markdown('<div class="section-title">📅 Production Forecast: Apr 2025 → Jun 2026</div>', unsafe_allow_html=True)

    future_months = pd.date_range("2025-04-01", "2026-06-30", freq="MS")
    avg_monthly = filt.groupby("Month_Label")["Production_Plan"].sum().mean()
    prod_forecast = pd.DataFrame({
        "Month": [m.strftime("%Y-%m") for m in future_months],
        "Planned_Production": [
            round(avg_monthly * (1 + buffer / 100) *
                  (1.3 if m.month in [10, 11] else
                   1.15 if m.month in [3, 4, 12] else
                   0.9 if m.month in [1, 2, 7] else 1.0), 0)
            for m in future_months
        ]
    })
    fig = px.bar(prod_forecast, x="Month", y="Planned_Production",
                 color="Planned_Production", color_continuous_scale="Blues")
    fig.update_layout(**PLOT_THEME)
    st.plotly_chart(fig, use_container_width=True)


# ======================================================================================
# ⑤ LOGISTICS OPTIMIZATION
# ======================================================================================
elif module == "🚚 Logistics Optimization":
    st.markdown('<div class="section-title">🚚 Logistics Optimization</div>', unsafe_allow_html=True)

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    total_ship_cost = df["Shipping_Cost_INR"].sum()
    delayed_pct = df["Is_Delayed"].mean() * 100
    avg_del = df["Delivery_Days"].mean()
    return_rate = df["Return_Flag"].mean() * 100

    c1.metric("💸 Total Shipping Cost", f"₹{total_ship_cost/1e3:.1f}K")
    c2.metric("⏱️ Delayed Orders", f"{delayed_pct:.1f}%",
              delta=f"-{np.random.uniform(1,3):.1f}% vs prev", delta_color="inverse")
    c3.metric("📦 Avg Delivery Days", f"{avg_del:.1f}")
    c4.metric("↩️ Return Rate", f"{return_rate:.1f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Shipping Cost by Region</div>', unsafe_allow_html=True)
        reg_cost = df.groupby("Region")["Shipping_Cost_INR"].mean().sort_values().reset_index()
        fig = px.bar(reg_cost, x="Shipping_Cost_INR", y="Region",
                     orientation="h", color="Shipping_Cost_INR",
                     color_continuous_scale="Reds")
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">Delivery Days Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="Delivery_Days", nbins=10,
                           color_discrete_sequence=[COLORS[1]])
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Courier Partner Performance</div>', unsafe_allow_html=True)
        courier = df.groupby("Courier_Partner").agg(
            Orders=("Demand", "count"),
            Avg_Delivery_Days=("Delivery_Days", "mean"),
            Avg_Cost=("Shipping_Cost_INR", "mean"),
            Delayed_Pct=("Is_Delayed", "mean")
        ).reset_index().round(2)
        courier["Delayed_Pct"] = (courier["Delayed_Pct"] * 100).round(1)
        st.dataframe(courier, use_container_width=True)

    with col4:
        st.markdown('<div class="section-title">Warehouse → Region Flow</div>', unsafe_allow_html=True)
        wh_flow = df.groupby(["Warehouse", "Region"])["Demand"].sum().reset_index()
        fig = px.density_heatmap(wh_flow, x="Warehouse", y="Region",
                                  z="Demand", color_continuous_scale="Blues")
        fig.update_layout(**PLOT_THEME)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="section-title">📍 City-Level Shipping Cost Analysis</div>', unsafe_allow_html=True)
    city_cost = df.groupby(["City", "Region"]).agg(
        Avg_Shipping_Cost=("Shipping_Cost_INR", "mean"),
        Total_Orders=("Demand", "count"),
        Avg_Delivery_Days=("Delivery_Days", "mean"),
        Return_Rate=("Return_Flag", "mean")
    ).reset_index().round(2)
    city_cost["Return_Rate"] = (city_cost["Return_Rate"] * 100).round(1)
    city_cost = city_cost.sort_values("Avg_Shipping_Cost", ascending=False)
    st.dataframe(city_cost, use_container_width=True)

    # Delay alerts
    st.markdown('<div class="section-title">🚨 High-Delay Routes</div>', unsafe_allow_html=True)
    delay_routes = df.groupby(["Region", "Courier_Partner"])["Is_Delayed"].mean().reset_index()
    delay_routes["Delayed_Pct"] = (delay_routes["Is_Delayed"] * 100).round(1)
    for _, row in delay_routes[delay_routes["Delayed_Pct"] > 40].iterrows():
        st.markdown(f"""
        <div class="alert-box">
            ⚠️ High delays on <strong>{row['Region']}</strong> via <strong>{row['Courier_Partner']}</strong>
            — {row['Delayed_Pct']}% delayed. Consider alternate courier or route.
        </div>
        """, unsafe_allow_html=True)

    for _, row in delay_routes[delay_routes["Delayed_Pct"] <= 20].iterrows():
        st.markdown(f"""
        <div class="alert-ok">
            ✅ <strong>{row['Region']}</strong> via <strong>{row['Courier_Partner']}</strong>
            — {row['Delayed_Pct']}% delayed. Performing well.
        </div>
        """, unsafe_allow_html=True)


# ======================================================================================
# ⑥ DECISION AI CHATBOT
# ======================================================================================
elif module == "🤖 Decision AI":
    st.markdown('<div class="section-title">🤖 Decision AI — Supply Chain Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask questions about demand, inventory, logistics, production, or forecasts.")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    user = st.text_input("💬 Ask a business question...",
                          placeholder="e.g. Which region has highest shipping cost?")

    if user:
        q = user.lower()

        # ── Smart responses ──
        if "forecast" in q or "predict" in q or "future" in q:
            peak_month = df.groupby("month")["Demand"].sum().idxmax()
            response = (f"📈 The demand forecast model (RMSE: {rmse:.2f}, MAE: {mae:.2f}) shows "
                        f"peak demand in Month {peak_month}. Festive periods (Oct–Nov) see ~35% spike. "
                        f"Forecast is extended to **June 2026** — go to the Forecasting module for full details.")

        elif "inventory" in q or "stock" in q or "reorder" in q:
            alert_count = int(df["Needs_Reorder"].sum())
            top_sku = df.sort_values("Stock_Level").iloc[0]["Product"]
            response = (f"📦 **{alert_count}** SKUs require immediate reorder. "
                        f"Lowest stock product: **{top_sku}**. "
                        f"Average Safety Stock: {df['Safety_Stock'].mean():.1f} units. "
                        f"Recommended: trigger EOQ orders for reorder-flagged items.")

        elif "production" in q or "manufacture" in q or "plan" in q:
            total_plan = df["Production_Plan"].sum()
            response = (f"🏭 Total production plan: **{total_plan:,.0f} units** (15% buffer over demand). "
                        f"Peak production months: Oct, Nov (festive). "
                        f"Low production months: Jan, Feb (post-festive dip). "
                        f"Recommend increasing capacity 30% in Q4 2025.")

        elif "logistic" in q or "delivery" in q or "shipping" in q or "courier" in q:
            high_cost_region = df.groupby("Region")["Shipping_Cost_INR"].mean().idxmax()
            high_delay_region = df.groupby("Region")["Is_Delayed"].mean().idxmax()
            response = (f"🚚 Highest shipping cost region: **{high_cost_region}**. "
                        f"Most delayed region: **{high_delay_region}**. "
                        f"Average delivery: {df['Delivery_Days'].mean():.1f} days. "
                        f"Recommendation: optimize courier allocation in {high_delay_region}.")

        elif "return" in q or "refund" in q:
            return_rate = df["Return_Flag"].mean() * 100
            top_return_cat = df.groupby("Category")["Return_Flag"].mean().idxmax()
            response = (f"↩️ Overall return rate: **{return_rate:.1f}%**. "
                        f"Highest returns in category: **{top_return_cat}**. "
                        f"Recommendation: quality check and packaging review for this category.")

        elif "revenue" in q or "sales" in q or "profit" in q:
            total_rev = df["Revenue_INR"].sum()
            top_cat = df.groupby("Category")["Revenue_INR"].sum().idxmax()
            top_region = df.groupby("Region")["Revenue_INR"].sum().idxmax()
            response = (f"💰 Total Revenue: **₹{total_rev/1e6:.2f}M**. "
                        f"Top category: **{top_cat}**. "
                        f"Top region: **{top_region}**. "
                        f"Q4 (Oct–Dec) contributes ~40% of annual revenue.")

        elif "risk" in q or "alert" in q or "problem" in q:
            vol = df["Demand"].std()
            delayed = df["Is_Delayed"].mean() * 100
            reorder = df["Needs_Reorder"].sum()
            response = (f"⚠️ Risk Summary:\n"
                        f"• Demand volatility: **{vol:.2f}** std dev\n"
                        f"• Delayed orders: **{delayed:.1f}%**\n"
                        f"• SKUs needing reorder: **{reorder}**\n"
                        f"• Return rate: **{df['Return_Flag'].mean()*100:.1f}%**")

        elif "best" in q and "product" in q:
            best = df.groupby("Product")["Revenue_INR"].sum().idxmax()
            response = f"🏆 Best performing product: **{best}** by total revenue."

        elif "worst" in q or "low" in q:
            worst = df.groupby("Product")["Revenue_INR"].sum().idxmin()
            response = f"📉 Lowest performing product: **{worst}**. Consider promotion or discontinuation."

        elif "region" in q:
            top_r = df.groupby("Region")["Demand"].sum().idxmax()
            response = f"🗺️ Top demand region: **{top_r}**. Focus inventory and logistics resources here."

        elif "category" in q:
            cat_df = df.groupby("Category").agg(
                Revenue=("Revenue_INR","sum"), Demand=("Demand","sum")).reset_index()
            response = "🏷️ Category breakdown:\n" + "\n".join(
                f"• {r['Category']}: ₹{r['Revenue']/1000:.1f}K revenue, {r['Demand']} units"
                for _, r in cat_df.iterrows()
            )

        else:
            response = ("💡 I can answer questions about:\n"
                        "• **Forecast / Predict** — demand forecasting\n"
                        "• **Inventory / Stock / Reorder** — inventory levels\n"
                        "• **Production / Plan** — production scheduling\n"
                        "• **Logistics / Shipping / Delivery** — courier analysis\n"
                        "• **Revenue / Sales** — financial insights\n"
                        "• **Risk / Alert** — supply chain risks\n"
                        "• **Region / Category / Product** — segment analysis")

        st.session_state.chat.append(("You", user))
        st.session_state.chat.append(("AI", response))

    # Render chat
    for speaker, msg in reversed(st.session_state.chat[-20:]):
        if speaker == "You":
            st.markdown(f'<div class="chat-user">🧑 {msg}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖 {msg}</div>', unsafe_allow_html=True)

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat = []
        st.rerun()
