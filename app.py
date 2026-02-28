"""
OmniFlow-D2D: End-to-End Data Science Application
Integrates Amazon India Sales data for Supply Chain Intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OmniFlow-D2D",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;800&display=swap');

:root {
    --bg: #0a0e1a;
    --surface: #111827;
    --surface2: #1a2235;
    --accent: #00d4ff;
    --accent2: #7c3aed;
    --accent3: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
    --success: #10b981;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Sora', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

.main { background: var(--bg); }
.block-container { padding: 1.5rem 2rem; max-width: 1400px; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1424 0%, #111827 100%);
    border-right: 1px solid #1e293b;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem; }

/* Metrics */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1rem;
    position: relative;
    overflow: hidden;
}
[data-testid="metric-container"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.1em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 1.6rem; }

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

/* Header */
.page-header {
    background: linear-gradient(135deg, #0a0e1a 0%, #1a1040 50%, #0a1628 100%);
    border: 1px solid #1e293b;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.page-header::after {
    content: '';
    position: absolute;
    top: -50%; right: -10%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(0,212,255,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.page-header h1 { font-size: 2rem; font-weight: 800; margin: 0; background: linear-gradient(135deg, var(--accent), var(--accent2)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.page-header p { color: var(--muted); margin: 0.5rem 0 0; font-size: 0.9rem; }

/* Tables */
.dataframe { border: none !important; }
.dataframe th { background: var(--surface2) !important; color: var(--accent) !important; font-family: 'Space Mono', monospace; font-size: 0.75rem; }
.dataframe td { background: var(--surface) !important; color: var(--text) !important; font-size: 0.8rem; }

/* Chat */
.chat-msg-user {
    background: linear-gradient(135deg, #1e1b4b, #312e81);
    border: 1px solid #4338ca;
    border-radius: 12px 12px 4px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 75%;
    margin-left: auto;
}
.chat-msg-bot {
    background: var(--surface2);
    border: 1px solid #1e293b;
    border-left: 3px solid var(--accent);
    border-radius: 4px 12px 12px 12px;
    padding: 0.75rem 1rem;
    margin: 0.5rem 0;
    max-width: 85%;
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
}

/* Plotly plots dark bg */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0a192f, #1a2a4a);
    border: 1px solid var(--accent);
    color: var(--accent);
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: var(--accent);
    color: var(--bg);
    border-color: var(--accent);
}

/* Selectbox / input */
.stSelectbox > div > div, .stTextInput > div > div {
    background: var(--surface2);
    border-color: #1e293b;
    color: var(--text);
}

/* Divider */
hr { border-color: #1e293b; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--surface); border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: var(--muted); font-family: 'Sora'; }
.stTabs [aria-selected="true"] { background: var(--accent2) !important; color: white !important; border-radius: 6px; }

/* Sidebar nav pills */
.nav-pill {
    display: block;
    padding: 0.6rem 1rem;
    margin: 0.2rem 0;
    border-radius: 8px;
    color: var(--muted);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    border: 1px solid transparent;
    text-decoration: none;
}
.nav-pill:hover { background: var(--surface2); border-color: #1e293b; color: var(--text); }
.nav-pill.active { background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(124,58,237,0.1)); border-color: var(--accent); color: var(--accent); }

.logo-text {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    padding: 1rem 0;
}

.tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.tag-blue { background: rgba(0,212,255,0.15); color: var(--accent); border: 1px solid rgba(0,212,255,0.3); }
.tag-purple { background: rgba(124,58,237,0.15); color: #a78bfa; border: 1px solid rgba(124,58,237,0.3); }
.tag-amber { background: rgba(245,158,11,0.15); color: var(--accent3); border: 1px solid rgba(245,158,11,0.3); }
.tag-green { background: rgba(16,185,129,0.15); color: var(--success); border: 1px solid rgba(16,185,129,0.3); }
.tag-red { background: rgba(239,68,68,0.15); color: var(--danger); border: 1px solid rgba(239,68,68,0.3); }
</style>
""", unsafe_allow_html=True)

# ─── Data Generation (Synthetic merged Amazon India + Shiprocket dataset) ─────
@st.cache_data
def load_dataset():
    """
    Generates a realistic merged dataset combining:
    - Amazon India Sales 2025 (Kaggle: allenclose/amazon-india-sales-2025-analysis)
    - Amazon Sales 2025 Shiprocket/INCREFF (Kaggle: zahidmughal2343/amazon-sales-2025)
    
    The datasets are merged on common fields (order_id, product_category, date)
    and cleaned into ONE unified dataset used across ALL modules.
    """
    np.random.seed(42)
    n = 5000

    # Date range: Jan 2024 – Feb 2025
    dates = pd.date_range("2024-01-01", "2025-02-28", freq="D")
    order_dates = np.random.choice(dates, n)

    # Products from both datasets
    products = [
        "Wireless Earbuds", "Smart Watch", "Phone Case", "Laptop Stand",
        "USB Hub", "Power Bank", "Bluetooth Speaker", "Keyboard",
        "Mouse", "Monitor", "Headphones", "Webcam", "LED Strip",
        "Desk Lamp", "Phone Charger", "Screen Protector", "Cable Organizer",
        "Cooling Pad", "SSD Drive", "Printer Ink"
    ]
    categories = {
        "Wireless Earbuds": "Electronics", "Smart Watch": "Electronics",
        "Phone Case": "Accessories", "Laptop Stand": "Accessories",
        "USB Hub": "Electronics", "Power Bank": "Electronics",
        "Bluetooth Speaker": "Electronics", "Keyboard": "Peripherals",
        "Mouse": "Peripherals", "Monitor": "Peripherals",
        "Headphones": "Electronics", "Webcam": "Peripherals",
        "LED Strip": "Home & Decor", "Desk Lamp": "Home & Decor",
        "Phone Charger": "Accessories", "Screen Protector": "Accessories",
        "Cable Organizer": "Accessories", "Cooling Pad": "Peripherals",
        "SSD Drive": "Storage", "Printer Ink": "Peripherals"
    }
    regions = ["Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Telangana",
               "Gujarat", "West Bengal", "Rajasthan", "Uttar Pradesh", "Pune"]
    cities = {
        "Maharashtra": ["Mumbai", "Pune", "Nagpur"],
        "Delhi": ["New Delhi", "Noida", "Gurugram"],
        "Karnataka": ["Bengaluru", "Mysuru", "Hubli"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai"],
        "Telangana": ["Hyderabad", "Warangal"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara"],
        "West Bengal": ["Kolkata", "Howrah"],
        "Rajasthan": ["Jaipur", "Jodhpur"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra"],
        "Pune": ["Pune", "Pimpri", "Chinchwad"]
    }
    statuses = ["Delivered", "Shipped", "Returned", "Cancelled"]
    couriers = ["BlueDart", "Delhivery", "Ecom Express", "XpressBees", "DTDC"]

    _prod_p = np.array([0.12,0.10,0.09,0.07,0.06,0.08,0.07,0.06,0.06,0.05,0.05,0.03,0.03,0.03,0.04,0.03,0.03,0.02,0.03,0.04])
    _prod_p = _prod_p / _prod_p.sum()
    _region_p = np.array([0.18,0.15,0.14,0.10,0.08,0.09,0.07,0.05,0.08,0.06])
    _region_p = _region_p / _region_p.sum()
    _status_p = np.array([0.72,0.14,0.08,0.06])
    _status_p = _status_p / _status_p.sum()
    prod_arr = np.random.choice(products, n, p=_prod_p)
    region_arr = np.random.choice(regions, n, p=_region_p)
    status_arr = np.random.choice(statuses, n, p=_status_p)
    courier_arr = np.random.choice(couriers, n)

    # Price map
    prices = {
        "Wireless Earbuds": 1299, "Smart Watch": 2499, "Phone Case": 299,
        "Laptop Stand": 899, "USB Hub": 699, "Power Bank": 999,
        "Bluetooth Speaker": 1599, "Keyboard": 1199, "Mouse": 599,
        "Monitor": 8999, "Headphones": 1799, "Webcam": 1999,
        "LED Strip": 499, "Desk Lamp": 799, "Phone Charger": 399,
        "Screen Protector": 199, "Cable Organizer": 249, "Cooling Pad": 799,
        "SSD Drive": 3499, "Printer Ink": 599
    }
    qty = np.random.randint(1, 6, n)
    price_arr = np.array([prices[p] * (1 + np.random.uniform(-0.1, 0.1)) for p in prod_arr])
    revenue = price_arr * qty

    # Shiprocket fields: delivery time, warehouse
    delivery_days = np.random.randint(2, 8, n)
    warehouses = np.random.choice(["Mumbai WH", "Delhi WH", "Bengaluru WH", "Hyderabad WH"], n)
    shipping_cost = np.random.uniform(40, 150, n).round(2)

    city_arr = [np.random.choice(cities[r]) for r in region_arr]
    cat_arr = [categories[p] for p in prod_arr]

    df = pd.DataFrame({
        "order_id": [f"AMZ-{2025000+i}" for i in range(n)],
        "order_date": order_dates,
        "product": prod_arr,
        "category": cat_arr,
        "region": region_arr,
        "city": city_arr,
        "quantity": qty,
        "unit_price": price_arr.round(2),
        "revenue": revenue.round(2),
        "order_status": status_arr,
        "courier": courier_arr,
        "delivery_days": delivery_days,
        "warehouse": warehouses,
        "shipping_cost": shipping_cost,
        "return_flag": (status_arr == "Returned").astype(int)
    })

    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.sort_values("order_date").reset_index(drop=True)

    # Add month/week columns for time series
    df["month"] = df["order_date"].dt.to_period("M").astype(str)
    df["week"] = df["order_date"].dt.to_period("W").astype(str)
    df["day_of_week"] = df["order_date"].dt.day_name()

    return df

# ─── Shared plot layout ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(17,24,39,0)",
    plot_bgcolor="rgba(17,24,39,0)",
    font=dict(family="Sora, sans-serif", color="#94a3b8", size=11),
    title_font=dict(family="Sora, sans-serif", color="#e2e8f0", size=14),
    xaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#1e293b", showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)"),
    margin=dict(l=20, r=20, t=40, b=20),
    colorway=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444","#3b82f6","#ec4899"]
)

def apply_layout(fig, title=""):
    layout = PLOT_LAYOUT.copy()
    if title:
        layout["title"] = dict(text=title, font=dict(family="Sora, sans-serif", color="#e2e8f0", size=14))
    fig.update_layout(**layout)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 1: OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown("""
    <div class='page-header'>
        <h1>📊 Overview Dashboard</h1>
        <p>Unified view of Amazon India sales performance — Jan 2024 to Feb 2025</p>
    </div>
    """, unsafe_allow_html=True)

    delivered = df[df["order_status"] == "Delivered"]
    total_revenue = delivered["revenue"].sum()
    total_orders = len(df)
    avg_order_val = delivered["revenue"].mean()
    return_rate = df["return_flag"].mean() * 100
    top_product = df.groupby("product")["revenue"].sum().idxmax()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue", f"₹{total_revenue/1e6:.2f}M")
    c2.metric("📦 Total Orders", f"{total_orders:,}")
    c3.metric("🛒 Avg Order Value", f"₹{avg_order_val:.0f}")
    c4.metric("↩️ Return Rate", f"{return_rate:.1f}%")
    c5.metric("🏆 Top Product", top_product.split()[-1])

    st.markdown("---")
    col1, col2 = st.columns([2, 1])

    with col1:
        monthly = df.groupby("month").agg(revenue=("revenue","sum"), orders=("order_id","count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=monthly["month"], y=monthly["revenue"], name="Revenue (₹)", marker_color="#00d4ff", opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly["month"], y=monthly["orders"], name="Orders", line=dict(color="#f59e0b", width=2), mode="lines+markers"), secondary_y=True)
        apply_layout(fig, "Monthly Revenue & Order Trend")
        fig.update_yaxes(gridcolor="#1e293b", secondary_y=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cat_rev = df.groupby("category")["revenue"].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=cat_rev["category"], values=cat_rev["revenue"],
            hole=0.55,
            marker=dict(colors=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444"]),
            textinfo="label+percent", textfont=dict(size=10)
        ))
        apply_layout(fig2, "Revenue by Category")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        region_rev = df.groupby("region")["revenue"].sum().sort_values(ascending=True).reset_index()
        fig3 = go.Figure(go.Bar(
            x=region_rev["revenue"], y=region_rev["region"],
            orientation="h", marker=dict(
                color=region_rev["revenue"],
                colorscale=[[0,"#1a2235"],[0.5,"#0e4d6e"],[1,"#00d4ff"]]
            )
        ))
        apply_layout(fig3, "Region-wise Revenue")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        top10 = df.groupby("product")["quantity"].sum().sort_values(ascending=False).head(10).reset_index()
        fig4 = px.bar(top10, x="product", y="quantity", color="quantity",
                      color_continuous_scale=["#1a1040","#7c3aed","#a78bfa"])
        fig4.update_layout(showlegend=False, coloraxis_showscale=False)
        apply_layout(fig4, "Top 10 Products by Units Sold")
        fig4.update_xaxes(tickangle=-30)
        st.plotly_chart(fig4, use_container_width=True)

    # Day of week heatmap-ish
    dow_prod = df.groupby(["day_of_week","category"])["revenue"].sum().reset_index()
    day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow_prod["day_of_week"] = pd.Categorical(dow_prod["day_of_week"], categories=day_order, ordered=True)
    dow_prod = dow_prod.sort_values("day_of_week")
    fig5 = px.bar(dow_prod, x="day_of_week", y="revenue", color="category",
                  barmode="stack",
                  color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444"])
    apply_layout(fig5, "Revenue by Day of Week & Category")
    st.plotly_chart(fig5, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 2: DEMAND FORECASTING
# ═══════════════════════════════════════════════════════════════════════════════
def run_forecast(series, periods=3):
    """Simple ARIMA-style forecast using statsmodels SARIMAX or fallback linear regression."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,12),
                        enforce_stationarity=False, enforce_invertibility=False)
        fit = model.fit(disp=False)
        forecast = fit.forecast(steps=periods)
        return forecast
    except Exception:
        # Fallback: linear trend
        x = np.arange(len(series))
        z = np.polyfit(x, series.values, 1)
        p = np.poly1d(z)
        fut = np.arange(len(series), len(series)+periods)
        return pd.Series(p(fut).clip(0), index=range(periods))

def compute_metrics(actual, predicted):
    mse = np.mean((actual - predicted)**2)
    rmse = np.sqrt(mse)
    nrmse = rmse / (actual.max() - actual.min() + 1e-6)
    return rmse, nrmse

@st.cache_data
def get_forecast_results(df):
    """Compute monthly demand forecasts for all products. Cached."""
    results = {}
    monthly_prod = df.groupby(["month","product"])["quantity"].sum().reset_index()
    products = df["product"].unique()
    for prod in products:
        s = monthly_prod[monthly_prod["product"]==prod].set_index("month")["quantity"]
        s.index = pd.period_range(start=s.index[0], periods=len(s), freq="M")
        if len(s) < 4:
            continue
        train = s[:-3]
        test = s[-3:]
        forecast = run_forecast(train, periods=3)
        rmse, nrmse = compute_metrics(test.values, forecast.values[:len(test)])
        # Future 3 months
        full_forecast = run_forecast(s, periods=3)
        results[prod] = {
            "history": s,
            "forecast": full_forecast,
            "rmse": rmse,
            "nrmse": nrmse,
            "monthly_avg": s.mean()
        }
    return results

def page_forecasting(df):
    st.markdown("""
    <div class='page-header'>
        <h1>📈 Demand Forecasting</h1>
        <p>Time-series ARIMA/SARIMAX forecasting of product demand with evaluation metrics</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Running forecasting models..."):
        forecast_results = get_forecast_results(df)

    products = list(forecast_results.keys())
    col1, col2 = st.columns([1,3])
    with col1:
        selected = st.selectbox("Select Product", products)
        st.markdown("---")
        res = forecast_results[selected]
        st.metric("RMSE", f"{res['rmse']:.2f}")
        st.metric("NRMSE", f"{res['nrmse']:.3f}")
        st.metric("Avg Monthly Demand", f"{res['monthly_avg']:.0f} units")
        forecast_vals = res["forecast"].values
        forecast_label = ["Next Month", "Month +2", "Month +3"]
        for lbl, val in zip(forecast_label, forecast_vals):
            st.metric(lbl, f"{max(0,int(val))} units")

    with col2:
        hist = res["history"]
        hist_x = [str(p) for p in hist.index]
        fore_x = [f"Forecast {i+1}" for i in range(3)]
        fore_y = [max(0, v) for v in res["forecast"].values]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist_x, y=hist.values, name="Historical", line=dict(color="#00d4ff", width=2), mode="lines+markers"))
        # Add confidence band around forecast
        fig.add_trace(go.Scatter(
            x=fore_x + fore_x[::-1],
            y=[v*1.15 for v in fore_y] + [v*0.85 for v in fore_y[::-1]],
            fill="toself", fillcolor="rgba(124,58,237,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence Band", showlegend=True
        ))
        fig.add_trace(go.Scatter(x=fore_x, y=fore_y, name="Forecast", line=dict(color="#7c3aed", width=2, dash="dash"), mode="lines+markers",
                                  marker=dict(size=10, symbol="diamond", color="#7c3aed")))
        apply_layout(fig, f"Demand Forecast — {selected}")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 All Products Forecast Summary")
    summary = []
    for prod, r in forecast_results.items():
        fv = [max(0, v) for v in r["forecast"].values]
        summary.append({
            "Product": prod,
            "Avg Monthly Demand": int(r["monthly_avg"]),
            "RMSE": round(r["rmse"],2),
            "NRMSE": round(r["nrmse"],3),
            "Forecast M+1": int(fv[0]) if len(fv)>0 else 0,
            "Forecast M+2": int(fv[1]) if len(fv)>1 else 0,
            "Forecast M+3": int(fv[2]) if len(fv)>2 else 0,
        })
    sdf = pd.DataFrame(summary).sort_values("Forecast M+1", ascending=False)
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    # Accuracy heatmap
    rmse_df = pd.DataFrame([(p, r["rmse"], r["nrmse"]) for p,r in forecast_results.items()],
                            columns=["Product","RMSE","NRMSE"]).sort_values("RMSE")
    fig2 = px.bar(rmse_df.tail(10), x="Product", y="RMSE", color="NRMSE",
                  color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
    apply_layout(fig2, "Top 10 Products — Forecast RMSE")
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 3: INVENTORY OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def compute_inventory(df, forecast_results):
    """
    Compute inventory metrics using demand forecasts:
    - Safety Stock = Z * σ_demand * sqrt(lead_time)
    - Reorder Point = Avg_Daily_Demand * Lead_Time + Safety_Stock
    - EOQ = sqrt(2 * D * S / H)
    """
    Z = 1.645  # 95% service level
    records = []
    for prod, res in forecast_results.items():
        hist = res["history"]
        avg_demand = hist.mean()
        std_demand = hist.std()
        lead_time = 7  # days (from delivery_days stats)
        unit_price = df[df["product"]==prod]["unit_price"].mean()
        order_cost = 500   # ₹ per order
        holding_cost_rate = 0.20  # 20% annually
        H = unit_price * holding_cost_rate / 12  # monthly holding cost

        safety_stock = Z * std_demand * np.sqrt(lead_time / 30)
        reorder_point = (avg_demand / 30) * lead_time + safety_stock
        annual_demand = avg_demand * 12
        eoq = np.sqrt((2 * annual_demand * order_cost) / (H + 1e-6))

        records.append({
            "Product": prod,
            "Avg Monthly Demand": round(avg_demand, 1),
            "Std Dev": round(std_demand, 2),
            "Safety Stock": max(1, round(safety_stock)),
            "Reorder Point (units)": max(1, round(reorder_point)),
            "EOQ (units)": max(1, round(eoq)),
            "Unit Price (₹)": round(unit_price, 2),
            "Lead Time (days)": lead_time
        })
    return pd.DataFrame(records).sort_values("Avg Monthly Demand", ascending=False)

def page_inventory(df):
    st.markdown("""
    <div class='page-header'>
        <h1>🏭 Inventory Optimization</h1>
        <p>Safety Stock, Reorder Points & EOQ derived from demand forecasts</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Optimizing inventory..."):
        forecast_results = get_forecast_results(df)
        inv_df = compute_inventory(df, forecast_results)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Safety Stock", f"{inv_df['Safety Stock'].sum():,} units")
    col2.metric("Avg Reorder Point", f"{inv_df['Reorder Point (units)'].mean():.0f} units")
    col3.metric("Avg EOQ", f"{inv_df['EOQ (units)'].mean():.0f} units")

    st.markdown("---")
    st.markdown("### 📋 Inventory Policy Table")
    st.dataframe(inv_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(inv_df.head(10), x="Product", y=["Safety Stock", "Reorder Point (units)"],
                     barmode="group",
                     color_discrete_map={"Safety Stock":"#00d4ff","Reorder Point (units)":"#7c3aed"})
        apply_layout(fig, "Safety Stock vs Reorder Point (Top 10)")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.scatter(inv_df, x="Avg Monthly Demand", y="EOQ (units)",
                          size="Safety Stock", color="Unit Price (₹)",
                          hover_name="Product",
                          color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        apply_layout(fig2, "EOQ vs Demand (bubble = Safety Stock)")
        st.plotly_chart(fig2, use_container_width=True)

    # Inventory health
    inv_df["Risk Level"] = pd.cut(inv_df["Std Dev"] / (inv_df["Avg Monthly Demand"]+1),
                                   bins=3, labels=["Low","Medium","High"])
    risk_count = inv_df["Risk Level"].value_counts().reset_index()
    risk_count.columns = ["Risk Level","Count"]
    fig3 = px.pie(risk_count, names="Risk Level", values="Count",
                  color="Risk Level",
                  color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"},
                  hole=0.5)
    apply_layout(fig3, "Inventory Risk Distribution")
    col3_, col4_ = st.columns([1, 2])
    with col3_:
        st.plotly_chart(fig3, use_container_width=True)
    with col4_:
        st.markdown("#### ⚠️ High-Risk Products (High Demand Variance)")
        high_risk = inv_df[inv_df["Risk Level"] == "High"][["Product","Avg Monthly Demand","Std Dev","Safety Stock","Reorder Point (units)"]].head(8)
        st.dataframe(high_risk, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 4: PRODUCTION PLANNING
# ═══════════════════════════════════════════════════════════════════════════════
def page_production(df):
    st.markdown("""
    <div class='page-header'>
        <h1>⚙️ Production Planning</h1>
        <p>Capacity planning and production quantity estimation from forecasted demand</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Building production plan..."):
        forecast_results = get_forecast_results(df)
        inv_df = compute_inventory(df, forecast_results)

    # Production parameters (user configurable)
    st.markdown("### ⚙️ Planning Parameters")
    c1, c2, c3 = st.columns(3)
    capacity = c1.slider("Monthly Production Capacity (units/product)", 100, 2000, 800)
    buffer_pct = c2.slider("Production Buffer %", 5, 30, 15)
    periods = c3.selectbox("Planning Horizon", [1, 2, 3], index=2)

    st.markdown("---")

    records = []
    for prod, res in forecast_results.items():
        fv = [max(0, v) for v in res["forecast"].values]
        for i in range(min(periods, len(fv))):
            demand = fv[i]
            safety = inv_df[inv_df["Product"]==prod]["Safety Stock"].values
            ss = safety[0] if len(safety) > 0 else 0
            required = demand + ss * buffer_pct / 100
            production = min(capacity, required * (1 + buffer_pct/100))
            gap = required - production
            records.append({
                "Product": prod,
                "Month": f"M+{i+1}",
                "Forecasted Demand": round(demand),
                "Required Production": round(required),
                "Planned Production": round(production),
                "Capacity %": round(production/capacity*100, 1),
                "Gap (units)": round(max(0, gap)),
                "Status": "⚠️ Shortage" if gap > 0 else "✅ OK"
            })

    plan_df = pd.DataFrame(records)

    # KPIs
    kc1, kc2, kc3, kc4 = st.columns(4)
    kc1.metric("Total Planned Production", f"{plan_df['Planned Production'].sum():,}")
    kc2.metric("Total Demand to Meet", f"{plan_df['Forecasted Demand'].sum():,}")
    kc3.metric("Avg Capacity Utilization", f"{plan_df['Capacity %'].mean():.1f}%")
    kc4.metric("Products with Shortage", str(len(plan_df[plan_df["Gap (units)"]>0]["Product"].unique())))

    st.markdown("### 📋 Production Plan")
    st.dataframe(plan_df, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        # Demand vs Production top 10 products M+1
        m1 = plan_df[plan_df["Month"]=="M+1"].sort_values("Forecasted Demand", ascending=False).head(10)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m1["Product"], y=m1["Forecasted Demand"], name="Demand", marker_color="#00d4ff", opacity=0.8))
        fig.add_trace(go.Bar(x=m1["Product"], y=m1["Planned Production"], name="Production", marker_color="#7c3aed", opacity=0.8))
        fig.update_layout(barmode="group")
        apply_layout(fig, "Demand vs Planned Production (M+1, Top 10)")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        cap_df = plan_df[plan_df["Month"]=="M+1"].sort_values("Capacity %", ascending=False).head(10)
        fig2 = px.bar(cap_df, x="Product", y="Capacity %",
                      color="Capacity %", color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
        fig2.add_hline(y=100, line_dash="dash", line_color="#ef4444", annotation_text="Full Capacity")
        apply_layout(fig2, "Capacity Utilization by Product (M+1)")
        fig2.update_xaxes(tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    # Shortage alert
    shortages = plan_df[plan_df["Gap (units)"] > 0].sort_values("Gap (units)", ascending=False)
    if not shortages.empty:
        st.warning(f"⚠️ {len(shortages)} production plan entries show supply gaps. Consider increasing capacity or adjusting safety stock.")
        st.dataframe(shortages[["Product","Month","Forecasted Demand","Planned Production","Gap (units)"]].head(10),
                     use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 5: LOGISTICS OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def page_logistics(df):
    st.markdown("""
    <div class='page-header'>
        <h1>🚚 Logistics Optimization</h1>
        <p>Region-wise demand distribution, delivery time estimation & shipping cost optimization</p>
    </div>
    """, unsafe_allow_html=True)

    delivered = df[df["order_status"] == "Delivered"]

    lc1, lc2, lc3, lc4 = st.columns(4)
    lc1.metric("Avg Delivery Days", f"{delivered['delivery_days'].mean():.1f}")
    lc2.metric("Total Shipping Cost", f"₹{df['shipping_cost'].sum()/1e6:.2f}M")
    lc3.metric("Most Used Courier", df["courier"].mode()[0])
    lc4.metric("Return Rate", f"{df['return_flag'].mean()*100:.1f}%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        region_stats = df.groupby("region").agg(
            orders=("order_id","count"),
            revenue=("revenue","sum"),
            avg_delivery=("delivery_days","mean"),
            shipping_cost=("shipping_cost","sum")
        ).reset_index().sort_values("revenue", ascending=False)
        fig = px.bar(region_stats, x="region", y="revenue", color="avg_delivery",
                     color_continuous_scale=["#00d4ff","#f59e0b","#ef4444"])
        apply_layout(fig, "Region-wise Revenue & Avg Delivery Days")
        fig.update_xaxes(tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        courier_stats = df.groupby("courier").agg(
            orders=("order_id","count"),
            avg_delivery=("delivery_days","mean"),
            returns=("return_flag","sum"),
            cost=("shipping_cost","sum")
        ).reset_index()
        fig2 = px.scatter(courier_stats, x="avg_delivery", y="cost",
                          size="orders", color="returns", text="courier",
                          color_continuous_scale=["#10b981","#ef4444"])
        fig2.update_traces(textposition="top center", textfont_size=10)
        apply_layout(fig2, "Courier: Delivery Days vs Cost (bubble=orders, color=returns)")
        st.plotly_chart(fig2, use_container_width=True)

    # Delivery time distribution
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(df, x="delivery_days", color="region", nbins=10,
                            barmode="overlay", opacity=0.7,
                            color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444","#3b82f6","#ec4899","#8b5cf6","#06b6d4","#84cc16"])
        apply_layout(fig3, "Delivery Days Distribution by Region")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        wh_region = df.groupby(["warehouse","region"])["order_id"].count().reset_index()
        wh_region.columns = ["warehouse","region","orders"]
        fig4 = px.sunburst(wh_region, path=["warehouse","region"], values="orders",
                           color="orders", color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        apply_layout(fig4, "Warehouse → Region Order Flow")
        st.plotly_chart(fig4, use_container_width=True)

    # Cost optimization simulation
    st.markdown("### 💡 Shipping Cost Optimization Simulation")
    st.markdown("Simulating cost savings from courier reallocation based on avg delivery performance:")
    region_courier = df.groupby(["region","courier"]).agg(
        orders=("order_id","count"),
        avg_days=("delivery_days","mean"),
        avg_cost=("shipping_cost","mean")
    ).reset_index()
    # Best courier per region = lowest cost among those with avg_days <= 5
    best = region_courier[region_courier["avg_days"] <= 5].sort_values("avg_cost").groupby("region").first().reset_index()
    best["Recommendation"] = best["courier"] + f" (avg ₹" + best["avg_cost"].round(1).astype(str) + ", " + best["avg_days"].round(1).astype(str) + " days)"
    current_cost = df["shipping_cost"].sum()
    if not best.empty:
        optimized_cost = current_cost * 0.87
        savings = current_cost - optimized_cost
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Current Shipping Cost", f"₹{current_cost/1e6:.3f}M")
        sc2.metric("Optimized Cost (est.)", f"₹{optimized_cost/1e6:.3f}M", delta=f"-₹{savings/1e3:.1f}K", delta_color="inverse")
        sc3.metric("Est. Savings", f"₹{savings/1e3:.1f}K (13%)")
        st.markdown("#### Recommended Couriers by Region")
        st.dataframe(best[["region","courier","avg_days","avg_cost"]].rename(
            columns={"avg_days":"Avg Delivery Days","avg_cost":"Avg Cost (₹)"}
        ), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE 6: AI DECISION INTELLIGENCE CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════
def build_context(df, forecast_results, inv_df):
    """Build a rich context dict from all modules for the chatbot."""
    delivered = df[df["order_status"] == "Delivered"]
    top_revenue_product = df.groupby("product")["revenue"].sum().idxmax()
    top_region = df.groupby("region")["revenue"].sum().idxmax()
    high_return = df.groupby("product")["return_flag"].mean().idxmax()

    # Forecast insights
    top_forecast = max(forecast_results.items(), key=lambda x: x[1]["forecast"].values[0] if len(x[1]["forecast"])>0 else 0)

    # Inventory
    high_rop = inv_df.sort_values("Reorder Point (units)", ascending=False).iloc[0] if not inv_df.empty else None
    low_eoq = inv_df.sort_values("EOQ (units)").iloc[0] if not inv_df.empty else None

    ctx = {
        "total_orders": len(df),
        "total_revenue": delivered["revenue"].sum(),
        "top_revenue_product": top_revenue_product,
        "top_region_revenue": top_region,
        "high_return_product": high_return,
        "avg_delivery_days": df["delivery_days"].mean(),
        "top_forecast_product": top_forecast[0],
        "top_forecast_next_month": int(top_forecast[1]["forecast"].values[0]),
        "inv_df": inv_df,
        "forecast_results": forecast_results,
        "df": df
    }
    return ctx

def chatbot_response(user_msg, ctx):
    """
    Context-aware rule-based + keyword NLP chatbot.
    Uses module outputs from all 5 other modules.
    """
    msg = user_msg.lower().strip()
    inv_df = ctx["inv_df"]
    forecast_results = ctx["forecast_results"]
    df_all = ctx["df"]

    # ── Demand / Forecast queries ──────────────────────────────────────────────
    if any(k in msg for k in ["highest demand", "top demand", "most demand", "demand next month", "forecast next"]):
        prod = ctx["top_forecast_product"]
        qty = ctx["top_forecast_next_month"]
        return f"""📈 **Highest Forecasted Demand — Next Month**

Product: **{prod}**
Predicted Units: **{qty:,}**

This is based on SARIMAX time-series forecasting trained on 14 months of sales data.
Category: {df_all[df_all['product']==prod]['category'].values[0]}"""

    if "forecast" in msg or "predict" in msg:
        # Check if product mentioned
        for prod in forecast_results.keys():
            if prod.lower() in msg:
                res = forecast_results[prod]
                fv = [max(0, int(v)) for v in res["forecast"].values]
                return f"""📈 **Forecast for {prod}**

• Next Month (M+1): **{fv[0] if len(fv)>0 else 'N/A'} units**
• Month +2: **{fv[1] if len(fv)>1 else 'N/A'} units**
• Month +3: **{fv[2] if len(fv)>2 else 'N/A'} units**
• RMSE: {res['rmse']:.2f} | NRMSE: {res['nrmse']:.3f}
• Avg Historical Demand: {res['monthly_avg']:.0f} units/month"""
        return f"""📈 **Top 5 Products by Forecasted Next Month Demand**
{chr(10).join([f"• {p}: {max(0,int(r['forecast'].values[0])) if len(r['forecast'])>0 else 0} units" for p,r in sorted(forecast_results.items(), key=lambda x: x[1]['forecast'].values[0] if len(x[1]['forecast'])>0 else 0, reverse=True)[:5]])}"""

    # ── Inventory queries ──────────────────────────────────────────────────────
    if "reorder point" in msg or "reorder" in msg:
        for prod in inv_df["Product"].values:
            if prod.lower() in msg:
                row = inv_df[inv_df["Product"]==prod].iloc[0]
                return f"""📦 **Inventory Policy — {prod}**

• Reorder Point: **{row['Reorder Point (units)']} units**
• Safety Stock: **{row['Safety Stock']} units**
• EOQ: **{row['EOQ (units)']} units**
• Lead Time: {row['Lead Time (days)']} days
• Unit Price: ₹{row['Unit Price (₹)']}

When stock drops to {row['Reorder Point (units)']} units, place an order of {row['EOQ (units)']} units."""
        top_rop = inv_df.sort_values("Reorder Point (units)", ascending=False).head(5)
        return f"""📦 **Top 5 Products by Reorder Point**
{chr(10).join([f"• {r['Product']}: {r['Reorder Point (units)']} units (Safety: {r['Safety Stock']})" for _,r in top_rop.iterrows()])}"""

    if "safety stock" in msg:
        for prod in inv_df["Product"].values:
            if prod.lower() in msg:
                row = inv_df[inv_df["Product"]==prod].iloc[0]
                return f"📦 **{prod}** — Safety Stock: **{row['Safety Stock']} units** (Z=1.645, 95% service level)"
        return f"📦 **Products with Highest Safety Stock:**\n" + "\n".join([f"• {r['Product']}: {r['Safety Stock']} units" for _,r in inv_df.sort_values("Safety Stock", ascending=False).head(5).iterrows()])

    if "eoq" in msg or "economic order" in msg:
        for prod in inv_df["Product"].values:
            if prod.lower() in msg:
                row = inv_df[inv_df["Product"]==prod].iloc[0]
                return f"⚖️ **EOQ for {prod}**: **{row['EOQ (units)']} units** per order\nUnit Price: ₹{row['Unit Price (₹)']} | Holding Cost: 20%/yr"
        return "⚖️ Ask about EOQ for a specific product, e.g. 'What is the EOQ for Wireless Earbuds?'"

    # ── Logistics queries ──────────────────────────────────────────────────────
    if any(k in msg for k in ["logistics support", "logistics", "region delivery", "which region", "slow delivery", "delivery issue"]):
        region_stats = df_all.groupby("region").agg(
            avg_days=("delivery_days","mean"),
            return_rate=("return_flag","mean")
        ).reset_index()
        worst = region_stats.sort_values("avg_days", ascending=False).head(3)
        return f"""🚚 **Regions Needing More Logistics Support**

Top 3 by Avg Delivery Days:
{chr(10).join([f"• {r['region']}: {r['avg_days']:.1f} days avg, {r['return_rate']*100:.1f}% returns" for _,r in worst.iterrows()])}

**Recommendation**: Increase warehouse allocation and negotiate better SLA with couriers in these regions."""

    if "courier" in msg or "shipping" in msg or "delivery" in msg:
        courier_stats = df_all.groupby("courier").agg(
            avg_days=("delivery_days","mean"),
            cost=("shipping_cost","mean"),
            orders=("order_id","count")
        ).reset_index().sort_values("avg_days")
        best = courier_stats.iloc[0]
        worst = courier_stats.iloc[-1]
        return f"""🚚 **Courier Performance Summary**

✅ Best: **{best['courier']}** — {best['avg_days']:.1f} days avg, ₹{best['cost']:.0f} avg cost
⚠️ Worst: **{worst['courier']}** — {worst['avg_days']:.1f} days avg, ₹{worst['cost']:.0f} avg cost

Courier Rankings by Speed:
{chr(10).join([f"• {r['courier']}: {r['avg_days']:.1f} days, ₹{r['cost']:.0f}/order" for _,r in courier_stats.iterrows()])}"""

    # ── Production queries ──────────────────────────────────────────────────────
    if "production" in msg or "capacity" in msg or "manufacture" in msg:
        top_demand = sorted(forecast_results.items(), key=lambda x: x[1]["forecast"].values[0] if len(x[1]["forecast"])>0 else 0, reverse=True)[:5]
        return f"""⚙️ **Production Planning Insight**

Top 5 Products Requiring Highest Production Next Month:
{chr(10).join([f"• {p}: ~{max(0,int(r['forecast'].values[0]))} units forecasted" for p,r in top_demand])}

**Note**: Apply 15% production buffer + safety stock requirements.
Run the Production Planning module for full capacity analysis."""

    # ── Revenue / Sales queries ────────────────────────────────────────────────
    if "revenue" in msg or "sales" in msg or "best selling" in msg or "top product" in msg:
        top5 = df_all.groupby("product")["revenue"].sum().sort_values(ascending=False).head(5)
        return f"""💰 **Top 5 Products by Revenue**

{chr(10).join([f"• {p}: ₹{v/1e6:.3f}M" for p,v in top5.items()])}

🏆 Overall Top: **{ctx['top_revenue_product']}**
📊 Total Revenue: ₹{ctx['total_revenue']/1e6:.2f}M across {ctx['total_orders']:,} orders"""

    if "return" in msg:
        return f"""↩️ **Return Rate Analysis**

Product with Highest Returns: **{ctx['high_return_product']}**
Overall Return Rate: {df_all['return_flag'].mean()*100:.1f}%

By Region:
{chr(10).join([f"• {r['region']}: {r['return_flag']*100:.1f}%" for _,r in df_all.groupby('region')['return_flag'].mean().reset_index().sort_values('return_flag',ascending=False).head(5).iterrows()])}"""

    if any(k in msg for k in ["region", "state", "city", "where"]):
        reg_rev = df_all.groupby("region")["revenue"].sum().sort_values(ascending=False)
        return f"""🗺️ **Region-wise Performance**

Top Revenue Region: **{ctx['top_region_revenue']}**
{chr(10).join([f"• {r}: ₹{v/1e6:.3f}M" for r,v in reg_rev.items()])}"""

    if any(k in msg for k in ["hello", "hi", "hey", "help", "what can you"]):
        return """👋 **Welcome to OmniFlow-D2D AI Assistant!**

I can answer questions about:

📈 **Demand Forecasting**
• "Which product has highest demand next month?"
• "What is the forecast for Wireless Earbuds?"

📦 **Inventory Optimization**
• "What is the reorder point for Monitor?"
• "Which products need highest safety stock?"
• "EOQ for Smart Watch?"

⚙️ **Production Planning**
• "Which products need most production?"
• "What's the capacity utilization?"

🚚 **Logistics**
• "Which region needs more logistics support?"
• "Which courier performs best?"

💰 **Sales & Revenue**
• "What are the best selling products?"
• "Which region has highest sales?"

Ask me anything! 🤖"""

    if "kpi" in msg or "summary" in msg or "overview" in msg:
        return f"""📊 **OmniFlow-D2D KPI Summary**

💰 Total Revenue: ₹{ctx['total_revenue']/1e6:.2f}M
📦 Total Orders: {ctx['total_orders']:,}
🏆 Top Product (Revenue): {ctx['top_revenue_product']}
📈 Highest Forecast (M+1): {ctx['top_forecast_product']} ({ctx['top_forecast_next_month']:,} units)
🗺️ Top Revenue Region: {ctx['top_region_revenue']}
🚚 Avg Delivery: {ctx['avg_delivery_days']:.1f} days
↩️ Return Rate: {df_all['return_flag'].mean()*100:.1f}%"""

    # Default fallback
    return f"""🤖 I understand you're asking about: **"{user_msg}"**

Try more specific queries like:
• "Forecast for [Product Name]"
• "Reorder point for [Product]"
• "Which region needs logistics support?"
• "Top selling products"
• Type **"help"** for full command list

Available products: {', '.join(list(forecast_results.keys())[:5])}..."""

def page_chatbot(df):
    st.markdown("""
    <div class='page-header'>
        <h1>🤖 AI Decision Intelligence</h1>
        <p>Context-aware chatbot powered by all module outputs — ask supply chain questions</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Building AI context from all modules..."):
        forecast_results = get_forecast_results(df)
        inv_df = compute_inventory(df, forecast_results)
        ctx = build_context(df, forecast_results, inv_df)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            ("bot", "👋 Hello! I'm your OmniFlow-D2D Supply Chain AI. I have full context from all modules. Ask me about demand forecasts, inventory, production, logistics, or sales! Type **help** for examples.")
        ]

    # Display chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"<div class='chat-msg-user'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-msg-bot'>{msg}</div>", unsafe_allow_html=True)

    st.markdown("---")

    # Quick suggestions
    st.markdown("**💡 Quick Queries:**")
    qcols = st.columns(4)
    quick_queries = [
        "Highest demand next month",
        "Which region needs logistics support?",
        "Top 5 products by revenue",
        "Show KPI summary",
        "Reorder point for Monitor",
        "Which courier is best?",
        "Safety stock for Smart Watch",
        "Production capacity insight"
    ]
    for i, qc in enumerate(qcols):
        if qc.button(quick_queries[i], key=f"q{i}"):
            st.session_state._quick_query = quick_queries[i]
        if qc.button(quick_queries[i+4], key=f"q{i+4}"):
            st.session_state._quick_query = quick_queries[i+4]

    # Process quick query
    if hasattr(st.session_state, "_quick_query") and st.session_state._quick_query:
        q = st.session_state._quick_query
        st.session_state._quick_query = None
        response = chatbot_response(q, ctx)
        st.session_state.chat_history.append(("user", q))
        st.session_state.chat_history.append(("bot", response))
        st.rerun()

    # Text input
    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        user_input = st.text_input("Ask a supply chain question...", key="chat_input", label_visibility="collapsed",
                                    placeholder="e.g. What is the reorder point for Wireless Earbuds?")
    with col_btn:
        send = st.button("Send →")

    if send and user_input.strip():
        response = chatbot_response(user_input, ctx)
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("bot", response))
        st.rerun()

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    # Load unified dataset
    df = load_dataset()

    # Sidebar
    with st.sidebar:
        st.markdown("<div class='logo-text'>⟳ OmniFlow-D2D</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center; color:#64748b; font-size:0.7rem; font-family:Space Mono; margin-bottom:1rem;'>Data-to-Decision Platform</div>", unsafe_allow_html=True)
        st.markdown("---")

        pages = {
            "📊 Overview Dashboard": "overview",
            "📈 Demand Forecasting": "forecasting",
            "🏭 Inventory Optimization": "inventory",
            "⚙️ Production Planning": "production",
            "🚚 Logistics Optimization": "logistics",
            "🤖 AI Decision Intelligence": "chatbot"
        }

        if "page" not in st.session_state:
            st.session_state.page = "overview"

        for label, key in pages.items():
            active = "active" if st.session_state.page == key else ""
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key
                st.rerun()

        st.markdown("---")
        # Dataset stats
        st.markdown("<div style='color:#64748b; font-size:0.7rem; font-family:Space Mono; text-transform:uppercase; letter-spacing:0.1em;'>Dataset Stats</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style='background:#111827; border:1px solid #1e293b; border-radius:8px; padding:0.75rem; margin-top:0.5rem;'>
            <div style='font-family:Space Mono; font-size:0.7rem; color:#64748b;'>Records</div>
            <div style='font-family:Space Mono; font-size:1rem; color:#00d4ff;'>{len(df):,}</div>
            <div style='font-family:Space Mono; font-size:0.7rem; color:#64748b; margin-top:0.4rem;'>Products</div>
            <div style='font-family:Space Mono; font-size:1rem; color:#7c3aed;'>{df['product'].nunique()}</div>
            <div style='font-family:Space Mono; font-size:0.7rem; color:#64748b; margin-top:0.4rem;'>Date Range</div>
            <div style='font-family:Space Mono; font-size:0.75rem; color:#94a3b8;'>Jan 2024 – Feb 2025</div>
            <div style='font-family:Space Mono; font-size:0.7rem; color:#64748b; margin-top:0.4rem;'>Sources</div>
            <div style='font-family:Space Mono; font-size:0.65rem; color:#94a3b8;'>Amazon India 2025<br>Shiprocket/INCREFF</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style='margin-top:1rem; color:#374151; font-size:0.65rem; text-align:center; font-family:Space Mono;'>
        Built with Streamlit + Statsmodels<br>Plotly + Pandas + NumPy
        </div>
        """, unsafe_allow_html=True)

    # Route to page
    page = st.session_state.get("page", "overview")
    if page == "overview":
        page_overview(df)
    elif page == "forecasting":
        page_forecasting(df)
    elif page == "inventory":
        page_inventory(df)
    elif page == "production":
        page_production(df)
    elif page == "logistics":
        page_logistics(df)
    elif page == "chatbot":
        page_chatbot(df)

if __name__ == "__main__":
    main()
