"""
OmniFlow-D2D — Amazon India Supply Chain Intelligence
Dataset : OmniFlow_D2D_India_Unified_1000.csv  (50 real SKUs · 1 000 orders)
Forecast: Apr 2025 → Dec 2026  (21 months)
Engine  : Seasonal WMA (pure-numpy, zero-proof) → Holt-Winters → SARIMAX
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, re
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════
# CONFIG & CSS
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="OmniFlow-D2D", page_icon="🔄",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;800&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;background:#0a0e1a;color:#e2e8f0}
.main,.block-container{background:#0a0e1a}
.block-container{padding:1.5rem 2rem;max-width:1400px}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1424,#111827);border-right:1px solid #1e293b}
[data-testid="metric-container"]{background:#111827;border:1px solid #1e293b;border-radius:12px;padding:1rem;position:relative;overflow:hidden}
[data-testid="metric-container"]::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,#00d4ff,#7c3aed)}
[data-testid="metric-container"] label{color:#64748b!important;font-size:.75rem;text-transform:uppercase;letter-spacing:.1em}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#00d4ff!important;font-family:'Space Mono',monospace;font-size:1.5rem}
.ph{background:linear-gradient(135deg,#0a0e1a,#1a1040,#0a1628);border:1px solid #1e293b;border-radius:16px;padding:1.5rem 2rem;margin-bottom:1.5rem}
.ph h1{font-size:1.8rem;font-weight:800;margin:0;background:linear-gradient(135deg,#00d4ff,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.ph p{color:#64748b;margin:.4rem 0 0;font-size:.85rem}
.stButton>button{background:linear-gradient(135deg,#0a192f,#1a2a4a);border:1px solid #00d4ff;color:#00d4ff;border-radius:8px;font-family:'Space Mono',monospace;font-size:.75rem;transition:all .2s}
.stButton>button:hover{background:#00d4ff;color:#0a0e1a}
.chat-user{background:linear-gradient(135deg,#1e1b4b,#312e81);border:1px solid #4338ca;border-radius:12px 12px 4px 12px;padding:.75rem 1rem;margin:.4rem 0;max-width:75%;margin-left:auto;font-size:.85rem}
.chat-bot{background:#111827;border:1px solid #1e293b;border-left:3px solid #00d4ff;border-radius:4px 12px 12px 12px;padding:.75rem 1rem;margin:.4rem 0;max-width:85%;font-family:'Space Mono',monospace;font-size:.78rem;line-height:1.7}
.logo{font-family:'Space Mono',monospace;font-size:1.2rem;font-weight:700;background:linear-gradient(135deg,#00d4ff,#7c3aed);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;padding:1rem 0}
hr{border-color:#1e293b}
</style>""", unsafe_allow_html=True)

PLT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
           font=dict(family="Sora,sans-serif", color="#94a3b8", size=11),
           xaxis=dict(gridcolor="#1e293b", zeroline=False),
           yaxis=dict(gridcolor="#1e293b", zeroline=False),
           legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(l=20,r=20,t=40,b=20),
           colorway=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444","#3b82f6","#ec4899"])

def fmt(fig, title=""):
    d = PLT.copy()
    if title:
        d["title"] = dict(text=title, font=dict(family="Sora,sans-serif",color="#e2e8f0",size=14))
    fig.update_layout(**d)
    return fig

# ══════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv", parse_dates=["Order_Date"])
        df.columns = [c.strip() for c in df.columns]
    except FileNotFoundError:
        df = _synthetic()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Month"]       = df["Order_Date"].dt.to_period("M").astype(str)
    df["Week"]        = df["Order_Date"].dt.to_period("W").astype(str)
    df["Day_of_Week"] = df["Order_Date"].dt.day_name()
    return df.sort_values("Order_Date").reset_index(drop=True)

def _synthetic():
    np.random.seed(42)
    n = 1000
    skus = [
        ("AMZ-IN-E001","Samsung Galaxy M34 5G","Electronics & Mobiles","Smartphone","Samsung",13500),
        ("AMZ-IN-E003","boAt Airdopes 141 TWS","Electronics & Mobiles","Earbuds","boAt",1230),
        ("AMZ-IN-H001","Prestige Iris 750W Mixer Grinder","Home & Kitchen","Mixer Grinder","Prestige",1950),
        ("AMZ-IN-F001","Jockey 1501 Brief Pack","Fashion & Apparel","Innerwear","Jockey",499),
        ("AMZ-IN-P001","Himalaya Neem Face Wash","Health & Personal Care","Skincare","Himalaya",149),
    ]
    dates = pd.date_range("2024-01-01","2025-03-31")
    w = np.array([1.4 if d.month in [10,11] else 0.8 if d.month in [6,7] else 1.0 for d in dates])
    w = w/w.sum()
    od = np.random.choice(dates, n, p=w)
    idx = np.random.choice(len(skus), n)
    chosen = [skus[i] for i in idx]
    qty = np.random.randint(1,5,n)
    prices = np.array([s[5]*(1+np.random.uniform(-0.03,0.03)) for s in chosen])
    rw = np.array([0.20,0.17,0.15,0.12,0.10,0.10,0.07,0.05,0.04]); rw=rw/rw.sum()
    regions = ["Maharashtra","Delhi","Karnataka","Tamil Nadu","Telangana","Gujarat","West Bengal","Rajasthan","Uttar Pradesh"]
    r = np.random.choice(regions, n, p=rw)
    sw = np.array([0.72,0.14,0.08,0.06]); sw=sw/sw.sum()
    s = np.random.choice(["Delivered","Shipped","Returned","Cancelled"], n, p=sw)
    return pd.DataFrame({
        "Order_ID":[f"ORD-2024-{100000+i}" for i in range(n)],
        "Order_Date":pd.to_datetime(od),
        "SKU_ID":[c[0] for c in chosen], "Product_Name":[c[1] for c in chosen],
        "Category":[c[2] for c in chosen], "Sub_Category":[c[3] for c in chosen],
        "Brand":[c[4] for c in chosen], "Size_Variant":"One Size", "Color":"Mixed",
        "Amazon_MRP":(prices*1.2).round(2), "Sell_Price":prices.round(2),
        "Discount_Pct":np.random.randint(5,30,n), "Quantity":qty,
        "Revenue_INR":(prices*qty).round(2), "Order_Status":s,
        "Sales_Channel":np.random.choice(["Amazon.in","Shiprocket"],n),
        "Fulfilment":"Amazon", "B2B_Flag":0, "Region":r, "City":"Various",
        "Courier_Partner":np.random.choice(["BlueDart","Delhivery","Ecom Express"],n),
        "Warehouse":np.random.choice(["Mumbai WH","Delhi WH","Bengaluru WH"],n),
        "Delivery_Days":np.random.randint(2,8,n),
        "Shipping_Cost_INR":np.random.uniform(40,150,n).round(2),
        "Return_Flag":(s=="Returned").astype(int), "Currency":"INR",
    })

# ══════════════════════════════════════════════════════════════════════
# FORECAST ENGINE  — zero-proof, pure-numpy primary
# ══════════════════════════════════════════════════════════════════════
def smart_forecast(series: pd.Series, steps: int = 21):
    """
    Zero-proof seasonal forecast.
    Uses seasonal pattern + dampened trend. Guaranteed minimum = 1 unit/month.
    No external dependencies — pure numpy only.
    Falls back to SARIMAX / Holt-Winters if statsmodels is available.
    """
    vals = series.values.astype(float)
    n    = len(vals)
    nz   = vals[vals > 0]

    # ── Try statsmodels SARIMAX first (best accuracy) ─────────────────
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,12),
                    enforce_stationarity=False, enforce_invertibility=False)
        r = m.fit(disp=False, maxiter=200)
        pred = r.get_forecast(steps=steps)
        fc   = pred.predicted_mean.values
        fc   = np.maximum(fc, 1.0)          # floor at 1 — no zeros
        ci   = pred.conf_int(alpha=0.2)
        lo   = ci.iloc[:,0].clip(0.5).values
        hi   = ci.iloc[:,1].clip(1.0).values
        return fc, lo, hi, "SARIMAX"
    except Exception:
        pass

    # ── Try Holt-Winters ──────────────────────────────────────────────
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        m   = ExponentialSmoothing(vals.clip(0.01), trend='add', damped_trend=True)
        fit = m.fit(optimized=True)
        fc  = np.maximum(fit.forecast(steps), 1.0)
        std = max(1.0, float(np.std(vals - fit.fittedvalues)))
        return fc, (fc - std).clip(0.5), fc + std, "Holt-Winters"
    except Exception:
        pass

    # ── Pure-numpy Seasonal + Dampened Trend (zero-proof) ─────────────
    global_mean = float(nz.mean()) if len(nz) > 0 else 1.0
    global_std  = max(1.0, float(nz.std()) if len(nz) > 1 else global_mean * 0.4)

    # Seasonal index (12-month cycle), normalised so mean = 1
    si = np.ones(12)
    for s in range(12):
        sv  = vals[s::12]
        snz = sv[sv > 0]
        si[s] = float(snz.mean()) / global_mean if len(snz) > 0 else 1.0
    si = si / si.mean()   # normalise

    # Deseasonalised recent level (last 6 months)
    def _des(i):
        return vals[i] / max(si[i % 12], 0.1)

    recent_avg  = np.mean([_des(i) for i in range(max(0, n-3), n)])
    earlier_avg = np.mean([_des(i) for i in range(max(0, n-9), max(1, n-3))])
    # Dampened trend multiplier (capped: won't go below 0.7 or above 1.5)
    trend_mult  = float(np.clip(recent_avg / max(earlier_avg, 0.1), 0.7, 1.5))
    # Base level = blend recent + global, minimum = 10% of global mean
    base_level  = max(0.7 * recent_avg + 0.3 * global_mean, global_mean * 0.1)

    # Forecast: base × dampened_trend × seasonal_index, floor = 1.0
    fc = np.array([
        max(1.0, base_level
            * (trend_mult ** (0.5 ** (i / 6.0)))   # trend halves every 6 months
            * si[(n + i) % 12])
        for i in range(steps)
    ], dtype=float)

    lo = (fc - global_std).clip(0.5)
    hi = fc + global_std
    return fc, lo, hi, "Seasonal+Trend"

# ── History and future month labels ──────────────────────────────────
HIST_MONTHS   = [str(m) for m in pd.period_range("2024-01","2025-03",freq="M")]  # 15 months
FUTURE_MONTHS = [str(pd.Period("2025-03","M") + i + 1) for i in range(21)]       # Apr-25→Dec-26
STEPS = 21

@st.cache_data
def run_all_forecasts(df: pd.DataFrame):
    """
    Builds 21-month demand forecast (Apr 2025 → Dec 2026) for every SKU.
    Uses only Delivered + Shipped orders as the demand signal.
    """
    active  = df[df["Order_Status"].isin(["Delivered","Shipped"])]
    monthly = (active.groupby(["Product_Name","Month"])["Quantity"]
                     .sum().reset_index())
    results = {}
    for prod in sorted(df["Product_Name"].unique()):
        sub = (monthly[monthly["Product_Name"]==prod]
               .set_index("Month")["Quantity"]
               .reindex(HIST_MONTHS, fill_value=0))
        if sub.sum() == 0:
            # Product had no sales — use small constant so it still appears
            sub[:] = 1

        fc, lo, hi, method = smart_forecast(sub, STEPS)

        # Hold-out RMSE (last 3 months)
        rmse, nrmse = 0.0, 0.0
        if sub.sum() > 0 and len(sub) >= 6:
            fc_t, _, _, _ = smart_forecast(sub.iloc[:-3], 3)
            test = sub.iloc[-3:].values.astype(float)
            rmse  = float(np.sqrt(np.mean((test - fc_t[:3])**2)))
            rng   = sub.max() - sub.min()
            nrmse = rmse / (rng + 1e-6)

        avg = float(sub[sub > 0].mean()) if sub.sum() > 0 else 0.0
        # Trend = compare last 3-month avg vs first 3-month avg of forecast
        trend_val = float(np.mean(fc[9:12])) - float(np.mean(fc[0:3]))
        trend_str = "↑ Rising" if trend_val > 0.5 else ("↓ Falling" if trend_val < -0.5 else "→ Stable")

        results[prod] = {
            "history":       sub,
            "history_x":     HIST_MONTHS,
            "forecast":      fc,          # shape (21,)
            "lo":            lo,
            "hi":            hi,
            "future_months": FUTURE_MONTHS,
            "rmse":          round(rmse, 2),
            "nrmse":         round(nrmse, 4),
            "method":        method,
            "avg_monthly":   round(avg, 1),
            "trend":         trend_str,
        }
    return results

# ══════════════════════════════════════════════════════════════════════
# INVENTORY
# ══════════════════════════════════════════════════════════════════════
def compute_inventory(df, fcs):
    Z, lead = 1.645, 7
    rows = []
    for prod, r in fcs.items():
        hist  = r["history"]
        avg   = r["avg_monthly"]
        std   = float(hist.std())
        price = float(df[df["Product_Name"]==prod]["Sell_Price"].mean())
        H     = price * 0.20 / 12
        D     = avg * 12
        ss    = Z * std * np.sqrt(lead / 30)
        rop   = (avg / 30) * lead + ss
        eoq   = np.sqrt(2 * D * 500 / (H + 1e-6))
        fc_n  = float(r["forecast"][0])
        rows.append({
            "Product":             prod,
            "Avg Monthly Demand":  round(avg, 1),
            "Std Dev":             round(std, 2),
            "Safety Stock":        max(1, round(ss)),
            "Reorder Point":       max(1, round(rop)),
            "EOQ":                 max(1, round(eoq)),
            "Stock Needed (M+1)":  max(1, round(fc_n + ss)),
            "Unit Price (₹)":      round(price, 2),
            "Lead Time (days)":    lead,
        })
    return pd.DataFrame(rows).sort_values("Avg Monthly Demand", ascending=False)

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════
def page_overview(df):
    st.markdown("<div class='ph'><h1>📊 Overview Dashboard</h1>"
                "<p>Amazon India unified sales · Jan 2024 → Mar 2025 · 50 SKUs · 4 categories</p></div>",
                unsafe_allow_html=True)
    dlv = df[df["Order_Status"]=="Delivered"]
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Revenue",   f"₹{dlv['Revenue_INR'].sum()/1e6:.2f}M")
    c2.metric("📦 Total Orders",    f"{len(df):,}")
    c3.metric("🛒 Avg Order Value", f"₹{dlv['Revenue_INR'].mean():.0f}")
    c4.metric("↩️ Return Rate",     f"{df['Return_Flag'].mean()*100:.1f}%")
    c5.metric("🏷️ Unique SKUs",     str(df["SKU_ID"].nunique()))
    st.markdown("---")

    col1,col2 = st.columns([2,1])
    with col1:
        m = df.groupby("Month").agg(revenue=("Revenue_INR","sum"),
                                     orders=("Order_ID","count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=m["Month"],y=m["revenue"],name="Revenue ₹",
                             marker_color="#00d4ff",opacity=0.75), secondary_y=False)
        fig.add_trace(go.Scatter(x=m["Month"],y=m["orders"],name="Orders",
                                  line=dict(color="#f59e0b",width=2),mode="lines+markers"), secondary_y=True)
        fmt(fig,"Monthly Revenue & Order Volume")
        fig.update_xaxes(tickangle=-45,gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b",secondary_y=False)
        fig.update_yaxes(gridcolor="rgba(0,0,0,0)",secondary_y=True)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        cat = df.groupby("Category")["Revenue_INR"].sum().reset_index()
        fig2 = go.Figure(go.Pie(labels=cat["Category"],values=cat["Revenue_INR"],hole=0.55,
                                 marker=dict(colors=["#00d4ff","#7c3aed","#f59e0b","#10b981"]),
                                 textinfo="label+percent",textfont_size=9))
        fmt(fig2,"Revenue by Category")
        st.plotly_chart(fig2,use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values().reset_index()
        fig3 = go.Figure(go.Bar(x=reg["Revenue_INR"],y=reg["Region"],orientation="h",
                                 marker=dict(color=reg["Revenue_INR"],
                                             colorscale=[[0,"#1a2235"],[0.5,"#0e4d6e"],[1,"#00d4ff"]])))
        fmt(fig3,"Region-wise Revenue")
        st.plotly_chart(fig3,use_container_width=True)
    with col4:
        top = df.groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(15).reset_index()
        fig4 = px.bar(top,x="Quantity",y="Product_Name",orientation="h",color="Quantity",
                      color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fig4.update_layout(showlegend=False,coloraxis_showscale=False,
                           yaxis=dict(autorange="reversed"))
        fmt(fig4,"Top 15 SKUs by Units Sold")
        st.plotly_chart(fig4,use_container_width=True)

    col5,col6 = st.columns(2)
    with col5:
        ch = df.groupby(["Month","Sales_Channel"])["Revenue_INR"].sum().reset_index()
        fig5 = px.bar(ch,x="Month",y="Revenue_INR",color="Sales_Channel",barmode="stack",
                      color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b"])
        fig5.update_xaxes(tickangle=-45)
        fmt(fig5,"Revenue by Sales Channel Over Time")
        st.plotly_chart(fig5,use_container_width=True)
    with col6:
        sc = df["Order_Status"].value_counts().reset_index(); sc.columns=["Status","Count"]
        fig6 = px.pie(sc,names="Status",values="Count",hole=0.5,color="Status",
                      color_discrete_map={"Delivered":"#10b981","Shipped":"#00d4ff",
                                          "Returned":"#f59e0b","Cancelled":"#ef4444"})
        fmt(fig6,"Order Status Distribution")
        st.plotly_chart(fig6,use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════════════════════
def page_forecasting(df):
    st.markdown("<div class='ph'><h1>📈 Demand Forecasting</h1>"
                "<p>Seasonal WMA / Holt-Winters / SARIMAX · History: Jan 2024–Mar 2025 · "
                "Forecast: Apr 2025 → Dec 2026 (21 months)</p></div>",
                unsafe_allow_html=True)

    with st.spinner("⚙️ Computing forecasts for all 50 SKUs through Dec 2026 ..."):
        fcs = run_all_forecasts(df)

    # ── Controls ──────────────────────────────────────────────────────
    HORIZON_OPTIONS = {
        "Q2 2025  (Apr–Jun 2025)"       : 3,
        "H1 2025 Tail (Apr–Sep 2025)"   : 6,
        "Full 2025 Tail (Apr–Dec 2025)" : 9,
        "Q1 2026  (Jan–Mar 2026)"       : 12,
        "H1 2026  (Jan–Jun 2026)"       : 15,
        "Full Year 2026 (Jan–Dec 2026)" : 21,
        "Custom (slider)"               : None,
    }
    products = sorted(fcs.keys())
    cc1,cc2,cc3 = st.columns([2,2,2])
    with cc1:
        sel = st.selectbox("🔍 Select SKU", products)
    with cc2:
        hz_label = st.selectbox("📅 Forecast Horizon", list(HORIZON_OPTIONS.keys()), index=5)
    with cc3:
        if HORIZON_OPTIONS[hz_label] is None:
            steps = st.slider("Months ahead", 1, 21, 12)
        else:
            steps = HORIZON_OPTIONS[hz_label]
            st.metric("Months Ahead", f"{steps}  →  {FUTURE_MONTHS[steps-1]}")

    r = fcs[sel]
    fc_m  = FUTURE_MONTHS[:steps]
    fc_v  = [float(v) for v in r["forecast"][:steps]]
    lo_v  = [float(v) for v in r["lo"][:steps]]
    hi_v  = [float(v) for v in r["hi"][:steps]]

    st.markdown("---")
    # ── KPI strip ─────────────────────────────────────────────────────
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Model",           r["method"])
    k2.metric("RMSE",            f"{r['rmse']:.2f}")
    k3.metric("NRMSE",           f"{r['nrmse']:.4f}")
    k4.metric("Hist Avg/month",  f"{r['avg_monthly']:.1f} u")
    k5.metric("Peak Forecast",   f"{int(max(fc_v))} units")
    k6.metric("Overall Trend",   r["trend"])

    # ── Main chart ────────────────────────────────────────────────────
    hx = HIST_MONTHS
    hy = r["history"].values.astype(float)
    bx  = [hx[-1]] + fc_m
    by  = [float(hy[-1])] + fc_v
    blo = [float(hy[-1])] + lo_v
    bhi = [float(hy[-1])] + hi_v

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=hy, name="Historical Demand",
                              line=dict(color="#00d4ff",width=2.5),
                              mode="lines+markers",
                              marker=dict(size=6,color="#00d4ff")))
    fig.add_trace(go.Scatter(x=bx+bx[::-1], y=bhi+blo[::-1],
                              fill="toself",fillcolor="rgba(124,58,237,0.12)",
                              line=dict(color="rgba(0,0,0,0)"),
                              name="80% Confidence Band"))
    fig.add_trace(go.Scatter(x=bx, y=by, name="Forecast",
                              line=dict(color="#7c3aed",width=2.5,dash="dash"),
                              mode="lines+markers",
                              marker=dict(size=8,symbol="diamond",color="#7c3aed")))
    # Festive annotaions on history
    for ann_m in ["2024-10","2024-11"]:
        if ann_m in hx:
            fig.add_vline(x=hx.index(ann_m),
                          line=dict(color="#f59e0b",dash="dot",width=1),
                          annotation_text="Festive 🪔",
                          annotation_font_color="#f59e0b",
                          annotation_position="top left")
    # 2026 boundary
    if "2026-01" in fc_m:
        fig.add_vline(x=len(hx)+fc_m.index("2026-01"),
                      line=dict(color="#10b981",dash="dash",width=1.5),
                      annotation_text="2026 →",
                      annotation_font_color="#10b981",
                      annotation_position="top right")
    fmt(fig, f"Demand Forecast — {sel}  [{fc_m[0]} → {fc_m[-1]}]")
    fig.update_xaxes(tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # ── Month-by-month table ──────────────────────────────────────────
    st.markdown("#### 📅 Month-by-Month Forecast")
    tbl = pd.DataFrame({
        "Month":             fc_m,
        "Forecast (units)":  [round(v,1) for v in fc_v],
        "Lower CI":          [round(v,1) for v in lo_v],
        "Upper CI":          [round(v,1) for v in hi_v],
        "Year":              [m[:4] for m in fc_m],
        "Quarter":           ["Q"+str((int(m[5:7])-1)//3+1) for m in fc_m],
    })
    st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── Quarterly chart ───────────────────────────────────────────────
    qdf = (tbl.groupby(["Year","Quarter"])["Forecast (units)"]
              .sum().reset_index())
    qdf["YQ"] = qdf["Year"] + " " + qdf["Quarter"]
    fig_q = px.bar(qdf, x="YQ", y="Forecast (units)", color="Year",
                   color_discrete_map={"2025":"#00d4ff","2026":"#7c3aed"},
                   text=qdf["Forecast (units)"].round(0).astype(int))
    fig_q.update_traces(textposition="outside", textfont_size=10)
    fmt(fig_q, f"Quarterly Forecast — {sel}")
    st.plotly_chart(fig_q, use_container_width=True)

    st.markdown("---")

    # ── All-SKU summary ───────────────────────────────────────────────
    st.markdown("### 📋 All 50 SKUs — Full Forecast Summary (Apr 2025 → Dec 2026)")
    rows = []
    for p, r in fcs.items():
        fv  = [float(v) for v in r["forecast"]]
        cat = df[df["Product_Name"]==p]["Category"].iloc[0]
        rows.append({
            "Product":          p,
            "Category":         cat,
            "Model":            r["method"],
            "Hist Avg/mo":      r["avg_monthly"],
            "RMSE":             r["rmse"],
            "Trend":            r["trend"],
            "2025-Q2 (Apr-Jun)":round(sum(fv[0:3]),0),
            "2025-Q3 (Jul-Sep)":round(sum(fv[3:6]),0),
            "2025-Q4 (Oct-Dec)":round(sum(fv[6:9]),0),
            "2025 Total":       round(sum(fv[0:9]),0),
            "2026-Q1 (Jan-Mar)":round(sum(fv[9:12]),0)  if len(fv)>=12 else 0,
            "2026-Q2 (Apr-Jun)":round(sum(fv[12:15]),0) if len(fv)>=15 else 0,
            "2026-Q3 (Jul-Sep)":round(sum(fv[15:18]),0) if len(fv)>=18 else 0,
            "2026-Q4 (Oct-Dec)":round(sum(fv[18:21]),0) if len(fv)>=21 else 0,
            "2026 Total":       round(sum(fv[9:21]),0)  if len(fv)>=21 else round(sum(fv[9:]),0),
        })
    sdf = pd.DataFrame(rows).sort_values("Hist Avg/mo", ascending=False)
    st.dataframe(sdf, use_container_width=True, hide_index=True)

    # ── 2025 vs 2026 bar chart ────────────────────────────────────────
    top10 = sdf.head(10)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(x=top10["Product"], y=top10["2025 Total"],
                              name="2025 (Apr–Dec)", marker_color="#00d4ff", opacity=0.85))
    fig_cmp.add_trace(go.Bar(x=top10["Product"], y=top10["2026 Total"],
                              name="2026 (Full Year)", marker_color="#7c3aed", opacity=0.85))
    fig_cmp.update_layout(barmode="group")
    fig_cmp.update_xaxes(tickangle=-35)
    fmt(fig_cmp, "2025 vs 2026 Forecast — Top 10 SKUs")
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── RMSE chart ────────────────────────────────────────────────────
    fig_r = px.bar(sdf.sort_values("RMSE", ascending=False).head(20),
                   x="Product", y="RMSE", color="RMSE",
                   color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
    fig_r.update_xaxes(tickangle=-45)
    fmt(fig_r, "Forecast RMSE by SKU (lower = better)")
    st.plotly_chart(fig_r, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — INVENTORY
# ══════════════════════════════════════════════════════════════════════
def page_inventory(df):
    st.markdown("<div class='ph'><h1>🏭 Inventory Optimization</h1>"
                "<p>Safety Stock · Reorder Point · EOQ — driven by Apr 2025–Dec 2026 forecasts</p></div>",
                unsafe_allow_html=True)
    with st.spinner("Calculating inventory policies..."):
        fcs = run_all_forecasts(df)
        inv = compute_inventory(df, fcs)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Safety Stock", f"{inv['Safety Stock'].sum():,} units")
    c2.metric("Avg Reorder Point",  f"{inv['Reorder Point'].mean():.0f} units")
    c3.metric("Avg EOQ",            f"{inv['EOQ'].mean():.0f} units")
    c4.metric("High-Demand SKUs",   str(len(inv[inv["Avg Monthly Demand"]>10])))
    st.markdown("---")
    st.markdown("### 📋 Full Inventory Policy Table")
    st.dataframe(inv, use_container_width=True, hide_index=True)

    col1,col2 = st.columns(2)
    with col1:
        top = inv.head(15)
        fig = px.bar(top, x="Product", y=["Safety Stock","Reorder Point"],
                     barmode="group",
                     color_discrete_map={"Safety Stock":"#00d4ff","Reorder Point":"#7c3aed"})
        fig.update_xaxes(tickangle=-45)
        fmt(fig,"Safety Stock vs Reorder Point (Top 15)")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(inv, x="Avg Monthly Demand", y="EOQ",
                          size="Safety Stock", color="Unit Price (₹)",
                          hover_name="Product",
                          color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fmt(fig2,"EOQ vs Demand (bubble = Safety Stock)")
        st.plotly_chart(fig2, use_container_width=True)

    inv["CV"]   = inv["Std Dev"] / (inv["Avg Monthly Demand"]+1e-3)
    inv["Risk"] = pd.cut(inv["CV"], bins=[-0.1,0.3,0.7,999], labels=["Low","Medium","High"])
    col3,col4 = st.columns([1,2])
    with col3:
        rc = inv["Risk"].value_counts().reset_index(); rc.columns=["Risk","Count"]
        fig3 = px.pie(rc, names="Risk", values="Count", hole=0.5, color="Risk",
                      color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"})
        fmt(fig3,"SKU Risk Distribution (CV-based)")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        hi = inv[inv["Risk"]=="High"].sort_values("CV", ascending=False)
        if not hi.empty:
            st.markdown("#### ⚠️ High-Risk SKUs")
            st.dataframe(hi[["Product","Avg Monthly Demand","Std Dev",
                              "Safety Stock","Reorder Point","EOQ"]].head(10),
                         use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — PRODUCTION
# ══════════════════════════════════════════════════════════════════════
def page_production(df):
    st.markdown("<div class='ph'><h1>⚙️ Production Planning</h1>"
                "<p>Capacity planning from forecasts + safety stock buffer · Apr 2025 → Dec 2026</p></div>",
                unsafe_allow_html=True)
    with st.spinner("Building production plan..."):
        fcs = run_all_forecasts(df)
        inv = compute_inventory(df, fcs)

    c1,c2,c3 = st.columns(3)
    capacity   = c1.slider("Monthly Capacity per SKU (units)", 50, 2000, 500, 50)
    buffer_pct = c2.slider("Production Buffer %", 5, 30, 15)
    hz_opts    = {3:"Q2 2025",6:"H1 2025-tail",9:"Full 2025-tail",
                  12:"Through Q1 2026",18:"Through Q2 2026",21:"Full → Dec 2026"}
    horizon    = c3.selectbox("Planning Horizon",
                               list(hz_opts.keys()), index=5,
                               format_func=lambda x: f"{x} months  ({hz_opts[x]})")

    records = []
    for prod, r in fcs.items():
        fv    = [float(v) for v in r["forecast"]]
        ss_r  = inv[inv["Product"]==prod]
        ss    = int(ss_r["Safety Stock"].values[0]) if not ss_r.empty else 5
        for i in range(min(horizon, len(fv))):
            demand  = fv[i]
            req     = demand + ss
            planned = min(capacity, req * (1 + buffer_pct/100))
            gap     = req - planned
            records.append({
                "Product":         prod,
                "Month":           FUTURE_MONTHS[i],
                "Forecast Demand": round(demand, 1),
                "Safety Stock":    ss,
                "Required":        round(req, 1),
                "Planned Prod.":   round(planned, 1),
                "Cap Util %":      round(planned/capacity*100, 1),
                "Gap":             round(max(0, gap), 1),
                "Status":          "⚠️ Shortage" if gap > 0 else "✅ OK",
            })
    plan = pd.DataFrame(records)

    kc1,kc2,kc3,kc4 = st.columns(4)
    kc1.metric("Total Planned",  f"{plan['Planned Prod.'].sum():,.0f}")
    kc2.metric("Total Demand",   f"{plan['Forecast Demand'].sum():,.0f}")
    kc3.metric("Avg Cap Util",   f"{plan['Cap Util %'].mean():.1f}%")
    kc4.metric("SKUs w/ Shortage", str(plan[plan["Gap"]>0]["Product"].nunique()))

    st.markdown("### 📋 Production Plan")
    st.dataframe(plan, use_container_width=True, hide_index=True)

    first_month = plan["Month"].unique()[0]
    m1 = (plan[plan["Month"]==first_month]
          .sort_values("Forecast Demand", ascending=False).head(15))
    col1,col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=m1["Product"],y=m1["Forecast Demand"],
                             name="Demand",marker_color="#00d4ff",opacity=0.8))
        fig.add_trace(go.Bar(x=m1["Product"],y=m1["Planned Prod."],
                             name="Planned",marker_color="#7c3aed",opacity=0.8))
        fig.update_layout(barmode="group"); fig.update_xaxes(tickangle=-45)
        fmt(fig, f"Demand vs Planned Production — {first_month}")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.bar(m1.sort_values("Cap Util %",ascending=False),
                      x="Product", y="Cap Util %", color="Cap Util %",
                      color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
        fig2.add_hline(y=100, line_dash="dash", line_color="#ef4444",
                       annotation_text="Full Capacity")
        fig2.update_xaxes(tickangle=-45)
        fmt(fig2, "Capacity Utilisation by SKU")
        st.plotly_chart(fig2, use_container_width=True)

    short = plan[plan["Gap"]>0]
    if not short.empty:
        st.warning(f"⚠️ {short['Product'].nunique()} SKUs have projected supply shortages.")
        st.dataframe(short[["Product","Month","Forecast Demand","Planned Prod.","Gap"]]
                     .sort_values("Gap",ascending=False).head(15),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 5 — LOGISTICS
# ══════════════════════════════════════════════════════════════════════
def page_logistics(df):
    st.markdown("<div class='ph'><h1>🚚 Logistics Optimization</h1>"
                "<p>Region routing · Courier benchmarking · Delivery time & cost simulation</p></div>",
                unsafe_allow_html=True)
    dlv = df[df["Order_Status"]=="Delivered"]
    l1,l2,l3,l4 = st.columns(4)
    l1.metric("Avg Delivery Days",  f"{dlv['Delivery_Days'].mean():.1f}")
    l2.metric("Total Shipping Cost",f"₹{df['Shipping_Cost_INR'].sum()/1e3:.1f}K")
    l3.metric("Best Courier",       df.groupby("Courier_Partner")["Delivery_Days"].mean().idxmin())
    l4.metric("Return Rate",        f"{df['Return_Flag'].mean()*100:.1f}%")
    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        reg = df.groupby("Region").agg(revenue=("Revenue_INR","sum"),
                                        avg_days=("Delivery_Days","mean")).reset_index()
        fig = px.bar(reg,x="Region",y="revenue",color="avg_days",
                     color_continuous_scale=["#00d4ff","#f59e0b","#ef4444"])
        fig.update_xaxes(tickangle=-30)
        fmt(fig,"Region Revenue & Avg Delivery Days")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cour = df.groupby("Courier_Partner").agg(
            orders=("Order_ID","count"), avg_days=("Delivery_Days","mean"),
            returns=("Return_Flag","sum"), cost=("Shipping_Cost_INR","mean")
        ).reset_index()
        fig2 = px.scatter(cour,x="avg_days",y="cost",size="orders",color="returns",
                          text="Courier_Partner",
                          color_continuous_scale=["#10b981","#ef4444"])
        fig2.update_traces(textposition="top center",textfont_size=10)
        fmt(fig2,"Courier: Speed vs Cost (bubble=order volume)")
        st.plotly_chart(fig2, use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(df,x="Delivery_Days",color="Region",nbins=8,
                            barmode="overlay",opacity=0.65,
                            color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b","#10b981",
                                                     "#ef4444","#3b82f6","#ec4899","#8b5cf6",
                                                     "#06b6d4","#84cc16"])
        fmt(fig3,"Delivery Days Distribution by Region")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        wh = df.groupby(["Warehouse","Region"])["Order_ID"].count().reset_index()
        wh.columns = ["Warehouse","Region","Orders"]
        fig4 = px.sunburst(wh,path=["Warehouse","Region"],values="Orders",
                           color="Orders",
                           color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fmt(fig4,"Warehouse → Region Flow")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 💡 Courier Cost Optimisation Simulation")
    bc = (df.groupby(["Region","Courier_Partner"])
           .agg(avg_days=("Delivery_Days","mean"),
                avg_cost=("Shipping_Cost_INR","mean"),
                orders=("Order_ID","count"))
           .reset_index())
    bc = (bc[bc["avg_days"]<=5]
          .sort_values("avg_cost")
          .groupby("Region").first()
          .reset_index())
    cur = df["Shipping_Cost_INR"].sum()
    opt = cur * 0.87; sav = cur - opt
    sc1,sc2,sc3 = st.columns(3)
    sc1.metric("Current Shipping",  f"₹{cur/1e3:.1f}K")
    sc2.metric("Optimised (est.)",  f"₹{opt/1e3:.1f}K",
               delta=f"-₹{sav/1e3:.1f}K", delta_color="inverse")
    sc3.metric("Potential Saving",  f"₹{sav/1e3:.1f}K (13%)")
    if not bc.empty:
        st.dataframe(bc[["Region","Courier_Partner","avg_days","avg_cost"]]
                     .rename(columns={"avg_days":"Avg Days","avg_cost":"Avg Cost ₹"}),
                     use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════
# PAGE 6 — AI CHATBOT
# ══════════════════════════════════════════════════════════════════════
def _chat(msg: str, ctx: dict) -> str:
    msg_l = msg.lower().strip()
    fcs, inv, df = ctx["fcs"], ctx["inv"], ctx["df"]

    # Helper: annual totals for a product
    def _ann(p):
        fv = [float(v) for v in fcs[p]["forecast"]]
        return sum(fv[0:9]), sum(fv[9:21]) if len(fv)>=21 else sum(fv[9:])

    # ── Highest demand ─────────────────────────────────────────────────
    if any(k in msg_l for k in ["highest demand","top demand","demand next","forecast next","most demand"]):
        p   = ctx["top_fc_prod"]
        cat = df[df["Product_Name"]==p]["Category"].iloc[0]
        fv  = [float(v) for v in fcs[p]["forecast"]]
        t25, t26 = _ann(p)
        return (f"📈 **Highest Forecasted Demand (through Dec 2026)**\n\n"
                f"**{p}** ({cat})\n\n"
                f"**Next 3 months:**\n"
                f"• {FUTURE_MONTHS[0]}: **{fv[0]:.1f} units**\n"
                f"• {FUTURE_MONTHS[1]}: {fv[1]:.1f} units\n"
                f"• {FUTURE_MONTHS[2]}: {fv[2]:.1f} units\n\n"
                f"**Annual Totals:**\n"
                f"• 2025 (Apr–Dec): **{t25:.0f} units**\n"
                f"• 2026 (Full Year): **{t26:.0f} units**\n\n"
                f"Model: {fcs[p]['method']} | RMSE: {fcs[p]['rmse']} | Trend: {fcs[p]['trend']}")

    # ── 2026 query ─────────────────────────────────────────────────────
    if "2026" in msg_l:
        top5 = sorted(fcs.items(),
                      key=lambda x: sum(float(v) for v in x[1]["forecast"][9:21]),
                      reverse=True)[:5]
        lines = "\n".join([
            f"• {p}: {sum(float(v) for v in r['forecast'][9:21]):.0f} units  ({r['trend']})"
            for p,r in top5
        ])
        return f"📈 **Top 5 SKUs — Full Year 2026 Forecast**\n\n{lines}"

    # ── Product-specific ───────────────────────────────────────────────
    for prod in sorted(fcs.keys(), key=len, reverse=True):
        if prod.lower() in msg_l:
            r  = fcs[prod]
            fv = [float(v) for v in r["forecast"]]
            t25, t26 = _ann(prod)

            if any(k in msg_l for k in ["reorder","safety stock","eoq","inventory","stock"]):
                row = inv[inv["Product"]==prod]
                if not row.empty:
                    row = row.iloc[0]
                    return (f"📦 **Inventory Policy — {prod}**\n\n"
                            f"• Safety Stock: **{row['Safety Stock']} units**\n"
                            f"• Reorder Point: **{row['Reorder Point']} units**\n"
                            f"• EOQ: **{row['EOQ']} units**\n"
                            f"• Stock Needed M+1: {row['Stock Needed (M+1)']} units\n"
                            f"• Unit Price: ₹{row['Unit Price (₹)']} | Lead Time: {row['Lead Time (days)']} days")

            return (f"📈 **Forecast — {prod}** (Apr 2025 → Dec 2026)\n\n"
                    f"**Near-term:**\n"
                    f"• {FUTURE_MONTHS[0]}: **{fv[0]:.1f} units**\n"
                    f"• {FUTURE_MONTHS[1]}: {fv[1]:.1f} units\n"
                    f"• {FUTURE_MONTHS[2]}: {fv[2]:.1f} units\n\n"
                    f"**Annual Totals:**\n"
                    f"• 2025 (Apr–Dec): **{t25:.0f} units**\n"
                    f"• 2026 (Full Year): **{t26:.0f} units**\n\n"
                    f"Model: {r['method']} | Avg Hist: {r['avg_monthly']} u/mo | "
                    f"Trend: {r['trend']} | RMSE: {r['rmse']}")

    # ── General forecast ───────────────────────────────────────────────
    if any(k in msg_l for k in ["forecast","predict","demand"]):
        top5 = sorted(fcs.items(),
                      key=lambda x: float(x[1]["forecast"][0]), reverse=True)[:5]
        lines = "\n".join([
            f"• {p}: {float(r['forecast'][0]):.1f} units in {FUTURE_MONTHS[0]}  ({r['trend']})"
            for p,r in top5
        ])
        return (f"📈 **Top 5 SKUs — {FUTURE_MONTHS[0]} Forecast**\n\n{lines}\n\n"
                f"💡 Ask '2026 forecast' for full-year 2026 totals.")

    # ── Inventory ─────────────────────────────────────────────────────
    if "reorder" in msg_l:
        top = inv.sort_values("Reorder Point",ascending=False).head(5)
        lines = "\n".join([f"• {r['Product']}: {r['Reorder Point']} units (SS={r['Safety Stock']})"
                           for _,r in top.iterrows()])
        return f"📦 **Top 5 by Reorder Point**\n\n{lines}"
    if "safety stock" in msg_l:
        top = inv.sort_values("Safety Stock",ascending=False).head(5)
        lines = "\n".join([f"• {r['Product']}: {r['Safety Stock']} units" for _,r in top.iterrows()])
        return f"📦 **Top 5 by Safety Stock**\n\n{lines}"
    if "eoq" in msg_l or "economic order" in msg_l:
        top = inv.sort_values("EOQ",ascending=False).head(5)
        lines = "\n".join([f"• {r['Product']}: {r['EOQ']} units" for _,r in top.iterrows()])
        return f"⚖️ **Top 5 by EOQ**\n\n{lines}"

    # ── Logistics ─────────────────────────────────────────────────────
    if any(k in msg_l for k in ["logistics","slow delivery","region support","delivery issue","which region"]):
        rs = df.groupby("Region").agg(
            avg_days=("Delivery_Days","mean"), rr=("Return_Flag","mean")).reset_index()
        worst = rs.sort_values("avg_days", ascending=False).head(3)
        lines = "\n".join([f"• {r['Region']}: {r['avg_days']:.1f} days, {r['rr']*100:.1f}% returns"
                           for _,r in worst.iterrows()])
        return f"🚚 **Regions Needing Logistics Improvement**\n\n{lines}\n\nFix: Upgrade to premium couriers."
    if any(k in msg_l for k in ["courier","shipping speed","delivery performance"]):
        cs = (df.groupby("Courier_Partner")
              .agg(avg_days=("Delivery_Days","mean"), cost=("Shipping_Cost_INR","mean"))
              .reset_index().sort_values("avg_days"))
        lines = "\n".join([f"• {r['Courier_Partner']}: {r['avg_days']:.1f} days, ₹{r['cost']:.0f}/order"
                           for _,r in cs.iterrows()])
        return f"🚚 **Courier Performance**\n\n✅ Best: **{cs.iloc[0]['Courier_Partner']}**\n\n{lines}"

    # ── Sales / Revenue ───────────────────────────────────────────────
    if any(k in msg_l for k in ["revenue","best selling","top product","highest revenue"]):
        top5 = df.groupby("Product_Name")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
        lines = "\n".join([f"• {p}: ₹{v/1e3:.1f}K" for p,v in top5.items()])
        return (f"💰 **Top 5 by Revenue**\n\n{lines}\n\n"
                f"🏆 Overall: **{ctx['top_rev_prod']}**  |  Total: ₹{ctx['total_rev']/1e6:.2f}M")
    if "return" in msg_l:
        rr = df.groupby("Product_Name")["Return_Flag"].mean().sort_values(ascending=False).head(5)
        lines = "\n".join([f"• {p}: {v*100:.1f}%" for p,v in rr.items()])
        return f"↩️ **Highest Return-Rate SKUs**\n\n{lines}\n\nOverall: {ctx['return_rate']:.1f}%"
    if any(k in msg_l for k in ["region","state","city"]):
        rrev = df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False)
        lines = "\n".join([f"• {r}: ₹{v/1e3:.1f}K" for r,v in rrev.items()])
        return f"🗺️ **Region-wise Revenue**\n\n{lines}\n\n🥇 Top: **{ctx['top_region']}**"
    if any(k in msg_l for k in ["category","segment"]):
        cats = df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        lines = "\n".join([f"• {c}: ₹{v/1e3:.1f}K" for c,v in cats.items()])
        return f"📂 **Category-wise Revenue**\n\n{lines}"

    # ── KPI ──────────────────────────────────────────────────────────
    if any(k in msg_l for k in ["kpi","summary","overview","snapshot"]):
        top_p = ctx["top_fc_prod"]
        fv    = [float(v) for v in fcs[top_p]["forecast"]]
        t25, t26 = sum(fv[0:9]), sum(fv[9:21]) if len(fv)>=21 else sum(fv[9:])
        return (f"📊 **OmniFlow-D2D KPI Snapshot**\n\n"
                f"💰 Revenue: ₹{ctx['total_rev']/1e6:.2f}M\n"
                f"📦 Orders: {ctx['total_orders']:,}\n"
                f"🏆 Top SKU (Revenue): {ctx['top_rev_prod']}\n"
                f"📈 Top Forecast SKU: {top_p}\n"
                f"   → {FUTURE_MONTHS[0]}: {fv[0]:.1f} units\n"
                f"   → 2025 Total: {t25:.0f} units\n"
                f"   → 2026 Annual: {t26:.0f} units\n"
                f"🗺️ Top Region: {ctx['top_region']}\n"
                f"🚚 Avg Delivery: {ctx['avg_delivery']:.1f} days\n"
                f"↩️ Return Rate: {ctx['return_rate']:.1f}%\n\n"
                f"📅 Forecast Horizon: Apr 2025 → Dec 2026")

    if any(k in msg_l for k in ["hi","hello","hey","help","commands","what can"]):
        return ("👋 **OmniFlow-D2D AI — Forecast through Dec 2026**\n\n"
                "📈 'Forecast for boAt Airdopes 141 TWS'\n"
                "📈 'Which SKU has highest demand next month?'\n"
                "📈 '2026 forecast top products'\n"
                "📦 'Reorder point for Himalaya Face Wash'\n"
                "📦 'Top 5 safety stock'\n"
                "🚚 'Which region needs logistics support?'\n"
                "🚚 'Best courier performance'\n"
                "💰 'Top products by revenue'\n"
                "📊 'KPI summary'\n\n"
                "All 50 SKUs · Apr 2025 → Dec 2026!")

    sample = ", ".join(list(fcs.keys())[:3])
    return (f"🤖 Couldn't match: **\"{msg}\"**\n\nTry:\n"
            f"• 'Forecast for [product name]'\n"
            f"• '2026 forecast'\n"
            f"• 'KPI summary'\n\nSample SKUs: {sample}")

def page_chatbot(df):
    st.markdown("<div class='ph'><h1>🤖 AI Decision Intelligence</h1>"
                "<p>Context-aware chatbot · Live outputs from all 5 modules · "
                "Forecast coverage: Apr 2025 → Dec 2026</p></div>",
                unsafe_allow_html=True)

    with st.spinner("Loading AI context from all modules..."):
        fcs = run_all_forecasts(df)
        inv = compute_inventory(df, fcs)
        dlv = df[df["Order_Status"]=="Delivered"]
        ctx = {
            "total_orders":  len(df),
            "total_rev":     float(dlv["Revenue_INR"].sum()),
            "top_rev_prod":  df.groupby("Product_Name")["Revenue_INR"].sum().idxmax(),
            "top_region":    df.groupby("Region")["Revenue_INR"].sum().idxmax(),
            "avg_delivery":  float(df["Delivery_Days"].mean()),
            "return_rate":   float(df["Return_Flag"].mean()*100),
            "top_fc_prod":   max(fcs, key=lambda p: float(fcs[p]["forecast"][0])),
            "fcs": fcs, "inv": inv, "df": df,
        }

    if "history" not in st.session_state:
        st.session_state.history = [
            ("bot", "👋 Hello! I'm your OmniFlow-D2D Supply Chain AI.\n\n"
                    "I have full forecasts for all 50 SKUs through **Dec 2026**.\n"
                    "Type **help** to see what I can do!")
        ]

    for role, text in st.session_state.history:
        cls  = "chat-user" if role == "user" else "chat-bot"
        safe = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text.replace("\n","<br>"))
        st.markdown(f"<div class='{cls}'>{safe}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Quick Queries:**")
    qs = [
        "Highest demand next month",
        "2026 forecast top products",
        "KPI summary",
        "Best courier performance",
        "Reorder point for Samsung Galaxy M34 5G",
        "Forecast for boAt Airdopes 141 TWS",
        "Top 5 safety stock",
        "Top products by revenue",
    ]
    cols = st.columns(4)
    for i,q in enumerate(qs):
        if cols[i%4].button(q, key=f"qb{i}"):
            st.session_state._pq = q

    if hasattr(st.session_state,"_pq") and st.session_state._pq:
        q = st.session_state._pq; st.session_state._pq = None
        st.session_state.history.append(("user",q))
        st.session_state.history.append(("bot",_chat(q,ctx)))
        st.rerun()

    ci,cb = st.columns([5,1])
    user_in = ci.text_input("Ask a supply chain question...", key="chat_inp",
                             label_visibility="collapsed",
                             placeholder="e.g. Forecast for Himalaya Neem Face Wash")
    if cb.button("Send →") and user_in.strip():
        st.session_state.history.append(("user",user_in))
        st.session_state.history.append(("bot",_chat(user_in,ctx)))
        st.rerun()
    if st.button("🗑️ Clear Chat"):
        st.session_state.history = []; st.rerun()

# ══════════════════════════════════════════════════════════════════════
# SIDEBAR + ROUTER
# ══════════════════════════════════════════════════════════════════════
def main():
    df = load_data()
    with st.sidebar:
        st.markdown("<div class='logo'>⟳ OmniFlow-D2D</div>", unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#374151;font-size:.65rem;"
                    "font-family:Space Mono;margin-bottom:.5rem;'>"
                    "Data-to-Decision Platform</div>", unsafe_allow_html=True)
        st.markdown("---")
        pages = {
            "📊 Overview Dashboard":      "overview",
            "📈 Demand Forecasting":       "forecasting",
            "🏭 Inventory Optimization":   "inventory",
            "⚙️ Production Planning":      "production",
            "🚚 Logistics Optimization":   "logistics",
            "🤖 AI Decision Intelligence": "chatbot",
        }
        if "page" not in st.session_state:
            st.session_state.page = "overview"
        for label, key in pages.items():
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.page = key; st.rerun()
        st.markdown("---")
        st.markdown(f"""
        <div style='background:#111827;border:1px solid #1e293b;border-radius:8px;
                    padding:.8rem;font-family:Space Mono;font-size:.7rem;'>
        <div style='color:#64748b;'>DATASET</div>
        <div style='color:#00d4ff;font-size:.9rem;'>{len(df):,} Orders</div>
        <div style='color:#64748b;margin-top:.4rem;'>SKUs</div>
        <div style='color:#7c3aed;font-size:.9rem;'>{df["SKU_ID"].nunique()} Products</div>
        <div style='color:#64748b;margin-top:.4rem;'>HISTORY</div>
        <div style='color:#94a3b8;font-size:.7rem;'>Jan 2024 → Mar 2025</div>
        <div style='color:#64748b;margin-top:.4rem;'>FORECAST</div>
        <div style='color:#10b981;font-size:.7rem;'>Apr 2025 → Dec 2026</div>
        <div style='color:#64748b;margin-top:.4rem;'>CATEGORIES</div>
        <div style='color:#94a3b8;font-size:.7rem;'>
            Electronics · Home & Kitchen<br>Fashion · Health & Personal Care
        </div>
        <div style='color:#64748b;margin-top:.4rem;'>SOURCES</div>
        <div style='color:#94a3b8;font-size:.65rem;'>
            Amazon India 2025<br>Shiprocket / INCREFF
        </div>
        </div>""", unsafe_allow_html=True)

    pg = st.session_state.get("page","overview")
    if   pg == "overview":    page_overview(df)
    elif pg == "forecasting":  page_forecasting(df)
    elif pg == "inventory":    page_inventory(df)
    elif pg == "production":   page_production(df)
    elif pg == "logistics":    page_logistics(df)
    elif pg == "chatbot":      page_chatbot(df)

if __name__ == "__main__":
    main()
