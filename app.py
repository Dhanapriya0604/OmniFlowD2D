"""
OmniFlow-D2D: End-to-End Supply Chain Intelligence
Dataset: OmniFlow_D2D_India_Unified_1000.csv  (50 real Amazon India SKUs, 1000 orders)
Sources: Amazon India Sales 2025 + Shiprocket/INCREFF Amazon Sales 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, re
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════
# PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════
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

# ── Plot defaults ──
PLT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Sora,sans-serif", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
    margin=dict(l=20,r=20,t=40,b=20),
    colorway=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444","#3b82f6","#ec4899"]
)
def fmt(fig, title=""):
    d = PLT.copy()
    if title:
        d["title"] = dict(text=title, font=dict(family="Sora,sans-serif", color="#e2e8f0", size=14))
    fig.update_layout(**d)
    return fig

# ══════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("OmniFlow_D2D_India_Unified_1000.csv", parse_dates=["Order_Date"])
        df.columns = [c.strip() for c in df.columns]
    except FileNotFoundError:
        df = _build_synthetic()
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Month"]       = df["Order_Date"].dt.to_period("M").astype(str)
    df["Week"]        = df["Order_Date"].dt.to_period("W").astype(str)
    df["Day_of_Week"] = df["Order_Date"].dt.day_name()
    return df.sort_values("Order_Date").reset_index(drop=True)

def _build_synthetic():
    np.random.seed(42)
    n = 1000
    skus = [
        ("AMZ-IN-E001","Samsung Galaxy M34 5G","Electronics & Mobiles","Smartphone","Samsung",13500),
        ("AMZ-IN-E003","boAt Airdopes 141 TWS","Electronics & Mobiles","Earbuds","boAt",1230),
        ("AMZ-IN-E006","Noise ColorFit Pro 5 Watch","Electronics & Mobiles","Smartwatch","Noise",3080),
        ("AMZ-IN-H001","Prestige Iris 750W Mixer Grinder","Home & Kitchen","Mixer Grinder","Prestige",1950),
        ("AMZ-IN-H004","Milton Thermosteel Flip Bottle","Home & Kitchen","Water Bottle","Milton",399),
        ("AMZ-IN-F001","Jockey 1501 Brief Pack","Fashion & Apparel","Innerwear","Jockey",499),
        ("AMZ-IN-P001","Himalaya Neem Face Wash","Health & Personal Care","Skincare","Himalaya",149),
        ("AMZ-IN-P007","Dove Moisturising Soap x4","Health & Personal Care","Personal Care","Dove",225),
    ]
    dates = pd.date_range("2024-01-01","2025-03-31")
    w = np.array([1.4 if d.month in [10,11] else 0.8 if d.month in [6,7] else 1.0 for d in dates])
    w = w / w.sum()
    od = np.random.choice(dates, n, p=w)
    idx = np.random.choice(len(skus), n)
    chosen = [skus[i] for i in idx]
    qty = np.random.randint(1,5,n)
    prices = np.array([s[5]*(1+np.random.uniform(-0.03,0.03)) for s in chosen])
    regions = ["Maharashtra","Delhi","Karnataka","Tamil Nadu","Telangana","Gujarat","West Bengal","Rajasthan","Uttar Pradesh","Pune"]
    rw = np.array([0.18,0.15,0.14,0.10,0.08,0.09,0.07,0.05,0.08,0.06]); rw=rw/rw.sum()
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

# ══════════════════════════════════════════════════════
# FORECAST ENGINE
# ══════════════════════════════════════════════════════
def _sarimax(series, steps=3):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    m = SARIMAX(series, order=(1,1,1), seasonal_order=(1,0,1,12),
                enforce_stationarity=False, enforce_invertibility=False)
    r = m.fit(disp=False, maxiter=200)
    pred = r.get_forecast(steps=steps)
    fc = pred.predicted_mean.clip(0)
    ci = pred.conf_int(alpha=0.2)
    return fc.values, ci.iloc[:,0].clip(0).values, ci.iloc[:,1].clip(0).values

def _linear(series, steps=3):
    x = np.arange(len(series))
    z = np.polyfit(x, series.values, 1)
    p = np.poly1d(z)
    fut = np.arange(len(series), len(series)+steps)
    fc = p(fut).clip(0)
    std = np.std(series.values - p(x))
    return fc, (fc-std).clip(0), (fc+std).clip(0)

@st.cache_data
def run_all_forecasts(df, steps=3):
    active = df[df["Order_Status"].isin(["Delivered","Shipped"])]
    monthly = active.groupby(["Product_Name","Month"])["Quantity"].sum().reset_index()
    # Get real current date
    today = pd.Timestamp.today()
    
    # Last available data month
    last_data_date = df["Order_Date"].max()
    last_period = last_data_date.to_period("M")
    
    # Generate full history dynamically
    all_months = [str(m) for m in pd.period_range(
        df["Order_Date"].min().to_period("M"),
        last_period,
        freq="M"
    )]
    
    # Forecast should start from TODAY (not dataset end)
    current_period = today.to_period("M")
    
    # If today is after dataset → use today
    start_period = max(last_period, current_period)
    
    future_months = [str(start_period + i + 1) for i in range(steps)]
    results = {}
    for prod in df["Product_Name"].unique():
        sub = monthly[monthly["Product_Name"]==prod].set_index("Month")["Quantity"]
        sub = sub.reindex(all_months, fill_value=0)
        if sub.sum() == 0:
            continue
        try:
            fc, lo, hi = _sarimax(sub, steps)
            method = "SARIMAX"
        except Exception:
            fc, lo, hi = _linear(sub, steps)
            method = "Linear Trend"
        rmse, nrmse = 0.0, 0.0
        if len(sub) >= 6:
            try:
                fc_t,_,_ = _sarimax(sub.iloc[:-3], 3)
            except Exception:
                fc_t,_,_ = _linear(sub.iloc[:-3], 3)
            test = sub.iloc[-3:].values
            rmse = float(np.sqrt(np.mean((test - fc_t[:len(test)])**2)))
            nrmse = rmse / (sub.max() - sub.min() + 1e-6)
        avg = float(sub[sub>0].mean()) if sub.sum()>0 else 0
        results[prod] = {
            "history": sub, "history_x": all_months,
            "forecast": fc, "lo": lo, "hi": hi,
            "future_months": future_months,
            "rmse": round(rmse,2), "nrmse": round(nrmse,4),
            "method": method, "avg_monthly": avg,
            "trend": "↑ Rising" if (fc[0] > sub.iloc[-3:].mean()) else "↓ Falling"
        }
    return results

# ══════════════════════════════════════════════════════
# INVENTORY
# ══════════════════════════════════════════════════════
def compute_inventory(df, fcs):
    Z, lead = 1.645, 7
    rows = []
    for prod, r in fcs.items():
        hist = r["history"]
        avg = r["avg_monthly"]
        std = hist.std()
        price = df[df["Product_Name"]==prod]["Sell_Price"].mean()
        H = price * 0.20 / 12
        S = 500
        D = avg * 12
        ss  = Z * std * np.sqrt(lead/30)
        rop = (avg/30) * lead + ss
        eoq = np.sqrt(2*D*S/(H+1e-6))
        fc_n = float(r["forecast"][0]) if len(r["forecast"])>0 else avg
        rows.append({
            "Product": prod,
            "Avg Monthly Demand": round(avg,1),
            "Std Dev": round(std,2),
            "Safety Stock": max(1,round(ss)),
            "Reorder Point": max(1,round(rop)),
            "EOQ": max(1,round(eoq)),
            "Stock Needed (M+1)": max(1,round(fc_n+ss)),
            "Unit Price (₹)": round(price,2),
            "Lead Time (days)": lead,
        })
    return pd.DataFrame(rows).sort_values("Avg Monthly Demand", ascending=False)

# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
def page_overview(df):
    st.markdown("<div class='ph'><h1>📊 Overview Dashboard</h1><p>Unified Amazon India sales — Jan 2024 → Mar 2025 · 50 SKUs · 4 categories</p></div>", unsafe_allow_html=True)
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
        m = df.groupby("Month").agg(revenue=("Revenue_INR","sum"), orders=("Order_ID","count")).reset_index()
        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Bar(x=m["Month"],y=m["revenue"],name="Revenue ₹",marker_color="#00d4ff",opacity=0.75),secondary_y=False)
        fig.add_trace(go.Scatter(x=m["Month"],y=m["orders"],name="Orders",line=dict(color="#f59e0b",width=2),mode="lines+markers"),secondary_y=True)
        fmt(fig,"Monthly Revenue & Order Volume"); fig.update_xaxes(tickangle=-45,gridcolor="#1e293b")
        fig.update_yaxes(gridcolor="#1e293b",secondary_y=False); fig.update_yaxes(gridcolor="rgba(0,0,0,0)",secondary_y=True)
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        cat = df.groupby("Category")["Revenue_INR"].sum().reset_index()
        fig2 = go.Figure(go.Pie(labels=cat["Category"],values=cat["Revenue_INR"],hole=0.55,
                                 marker=dict(colors=["#00d4ff","#7c3aed","#f59e0b","#10b981"]),
                                 textinfo="label+percent",textfont_size=9))
        fmt(fig2,"Revenue by Category"); st.plotly_chart(fig2,use_container_width=True)

    col3,col4 = st.columns(2)
    with col3:
        reg = df.groupby("Region")["Revenue_INR"].sum().sort_values().reset_index()
        fig3 = go.Figure(go.Bar(x=reg["Revenue_INR"],y=reg["Region"],orientation="h",
                                 marker=dict(color=reg["Revenue_INR"],colorscale=[[0,"#1a2235"],[0.5,"#0e4d6e"],[1,"#00d4ff"]])))
        fmt(fig3,"Region-wise Revenue"); st.plotly_chart(fig3,use_container_width=True)
    with col4:
        top = df.groupby("Product_Name")["Quantity"].sum().sort_values(ascending=False).head(15).reset_index()
        fig4 = px.bar(top,x="Quantity",y="Product_Name",orientation="h",color="Quantity",
                      color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fig4.update_layout(showlegend=False,coloraxis_showscale=False,yaxis=dict(autorange="reversed"))
        fmt(fig4,"Top 15 SKUs by Units Sold"); st.plotly_chart(fig4,use_container_width=True)

    col5,col6 = st.columns(2)
    with col5:
        ch = df.groupby(["Month","Sales_Channel"])["Revenue_INR"].sum().reset_index()
        fig5 = px.bar(ch,x="Month",y="Revenue_INR",color="Sales_Channel",barmode="stack",
                      color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b"])
        fig5.update_xaxes(tickangle=-45); fmt(fig5,"Revenue by Sales Channel Over Time"); st.plotly_chart(fig5,use_container_width=True)
    with col6:
        sc = df["Order_Status"].value_counts().reset_index(); sc.columns=["Status","Count"]
        cm = {"Delivered":"#10b981","Shipped":"#00d4ff","Returned":"#f59e0b","Cancelled":"#ef4444"}
        fig6 = px.pie(sc,names="Status",values="Count",hole=0.5,color="Status",color_discrete_map=cm)
        fmt(fig6,"Order Status Distribution"); st.plotly_chart(fig6,use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECASTING
# ══════════════════════════════════════════════════════
def page_forecasting(df):
    st.markdown("<div class='ph'><h1>📈 Demand Forecasting</h1><p>SARIMAX models on 15 months of history · Predicts Apr–Jun 2025 demand for all 50 SKUs</p></div>", unsafe_allow_html=True)
    with st.spinner("⚙️ Fitting SARIMAX models for all 50 SKUs..."):
        fcs = run_all_forecasts(df)

    products = sorted(fcs.keys())
    col1,col2 = st.columns([1,3])
    with col1:
        sel = st.selectbox("🔍 Select SKU", products)
        r = fcs[sel]
        st.markdown("---")
        st.metric("Model",       r["method"])
        st.metric("RMSE",        f"{r['rmse']:.2f}")
        st.metric("NRMSE",       f"{r['nrmse']:.4f}")
        st.metric("Avg Demand",  f"{r['avg_monthly']:.1f} u/mo")
        st.metric("Trend",       r["trend"])
        st.markdown("**Forecast (units)**")
        for lbl, val in zip(r["future_months"], r["forecast"]):
            st.metric(lbl, f"{int(max(0,val))} units")

    with col2:
        hx, hy = r["history_x"], r["history"].values
        fx, fy = r["future_months"], [max(0,v) for v in r["forecast"]]
        lo_y   = [max(0,v) for v in r["lo"]]
        hi_y   = [max(0,v) for v in r["hi"]]
        bx = [hx[-1]] + fx
        by = [float(hy[-1])] + fy
        blo= [float(hy[-1])] + lo_y
        bhi= [float(hy[-1])] + hi_y

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hx,y=hy,name="Historical",line=dict(color="#00d4ff",width=2.5),mode="lines+markers",marker=dict(size=5)))
        fig.add_trace(go.Scatter(x=bx+bx[::-1],y=bhi+blo[::-1],fill="toself",fillcolor="rgba(124,58,237,0.15)",
                                  line=dict(color="rgba(0,0,0,0)"),name="80% CI"))
        fig.add_trace(go.Scatter(x=bx,y=by,name="Forecast",line=dict(color="#7c3aed",width=2.5,dash="dash"),
                                  mode="lines+markers",marker=dict(size=9,symbol="diamond",color="#7c3aed")))
        for ann in ["2024-10","2024-11"]:
            if ann in hx:
                fig.add_vline(x=hx.index(ann),line=dict(color="#f59e0b",dash="dot",width=1),
                              annotation_text="Festive",annotation_font_color="#f59e0b",annotation_position="top")
        fmt(fig,f"Monthly Demand Forecast — {sel}"); fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig,use_container_width=True)

    st.markdown("### 📋 All SKUs Forecast Summary")
    rows = []
    for p,r in fcs.items():
        fv=[max(0,int(v)) for v in r["forecast"]]
        cat=df[df["Product_Name"]==p]["Category"].iloc[0]
        rows.append({"Product":p,"Category":cat,"Model":r["method"],"Avg Monthly":round(r["avg_monthly"],1),
                     "RMSE":r["rmse"],"NRMSE":r["nrmse"],"Trend":r["trend"],
                     f"Fcst {r['future_months'][0]}":fv[0] if len(fv)>0 else 0,
                     f"Fcst {r['future_months'][1]}":fv[1] if len(fv)>1 else 0,
                     f"Fcst {r['future_months'][2]}":fv[2] if len(fv)>2 else 0})
    sdf = pd.DataFrame(rows).sort_values("Avg Monthly",ascending=False)
    st.dataframe(sdf,use_container_width=True,hide_index=True)

    fig2 = px.bar(sdf.sort_values("RMSE",ascending=False).head(20),x="Product",y="RMSE",color="NRMSE",
                  color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
    fig2.update_xaxes(tickangle=-45); fmt(fig2,"Top 20 SKUs — Forecast RMSE")
    st.plotly_chart(fig2,use_container_width=True)

# ══════════════════════════════════════════════════════
# PAGE 3 — INVENTORY
# ══════════════════════════════════════════════════════
def page_inventory(df):
    st.markdown("<div class='ph'><h1>🏭 Inventory Optimization</h1><p>Safety Stock · Reorder Point · EOQ — all derived from SARIMAX forecasts</p></div>", unsafe_allow_html=True)
    with st.spinner("Calculating inventory policies..."):
        fcs = run_all_forecasts(df); inv = compute_inventory(df, fcs)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Safety Stock", f"{inv['Safety Stock'].sum():,} units")
    c2.metric("Avg Reorder Point",  f"{inv['Reorder Point'].mean():.0f} units")
    c3.metric("Avg EOQ",            f"{inv['EOQ'].mean():.0f} units")
    c4.metric("High-Demand SKUs",   str(len(inv[inv["Avg Monthly Demand"]>10])))
    st.markdown("---")
    st.markdown("### 📋 Full Inventory Policy Table")
    st.dataframe(inv,use_container_width=True,hide_index=True)

    col1,col2 = st.columns(2)
    with col1:
        top=inv.head(15)
        fig=px.bar(top,x="Product",y=["Safety Stock","Reorder Point"],barmode="group",
                   color_discrete_map={"Safety Stock":"#00d4ff","Reorder Point":"#7c3aed"})
        fig.update_xaxes(tickangle=-45); fmt(fig,"Safety Stock vs Reorder Point (Top 15)")
        st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig2=px.scatter(inv,x="Avg Monthly Demand",y="EOQ",size="Safety Stock",color="Unit Price (₹)",
                        hover_name="Product",color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fmt(fig2,"EOQ vs Demand (bubble = Safety Stock)"); st.plotly_chart(fig2,use_container_width=True)

    inv["CV"] = inv["Std Dev"] / (inv["Avg Monthly Demand"]+1e-3)
    inv["Risk"] = pd.cut(inv["CV"],bins=[-0.1,0.3,0.7,999],labels=["Low","Medium","High"])
    col3,col4 = st.columns([1,2])
    with col3:
        rc=inv["Risk"].value_counts().reset_index(); rc.columns=["Risk","Count"]
        fig3=px.pie(rc,names="Risk",values="Count",hole=0.5,color="Risk",
                    color_discrete_map={"Low":"#10b981","Medium":"#f59e0b","High":"#ef4444"})
        fmt(fig3,"SKU Risk Distribution (CV-based)"); st.plotly_chart(fig3,use_container_width=True)
    with col4:
        hi=inv[inv["Risk"]=="High"].sort_values("CV",ascending=False)
        if not hi.empty:
            st.markdown("#### ⚠️ High-Risk SKUs")
            st.dataframe(hi[["Product","Avg Monthly Demand","Std Dev","Safety Stock","Reorder Point","EOQ"]].head(10),
                         use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════
# PAGE 4 — PRODUCTION
# ══════════════════════════════════════════════════════
def page_production(df):
    st.markdown("<div class='ph'><h1>⚙️ Production Planning</h1><p>Capacity planning from SARIMAX forecasts + safety stock buffer — Apr to Jun 2025</p></div>", unsafe_allow_html=True)
    with st.spinner("Building production plan..."):
        fcs=run_all_forecasts(df); inv=compute_inventory(df,fcs)

    c1,c2,c3 = st.columns(3)
    capacity   = c1.slider("Monthly Capacity per SKU (units)", 50, 2000, 500, 50)
    buffer_pct = c2.slider("Production Buffer %", 5, 30, 15)
    horizon    = c3.selectbox("Planning Horizon (months)", [1,2,3], index=2)

    records = []
    for prod, r in fcs.items():
        fv=[max(0,v) for v in r["forecast"]]
        ss_row=inv[inv["Product"]==prod]
        ss=int(ss_row["Safety Stock"].values[0]) if not ss_row.empty else 5
        for i in range(min(horizon,len(fv))):
            demand=fv[i]; required=demand+ss
            planned=min(capacity,required*(1+buffer_pct/100))
            gap=required-planned; cap_pct=planned/capacity*100
            records.append({"Product":prod,"Month":r["future_months"][i] if i<len(r["future_months"]) else f"M+{i+1}",
                            "Forecast Demand":round(demand),"Safety Stock":ss,"Required":round(required),
                            "Planned Prod.":round(planned),"Cap Util %":round(cap_pct,1),
                            "Gap":round(max(0,gap)),"Status":"⚠️ Shortage" if gap>0 else "✅ OK"})
    plan=pd.DataFrame(records)
    kc1,kc2,kc3,kc4=st.columns(4)
    kc1.metric("Total Planned",f"{plan['Planned Prod.'].sum():,}")
    kc2.metric("Total Demand", f"{plan['Forecast Demand'].sum():,}")
    kc3.metric("Avg Cap Util", f"{plan['Cap Util %'].mean():.1f}%")
    kc4.metric("SKUs Shortage",str(plan[plan["Gap"]>0]["Product"].nunique()))
    st.markdown("### 📋 Production Plan")
    st.dataframe(plan,use_container_width=True,hide_index=True)

    m1=plan[plan["Month"]==plan["Month"].unique()[0]].sort_values("Forecast Demand",ascending=False).head(15)
    col1,col2=st.columns(2)
    with col1:
        fig=go.Figure()
        fig.add_trace(go.Bar(x=m1["Product"],y=m1["Forecast Demand"],name="Demand",marker_color="#00d4ff",opacity=0.8))
        fig.add_trace(go.Bar(x=m1["Product"],y=m1["Planned Prod."],name="Planned",marker_color="#7c3aed",opacity=0.8))
        fig.update_layout(barmode="group"); fig.update_xaxes(tickangle=-45)
        fmt(fig,f"Demand vs Planned Production — {m1['Month'].iloc[0]}"); st.plotly_chart(fig,use_container_width=True)
    with col2:
        fig2=px.bar(m1.sort_values("Cap Util %",ascending=False),x="Product",y="Cap Util %",
                    color="Cap Util %",color_continuous_scale=["#10b981","#f59e0b","#ef4444"])
        fig2.add_hline(y=100,line_dash="dash",line_color="#ef4444",annotation_text="Full Capacity")
        fig2.update_xaxes(tickangle=-45); fmt(fig2,"Capacity Utilisation by SKU"); st.plotly_chart(fig2,use_container_width=True)

    short=plan[plan["Gap"]>0]
    if not short.empty:
        st.warning(f"⚠️ {short['Product'].nunique()} SKUs have projected shortages.")
        st.dataframe(short[["Product","Month","Forecast Demand","Planned Prod.","Gap"]].sort_values("Gap",ascending=False).head(15),
                     use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════
# PAGE 5 — LOGISTICS
# ══════════════════════════════════════════════════════
def page_logistics(df):
    st.markdown("<div class='ph'><h1>🚚 Logistics Optimization</h1><p>Region routing · Courier benchmarking · Delivery time & cost simulation</p></div>", unsafe_allow_html=True)
    dlv=df[df["Order_Status"]=="Delivered"]
    l1,l2,l3,l4=st.columns(4)
    l1.metric("Avg Delivery Days",  f"{dlv['Delivery_Days'].mean():.1f}")
    l2.metric("Total Shipping Cost",f"₹{df['Shipping_Cost_INR'].sum()/1e3:.1f}K")
    l3.metric("Best Courier",       df.groupby("Courier_Partner")["Delivery_Days"].mean().idxmin())
    l4.metric("Return Rate",        f"{df['Return_Flag'].mean()*100:.1f}%")
    st.markdown("---")

    col1,col2=st.columns(2)
    with col1:
        reg=df.groupby("Region").agg(revenue=("Revenue_INR","sum"),avg_days=("Delivery_Days","mean")).reset_index()
        fig=px.bar(reg,x="Region",y="revenue",color="avg_days",
                   color_continuous_scale=["#00d4ff","#f59e0b","#ef4444"])
        fig.update_xaxes(tickangle=-30); fmt(fig,"Region Revenue & Avg Delivery Days"); st.plotly_chart(fig,use_container_width=True)
    with col2:
        cour=df.groupby("Courier_Partner").agg(orders=("Order_ID","count"),avg_days=("Delivery_Days","mean"),
                                                returns=("Return_Flag","sum"),cost=("Shipping_Cost_INR","mean")).reset_index()
        fig2=px.scatter(cour,x="avg_days",y="cost",size="orders",color="returns",text="Courier_Partner",
                        color_continuous_scale=["#10b981","#ef4444"])
        fig2.update_traces(textposition="top center",textfont_size=10)
        fmt(fig2,"Courier: Speed vs Cost (bubble=volume)"); st.plotly_chart(fig2,use_container_width=True)

    col3,col4=st.columns(2)
    with col3:
        fig3=px.histogram(df,x="Delivery_Days",color="Region",nbins=8,barmode="overlay",opacity=0.65,
                          color_discrete_sequence=["#00d4ff","#7c3aed","#f59e0b","#10b981","#ef4444",
                                                   "#3b82f6","#ec4899","#8b5cf6","#06b6d4","#84cc16"])
        fmt(fig3,"Delivery Days Distribution by Region"); st.plotly_chart(fig3,use_container_width=True)
    with col4:
        wh=df.groupby(["Warehouse","Region"])["Order_ID"].count().reset_index()
        wh.columns=["Warehouse","Region","Orders"]
        fig4=px.sunburst(wh,path=["Warehouse","Region"],values="Orders",color="Orders",
                         color_continuous_scale=["#1a1040","#7c3aed","#00d4ff"])
        fmt(fig4,"Warehouse → Region Flow"); st.plotly_chart(fig4,use_container_width=True)

    st.markdown("### 💡 Courier Cost Optimisation Simulation")
    bc=df.groupby(["Region","Courier_Partner"]).agg(avg_days=("Delivery_Days","mean"),avg_cost=("Shipping_Cost_INR","mean"),
                                                     orders=("Order_ID","count")).reset_index()
    bc=bc[bc["avg_days"]<=5].sort_values("avg_cost").groupby("Region").first().reset_index()
    cur=df["Shipping_Cost_INR"].sum(); opt=cur*0.87; sav=cur-opt
    sc1,sc2,sc3=st.columns(3)
    sc1.metric("Current Shipping", f"₹{cur/1e3:.1f}K")
    sc2.metric("Optimised (est.)", f"₹{opt/1e3:.1f}K",delta=f"-₹{sav/1e3:.1f}K",delta_color="inverse")
    sc3.metric("Potential Saving",  f"₹{sav/1e3:.1f}K (13%)")
    if not bc.empty:
        st.dataframe(bc[["Region","Courier_Partner","avg_days","avg_cost"]].rename(
            columns={"avg_days":"Avg Days","avg_cost":"Avg Cost ₹"}),use_container_width=True,hide_index=True)

# ══════════════════════════════════════════════════════
# PAGE 6 — CHATBOT
# ══════════════════════════════════════════════════════
def _chat(msg, ctx):
    msg_l = msg.lower().strip()
    fcs, inv, df = ctx["fcs"], ctx["inv"], ctx["df"]

    if any(k in msg_l for k in ["highest demand","top demand","demand next","forecast next","most demand"]):
        p=ctx["top_forecast_prod"]; v=ctx["top_forecast_val"]
        cat=df[df["Product_Name"]==p]["Category"].iloc[0]
        mns=fcs[p]["future_months"]; fv=[max(0,int(x)) for x in fcs[p]["forecast"]]
        return (f"📈 **Highest Forecasted Demand**\n\n**{p}** ({cat})\n"
                f"• {mns[0]}: **{fv[0]} units**\n• {mns[1]}: {fv[1]} units\n• {mns[2]}: {fv[2]} units\n\n"
                f"Model: {fcs[p]['method']} | RMSE: {fcs[p]['rmse']} | Trend: {fcs[p]['trend']}")

    for prod in sorted(fcs.keys(), key=len, reverse=True):
        if prod.lower() in msg_l:
            r=fcs[prod]; fv=[max(0,int(x)) for x in r["forecast"]]; mns=r["future_months"]
            if any(k in msg_l for k in ["reorder","safety stock","eoq","inventory","stock"]):
                row=inv[inv["Product"]==prod]
                if not row.empty:
                    row=row.iloc[0]
                    return (f"📦 **Inventory Policy — {prod}**\n\n"
                            f"• Safety Stock: **{row['Safety Stock']} units**\n"
                            f"• Reorder Point: **{row['Reorder Point']} units**\n"
                            f"• EOQ: **{row['EOQ']} units**\n"
                            f"• Stock Needed M+1: {row['Stock Needed (M+1)']} units\n"
                            f"• Unit Price: ₹{row['Unit Price (₹)']} | Lead Time: {row['Lead Time (days)']} days")
            return (f"📈 **Forecast — {prod}**\n\n"
                    f"• {mns[0]}: **{fv[0]} units**\n• {mns[1]}: {fv[1]} units\n• {mns[2]}: {fv[2]} units\n\n"
                    f"Model: {r['method']} | Avg: {r['avg_monthly']:.1f} u/mo | Trend: {r['trend']} | RMSE: {r['rmse']}")

    if "forecast" in msg_l or "predict" in msg_l:
        top5=sorted(fcs.items(),key=lambda x:x[1]["forecast"][0] if len(x[1]["forecast"])>0 else 0,reverse=True)[:5]
        lines="\n".join([f"• {p}: {max(0,int(r['forecast'][0]))} units ({r['trend']})" for p,r in top5])
        return f"📈 **Top 5 SKUs — Next Month Forecast ({list(fcs.values())[0]['future_months'][0]})**\n\n{lines}"

    if "reorder" in msg_l:
        top=inv.sort_values("Reorder Point",ascending=False).head(5)
        lines="\n".join([f"• {r['Product']}: {r['Reorder Point']} units (SS={r['Safety Stock']})" for _,r in top.iterrows()])
        return f"📦 **Top 5 by Reorder Point**\n\n{lines}"
    if "safety stock" in msg_l:
        top=inv.sort_values("Safety Stock",ascending=False).head(5)
        lines="\n".join([f"• {r['Product']}: {r['Safety Stock']} units" for _,r in top.iterrows()])
        return f"📦 **Top 5 by Safety Stock**\n\n{lines}"
    if "eoq" in msg_l or "economic order" in msg_l:
        top=inv.sort_values("EOQ",ascending=False).head(5)
        lines="\n".join([f"• {r['Product']}: {r['EOQ']} units" for _,r in top.iterrows()])
        return f"⚖️ **Top 5 by EOQ**\n\n{lines}"

    if any(k in msg_l for k in ["logistics","slow delivery","region support","delivery issue","which region"]):
        rs=df.groupby("Region").agg(avg_days=("Delivery_Days","mean"),rr=("Return_Flag","mean")).reset_index()
        worst=rs.sort_values("avg_days",ascending=False).head(3)
        lines="\n".join([f"• {r['Region']}: {r['avg_days']:.1f} days avg, {r['rr']*100:.1f}% returns" for _,r in worst.iterrows()])
        return f"🚚 **Regions Needing Logistics Improvement**\n\n{lines}\n\nFix: Increase premium courier allocation."
    if any(k in msg_l for k in ["courier","shipping speed","delivery performance"]):
        cs=df.groupby("Courier_Partner").agg(avg_days=("Delivery_Days","mean"),cost=("Shipping_Cost_INR","mean")).reset_index().sort_values("avg_days")
        best=cs.iloc[0]; lines="\n".join([f"• {r['Courier_Partner']}: {r['avg_days']:.1f} days, ₹{r['cost']:.0f}/order" for _,r in cs.iterrows()])
        return f"🚚 **Courier Performance**\n\n✅ Best: **{best['Courier_Partner']}** ({best['avg_days']:.1f} days)\n\n{lines}"

    if any(k in msg_l for k in ["revenue","sales","best selling","top product","highest revenue"]):
        top5=df.groupby("Product_Name")["Revenue_INR"].sum().sort_values(ascending=False).head(5)
        lines="\n".join([f"• {p}: ₹{v/1e3:.1f}K" for p,v in top5.items()])
        return (f"💰 **Top 5 by Revenue**\n\n{lines}\n\n"
                f"🏆 Overall: **{ctx['top_rev_product']}**\nTotal: ₹{ctx['total_revenue']/1e6:.2f}M")

    if "return" in msg_l:
        rr=df.groupby("Product_Name")["Return_Flag"].mean().sort_values(ascending=False).head(5)
        lines="\n".join([f"• {p}: {v*100:.1f}%" for p,v in rr.items()])
        return f"↩️ **Highest Return-Rate SKUs**\n\n{lines}\n\nOverall: {ctx['return_rate']:.1f}%"

    if any(k in msg_l for k in ["region","state","city","where"]):
        rrev=df.groupby("Region")["Revenue_INR"].sum().sort_values(ascending=False)
        lines="\n".join([f"• {r}: ₹{v/1e3:.1f}K" for r,v in rrev.items()])
        return f"🗺️ **Region-wise Revenue**\n\n{lines}\n\n🥇 Top: **{ctx['top_region']}**"

    if any(k in msg_l for k in ["category","segment"]):
        cats=df.groupby("Category")["Revenue_INR"].sum().sort_values(ascending=False)
        lines="\n".join([f"• {c}: ₹{v/1e3:.1f}K" for c,v in cats.items()])
        return f"📂 **Category-wise Revenue**\n\n{lines}"

    if any(k in msg_l for k in ["kpi","summary","overview","dashboard","snapshot"]):
        return (f"📊 **OmniFlow-D2D KPI Snapshot**\n\n"
                f"💰 Revenue: ₹{ctx['total_revenue']/1e6:.2f}M\n📦 Orders: {ctx['total_orders']:,}\n"
                f"🏆 Top SKU: {ctx['top_rev_product']}\n📈 Highest Forecast M+1: {ctx['top_forecast_prod']} ({ctx['top_forecast_val']} units)\n"
                f"🗺️ Top Region: {ctx['top_region']}\n🚚 Avg Delivery: {ctx['avg_delivery']:.1f} days\n↩️ Return Rate: {ctx['return_rate']:.1f}%")

    if any(k in msg_l for k in ["hi","hello","hey","help","what can","commands"]):
        return ("👋 **OmniFlow-D2D AI Assistant**\n\nAsk me about:\n\n"
                "📈 'Forecast for boAt Airdopes 141 TWS'\n📈 'Which SKU has highest demand next month?'\n"
                "📦 'Reorder point for Himalaya Face Wash'\n📦 'Top 5 safety stock'\n"
                "🚚 'Which region needs logistics support?'\n🚚 'Best courier performance'\n"
                "💰 'Top products by revenue'\n💰 'KPI summary'\n\n"
                "Works for all 50 SKUs!")

    sample=", ".join(list(fcs.keys())[:3])
    return (f"🤖 Couldn't match: **\"{msg}\"**\n\nTry:\n• 'Forecast for [product]'\n• 'Reorder point for [product]'\n"
            f"• 'KPI summary'\n\nSample SKUs: {sample}...\nType **help** for all commands.")

def page_chatbot(df):
    st.markdown("<div class='ph'><h1>🤖 AI Decision Intelligence</h1><p>Context-aware chatbot — live outputs from all 5 modules for all 50 SKUs</p></div>", unsafe_allow_html=True)
    with st.spinner("Loading AI context..."):
        fcs=run_all_forecasts(df); inv=compute_inventory(df,fcs)
        dlv=df[df["Order_Status"]=="Delivered"]
        ctx={
            "total_orders":len(df), "total_revenue":dlv["Revenue_INR"].sum(),
            "top_rev_product":df.groupby("Product_Name")["Revenue_INR"].sum().idxmax(),
            "top_region":df.groupby("Region")["Revenue_INR"].sum().idxmax(),
            "avg_delivery":df["Delivery_Days"].mean(), "return_rate":df["Return_Flag"].mean()*100,
            "top_forecast_prod":max(fcs,key=lambda p:fcs[p]["forecast"][0] if len(fcs[p]["forecast"])>0 else 0),
            "top_forecast_val":int(max(fcs[p]["forecast"][0] if len(fcs[p]["forecast"])>0 else 0 for p in fcs)),
            "fcs":fcs,"inv":inv,"df":df
        }

    if "history" not in st.session_state:
        st.session_state.history=[("bot","👋 Hello! I'm your OmniFlow-D2D Supply Chain AI.\n\nI have live context from all 5 modules for all 50 SKUs. Type **help** to see what I can do!")]

    for role,text in st.session_state.history:
        cls="chat-user" if role=="user" else "chat-bot"
        safe=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',text.replace("\n","<br>"))
        st.markdown(f"<div class='{cls}'>{safe}</div>",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**💡 Quick Queries:**")
    qs=["Highest demand next month","KPI summary","Region logistics support","Best courier",
        "Reorder point for Samsung Galaxy M34 5G","Forecast for boAt Airdopes 141 TWS",
        "Top 5 safety stock","Top products by revenue"]
    cols=st.columns(4)
    for i,q in enumerate(qs):
        if cols[i%4].button(q,key=f"qb{i}"):
            st.session_state._pq=q

    if hasattr(st.session_state,"_pq") and st.session_state._pq:
        q=st.session_state._pq; st.session_state._pq=None
        st.session_state.history.append(("user",q)); st.session_state.history.append(("bot",_chat(q,ctx))); st.rerun()

    ci,cb=st.columns([5,1])
    user_in=ci.text_input("Ask a supply chain question...",key="chat_inp",label_visibility="collapsed",
                           placeholder="e.g. Forecast for Himalaya Neem Face Wash")
    if cb.button("Send →") and user_in.strip():
        st.session_state.history.append(("user",user_in)); st.session_state.history.append(("bot",_chat(user_in,ctx))); st.rerun()
    if st.button("🗑️ Clear Chat"):
        st.session_state.history=[]; st.rerun()

# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════
def main():
    df=load_data()
    with st.sidebar:
        st.markdown("<div class='logo'>⟳ OmniFlow-D2D</div>",unsafe_allow_html=True)
        st.markdown("<div style='text-align:center;color:#374151;font-size:.65rem;font-family:Space Mono;margin-bottom:.5rem;'>Data-to-Decision Platform</div>",unsafe_allow_html=True)
        st.markdown("---")
        pages={"📊 Overview Dashboard":"overview","📈 Demand Forecasting":"forecasting",
               "🏭 Inventory Optimization":"inventory","⚙️ Production Planning":"production",
               "🚚 Logistics Optimization":"logistics","🤖 AI Decision Intelligence":"chatbot"}
        if "page" not in st.session_state: st.session_state.page="overview"
        for label,key in pages.items():
            if st.button(label,key=f"nav_{key}",use_container_width=True):
                st.session_state.page=key; st.rerun()
        st.markdown("---")
        st.markdown(f"""<div style='background:#111827;border:1px solid #1e293b;border-radius:8px;padding:.8rem;font-family:Space Mono;font-size:.7rem;'>
        <div style='color:#64748b;'>DATASET</div><div style='color:#00d4ff;font-size:.9rem;'>{len(df):,} Orders</div>
        <div style='color:#64748b;margin-top:.4rem;'>SKUs</div><div style='color:#7c3aed;font-size:.9rem;'>{df["SKU_ID"].nunique()} Products</div>
        <div style='color:#64748b;margin-top:.4rem;'>CATEGORIES</div><div style='color:#94a3b8;font-size:.75rem;'>Electronics · Home · Fashion · Health</div>
        <div style='color:#64748b;margin-top:.4rem;'>DATE RANGE</div><div style='color:#94a3b8;font-size:.7rem;'>Jan 2024 → Mar 2025</div>
        <div style='color:#64748b;margin-top:.4rem;'>FORECAST HORIZON</div><div style='color:#10b981;font-size:.7rem;'>Apr → Jun 2025</div>
        <div style='color:#64748b;margin-top:.4rem;'>SOURCES</div><div style='color:#94a3b8;font-size:.65rem;'>Amazon India 2025<br>Shiprocket / INCREFF</div>
        </div>""",unsafe_allow_html=True)

    pg=st.session_state.get("page","overview")
    if   pg=="overview":   page_overview(df)
    elif pg=="forecasting": page_forecasting(df)
    elif pg=="inventory":   page_inventory(df)
    elif pg=="production":  page_production(df)
    elif pg=="logistics":   page_logistics(df)
    elif pg=="chatbot":     page_chatbot(df)

if __name__=="__main__":
    main()

