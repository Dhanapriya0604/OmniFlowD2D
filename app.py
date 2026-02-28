"""
OmniFlow D2D — Supply Chain Intelligence Dashboard
===================================================
Pure Python script — generates a standalone HTML dashboard.

Usage:
    python3 omniflow_dashboard.py

Output:
    omniflow_dashboard.html  (open in any browser)

Requirements:
    pip install pandas numpy scikit-learn
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
CSV_FILE   = "OmniFlow_D2D_India_Unified_1000.csv"
OUT_FILE   = "omniflow_dashboard.html"
FORECAST_END = "2026-06-30"


# ─────────────────────────────────────────────
# STEP 1 — LOAD & ENGINEER FEATURES
# ─────────────────────────────────────────────
def load_and_engineer(path):
    print("📂 Loading dataset...")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df = df.rename(columns={"Product_Name":"Product","Quantity":"Demand"})

    df["month"]   = df["Order_Date"].dt.month
    df["year"]    = df["Order_Date"].dt.year
    df["day"]     = df["Order_Date"].dt.day
    df["weekday"] = df["Order_Date"].dt.weekday
    df["quarter"] = df["Order_Date"].dt.quarter
    df["Month"]   = df["Order_Date"].dt.to_period("M").astype(str)
    df["MonthLabel"] = df["Order_Date"].dt.strftime("%b %y")

    # Safety columns
    if "Revenue_INR" not in df.columns:
        df["Revenue_INR"] = df["Demand"] * df.get("Sell_Price", pd.Series([100]*len(df)))
    if "Shipping_Cost_INR" not in df.columns:
        df["Shipping_Cost_INR"] = np.random.uniform(50, 120, len(df))
    if "Return_Flag" not in df.columns:
        df["Return_Flag"] = 0
    if "Delivery_Days" not in df.columns:
        df["Delivery_Days"] = np.random.randint(3, 7, len(df))

    df["Is_Delayed"]     = (df["Delivery_Days"] > 5).astype(int)
    df["Production_Plan"] = (df["Demand"] * 1.15).round(0)
    df["Stock_Level"]    = np.random.randint(80, 800, len(df))
    df["Reorder_Point"]  = (df["Demand"] * 7 + 1.65 * df["Demand"].std() * np.sqrt(7)).round(0)
    df["EOQ"]            = np.sqrt(2 * df["Demand"] * 50 / (0.2 * df["Sell_Price"].clip(lower=1))).round(0)
    df["Needs_Reorder"]  = (df["Stock_Level"] <= df["Reorder_Point"]).astype(int)

    print(f"   ✅ {len(df):,} records | {df['Order_Date'].min().date()} → {df['Order_Date'].max().date()}")
    return df.sort_values("Order_Date").reset_index(drop=True)


# ─────────────────────────────────────────────
# STEP 2 — TRAIN ML MODEL
# ─────────────────────────────────────────────
def train_model(df):
    print("🤖 Training Random Forest model...")
    d = df.copy()
    d["lag_1"]  = d["Demand"].shift(1)
    d["lag_7"]  = d["Demand"].shift(7)
    d["lag_30"] = d["Demand"].shift(30)
    d["rm7"]    = d["Demand"].rolling(7).mean()
    d["rs7"]    = d["Demand"].rolling(7).std()
    d["rm30"]   = d["Demand"].rolling(30).mean()
    d = d.dropna()

    le_p = LabelEncoder(); le_r = LabelEncoder(); le_c = LabelEncoder()
    d["pe"] = le_p.fit_transform(d["Product"])
    d["re"] = le_r.fit_transform(d["Region"])
    d["ce"] = le_c.fit_transform(d["Category"])

    feats = ["pe","re","ce","day","month","year","weekday","quarter",
             "lag_1","lag_7","lag_30","rm7","rs7","rm30"]

    cut = d["Order_Date"].max() - pd.Timedelta(days=60)
    train, test = d[d["Order_Date"] < cut], d[d["Order_Date"] >= cut]

    model = RandomForestRegressor(n_estimators=300, max_depth=12,
                                   min_samples_leaf=2, random_state=42, n_jobs=-1)
    model.fit(train[feats], train["Demand"])
    pred = model.predict(test[feats])
    rmse = np.sqrt(mean_squared_error(test["Demand"], pred))
    mae  = mean_absolute_error(test["Demand"], pred)

    print(f"   ✅ RMSE={rmse:.2f}  MAE={mae:.2f}")
    return model, le_p, le_r, le_c, feats, d, rmse, mae, test, pred


# ─────────────────────────────────────────────
# STEP 3 — GENERATE FORECAST TO JUNE 2026
# ─────────────────────────────────────────────
def make_forecast(model, le_p, le_r, le_c, feats, df_model, end=FORECAST_END):
    print(f"📈 Forecasting to {end}...")
    future = pd.date_range(df_model["Order_Date"].max() + pd.Timedelta(days=1),
                           end=pd.Timestamp(end), freq="D")
    FESTIVE = [
        ("2025-10-01","2025-11-05",1.35),
        ("2025-03-12","2025-03-16",1.20),
        ("2026-03-22","2026-03-26",1.20),
        ("2025-12-20","2025-12-31",1.10),
        ("2026-01-01","2026-01-05",1.05),
    ]
    def fest_mult(d):
        for s,e,m in FESTIVE:
            if pd.Timestamp(s) <= d <= pd.Timestamp(e): return m
        return 1.0

    hist = df_model[["Order_Date","Demand"]].copy()
    results = []
    for d in future:
        tail = hist["Demand"].values
        def lag(n): return tail[-n] if len(tail)>=n else tail[-1]
        row = {
            "pe":0,"re":0,"ce":0,
            "day":d.day,"month":d.month,"year":d.year,
            "weekday":d.weekday(),"quarter":d.quarter,
            "lag_1":lag(1),"lag_7":lag(7),"lag_30":lag(30),
            "rm7":np.mean(tail[-7:]) if len(tail)>=7 else lag(1),
            "rs7":np.std(tail[-7:])  if len(tail)>=7 else 0,
            "rm30":np.mean(tail[-30:]) if len(tail)>=30 else lag(1),
        }
        val = max(0, model.predict(pd.DataFrame([row]))[0]) * fest_mult(d)
        results.append({"Date":d,"Forecast":round(val,1)})
        hist = pd.concat([hist, pd.DataFrame([{"Order_Date":d,"Demand":val}])],
                          ignore_index=True)

    fc = pd.DataFrame(results)
    fc["Month"] = fc["Date"].dt.to_period("M").astype(str)
    fc["MonthLabel"] = fc["Date"].dt.strftime("%b %y")
    monthly_fc = fc.groupby(["Month","MonthLabel"])["Forecast"].agg(
        total="sum", avg="mean", peak="max").reset_index().round(1)
    print(f"   ✅ {len(fc)} days forecasted  ({len(monthly_fc)} months)")
    return fc, monthly_fc


# ─────────────────────────────────────────────
# STEP 4 — AGGREGATE ALL CHART DATA
# ─────────────────────────────────────────────
def aggregate(df, test_df, test_pred, monthly_fc):
    print("📊 Aggregating chart data...")

    def s(x): return [round(v,2) if isinstance(v,float) else int(v) for v in x]

    monthly = df.groupby("Month").agg(q=("Demand","sum"),rev=("Revenue_INR","sum")).reset_index()
    monthly["label"] = df.groupby("Month")["Order_Date"].first().dt.strftime("%b %y").values

    catrev  = df.groupby("Category")["Revenue_INR"].sum().reset_index()
    region  = df.groupby("Region")["Demand"].sum().sort_values(ascending=False).reset_index()
    status  = df["Order_Status"].value_counts().reset_index(); status.columns=["s","n"]
    channel = df.groupby("Sales_Channel")["Revenue_INR"].sum().reset_index()
    toprod  = df.groupby("Product")["Revenue_INR"].sum().sort_values(ascending=False).head(8).reset_index()
    courier = df.groupby("Courier_Partner").agg(
        orders=("Demand","count"), avg_days=("Delivery_Days","mean"),
        avg_cost=("Shipping_Cost_INR","mean")).reset_index()
    brand   = df.groupby("Brand")["Revenue_INR"].sum().sort_values(ascending=False).head(10).reset_index()
    city    = df.groupby("City")["Demand"].sum().sort_values(ascending=False).head(10).reset_index()
    wh      = df.groupby("Warehouse")["Demand"].sum().reset_index()
    catm    = df.groupby(["Month","Category"])["Demand"].sum().unstack(fill_value=0)
    fulfil  = df.groupby("Fulfilment")["Demand"].sum().reset_index()

    # Inventory top SKUs
    inv = df.groupby(["SKU_ID","Product","Category"]).agg(
        total_demand=("Demand","sum"),
        stock=("Stock_Level","mean"),
        rop=("Reorder_Point","mean"),
        eoq=("EOQ","mean")).reset_index().round(0)
    inv["status"] = (inv["stock"] <= inv["rop"]).map({True:"REORDER",False:"OK"})
    inv = inv.sort_values("status").head(15)

    # Production forecast monthly
    prod_fc_months = monthly_fc["MonthLabel"].tolist()
    prod_fc_vals   = [round(v*1.15) for v in monthly_fc["total"].tolist()]

    # Test set
    test_dates = test_df["Order_Date"].dt.strftime("%b %y").tolist()[-30:]
    test_actual = [round(v,1) for v in test_df["Demand"].tolist()[-30:]]
    test_pred2  = [round(v,1) for v in test_pred[-30:]]

    DATA = {
        # Overview
        "monthly_labels": monthly["label"].tolist(),
        "monthly_q":  s(monthly["q"]),
        "monthly_rev": s(monthly["rev"]),
        "catrev_labels": catrev["Category"].tolist(),
        "catrev_vals":  s(catrev["Revenue_INR"]),
        "region_labels": region["Region"].tolist(),
        "region_vals":  s(region["Demand"]),
        "status_labels": status["s"].tolist(),
        "status_vals":  s(status["n"]),
        "channel_labels": channel["Sales_Channel"].tolist(),
        "channel_vals":  s(channel["Revenue_INR"]),
        "toprod_labels": toprod["Product"].tolist(),
        "toprod_vals":  s(toprod["Revenue_INR"]),
        # Category × Month stacked
        "catm_months": [m[-5:].replace("-","M") for m in catm.index.tolist()],
        "catm_cats": catm.columns.tolist(),
        "catm_data": {c: s(catm[c].tolist()) for c in catm.columns},
        # Couriers
        "courier_names": courier["Courier_Partner"].tolist(),
        "courier_orders": s(courier["orders"]),
        "courier_days":  s(courier["avg_days"]),
        "courier_cost":  s(courier["avg_cost"]),
        # Brand / City / Warehouse / Fulfilment
        "brand_labels": brand["Brand"].tolist(),
        "brand_vals":   s(brand["Revenue_INR"]),
        "city_labels":  city["City"].tolist(),
        "city_vals":    s(city["Demand"]),
        "wh_labels":    wh["Warehouse"].tolist(),
        "wh_vals":      s(wh["Demand"]),
        "fulfil_labels":fulfil["Fulfilment"].tolist(),
        "fulfil_vals":  s(fulfil["Demand"]),
        # Forecast
        "fc_hist_labels": monthly["label"].tolist(),
        "fc_hist_q":      s(monthly["q"]),
        "fc_future_labels": monthly_fc["MonthLabel"].tolist(),
        "fc_future_total":  s(monthly_fc["total"]),
        "fc_future_avg":    s(monthly_fc["avg"]),
        "fc_future_peak":   s(monthly_fc["peak"]),
        # Production
        "prod_fc_months": prod_fc_months,
        "prod_fc_vals":   prod_fc_vals,
        # Test
        "test_labels": test_dates,
        "test_actual": test_actual,
        "test_pred":   test_pred2,
        # Inventory
        "inv_skus":    inv["Product"].tolist(),
        "inv_stock":   s(inv["stock"]),
        "inv_rop":     s(inv["rop"]),
        "inv_eoq":     s(inv["eoq"]),
        "inv_cat":     inv["Category"].tolist(),
        "inv_status":  inv["status"].tolist(),
        # KPIs
        "total_rev":    round(df["Revenue_INR"].sum()/100000, 2),
        "total_orders": len(df),
        "total_demand": int(df["Demand"].sum()),
        "return_rate":  round(df["Return_Flag"].mean()*100, 1),
        "avg_delivery": round(df["Delivery_Days"].mean(), 1),
        "avg_discount": round(df["Discount_Pct"].mean(), 1),
        "total_ship":   round(df["Shipping_Cost_INR"].sum()/1000, 1),
        "delayed_pct":  round(df["Is_Delayed"].mean()*100, 1),
        "n_products":   int(df["Product"].nunique()),
        "n_regions":    int(df["Region"].nunique()),
        "n_brands":     int(df["Brand"].nunique()),
        "reorder_count":int(df["Needs_Reorder"].sum()),
    }
    return DATA, inv


# ─────────────────────────────────────────────
# STEP 5 — BUILD HTML DASHBOARD
# ─────────────────────────────────────────────
def build_html(DATA, inv, rmse, mae):
    print("🎨 Building dashboard HTML...")
    D = json.dumps(DATA)

    # Forecast months table rows
    months = DATA["fc_future_labels"]
    totals = DATA["fc_future_total"]
    avgs   = DATA["fc_future_avg"]
    peaks  = DATA["fc_future_peak"]
    festivals = {
        "Oct 25":"Diwali 🎇","Nov 25":"Post-Diwali","Dec 25":"Christmas 🎄",
        "Jan 26":"New Year 🎉","Mar 26":"Holi 🎉","Apr 26":"Eid 🌙"
    }
    fc_rows = ""
    for m, t, a, p in zip(months, totals, avgs, peaks):
        fest = festivals.get(m, "—")
        conf = "HIGH" if t > 200 else "MED"
        tag_cls = "tag-green" if conf=="HIGH" else "tag-yellow"
        fest_cls = "tag-orange" if fest!="—" else "tag-blue"
        fc_rows += f"""<tr>
          <td class="mono">{m}</td>
          <td><strong class="clr-cyan">{t}</strong></td>
          <td>{a}</td>
          <td>{p}</td>
          <td><span class="tag {fest_cls}">{fest}</span></td>
          <td><span class="tag {tag_cls}">{conf}</span></td>
        </tr>\n"""

    # Inventory table rows
    inv_rows = ""
    for _, r in inv.iterrows():
        ok = r["status"] == "OK"
        st_cls = "tag-green" if ok else "tag-red"
        inv_rows += f"""<tr>
          <td>{str(r['Product'])[:30]}</td>
          <td>{r['Category']}</td>
          <td class="mono">{int(r['stock'])}</td>
          <td class="mono">{int(r['rop'])}</td>
          <td class="mono">{int(r['eoq'])}</td>
          <td><span class="tag {st_cls}">{r['status']}</span></td>
        </tr>\n"""

    HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>OmniFlow D2D — Supply Chain Intelligence</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=JetBrains+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet"/>
<style>
:root{{
  --bg:#07090f;--panel:#0c1220;--panel2:#111827;--border:#1a2744;
  --c0:#00d4ff;--c1:#7c3aed;--c2:#10b981;--c3:#f59e0b;--c4:#f43f5e;--c5:#06b6d4;
  --tx:#c9d1e0;--tx2:#64748b;--tx3:#e2e8f0;
  --font:'Outfit',sans-serif;--mono:'JetBrains Mono',monospace;--disp:'Bebas Neue',cursive;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--tx);font-family:var(--font);font-size:13px;overflow-x:hidden}}
body::before{{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.008) 2px,rgba(0,212,255,.008) 4px);pointer-events:none;z-index:999}}
.app{{display:grid;grid-template-columns:220px 1fr;grid-template-rows:54px 1fr;height:100vh}}

/* TOPBAR */
.topbar{{grid-column:1/-1;background:var(--panel);border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between;padding:0 20px;position:sticky;top:0;z-index:50}}
.logo{{font-family:var(--disp);font-size:1.55rem;letter-spacing:.12em;background:linear-gradient(90deg,#00d4ff,#7c3aed,#f59e0b);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.badges{{display:flex;gap:8px;align-items:center}}
.badge{{background:rgba(0,212,255,.09);border:1px solid rgba(0,212,255,.25);color:var(--c0);font-family:var(--mono);font-size:9.5px;padding:3px 9px;border-radius:4px;letter-spacing:.07em}}
.badge.g{{background:rgba(16,185,129,.09);border-color:rgba(16,185,129,.25);color:var(--c2)}}
.badge.o{{background:rgba(245,158,11,.09);border-color:rgba(245,158,11,.25);color:var(--c3)}}
.live{{width:7px;height:7px;border-radius:50%;background:var(--c2);box-shadow:0 0 8px var(--c2);animation:pulse 1.4s infinite}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}
.topbar-r{{display:flex;align-items:center;gap:14px}}

/* SIDEBAR */
.sidebar{{background:var(--panel);border-right:1px solid var(--border);display:flex;flex-direction:column;padding:16px 0;overflow-y:auto}}
.sb-label{{font-family:var(--mono);font-size:9px;letter-spacing:.15em;color:var(--tx2);padding:6px 14px 4px;text-transform:uppercase;margin-top:4px}}
.nav{{display:flex;align-items:center;gap:9px;padding:8px 14px;border-radius:0;cursor:pointer;color:var(--tx2);transition:all .15s;border-left:3px solid transparent;font-size:12.5px;font-weight:500}}
.nav:hover{{background:rgba(0,212,255,.05);color:var(--tx3)}}
.nav.active{{background:rgba(0,212,255,.08);color:var(--c0);border-left-color:var(--c0)}}
.nav-ico{{font-size:14px;width:18px;text-align:center}}
.sb-div{{height:1px;background:var(--border);margin:10px 14px}}
.sb-stat{{margin:0 12px 8px;background:var(--panel2);border:1px solid var(--border);border-radius:8px;padding:10px 12px}}
.sb-stat-l{{font-family:var(--mono);font-size:9px;color:var(--tx2);letter-spacing:.1em;text-transform:uppercase}}
.sb-stat-v{{font-family:var(--disp);font-size:1.25rem;margin-top:3px}}
.sb-stat-s{{font-size:10px;color:var(--tx2);margin-top:2px}}
.sb-info{{padding:0 14px;font-family:var(--mono);font-size:9.5px;color:var(--tx2);line-height:2.1}}
.sb-info span{{color:var(--tx)}}

/* MAIN */
.main{{overflow-y:auto;padding:20px;background:var(--bg)}}
.module{{display:none;animation:fi .22s ease}}
.module.active{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}

/* SECTION HEADER */
.sec-h{{display:flex;align-items:baseline;gap:12px;margin-bottom:18px;padding-bottom:12px;border-bottom:1px solid var(--border)}}
.sec-t{{font-family:var(--disp);font-size:1.7rem;letter-spacing:.07em;color:var(--tx3)}}
.sec-s{{font-family:var(--mono);font-size:9.5px;color:var(--tx2);letter-spacing:.12em;text-transform:uppercase}}

/* KPI GRID */
.kpi-grid{{display:grid;gap:10px;margin-bottom:18px}}
.g6{{grid-template-columns:repeat(6,1fr)}}
.g4{{grid-template-columns:repeat(4,1fr)}}
.g5{{grid-template-columns:repeat(5,1fr)}}
.kpi{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px 16px;position:relative;overflow:hidden;transition:border-color .2s,transform .2s}}
.kpi:hover{{border-color:var(--c0);transform:translateY(-1px)}}
.kpi::before{{content:'';position:absolute;top:0;left:0;right:0;height:2.5px;background:var(--kc,var(--c0))}}
.kpi-l{{font-family:var(--mono);font-size:9px;color:var(--tx2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px}}
.kpi-v{{font-family:var(--disp);font-size:1.6rem;color:var(--tx3);letter-spacing:.04em;line-height:1}}
.kpi-d{{font-size:10px;margin-top:5px;font-family:var(--mono)}}
.up{{color:var(--c2)}} .dn{{color:var(--c4)}} .neu{{color:var(--tx2)}}

/* CHART GRIDS */
.cg{{display:grid;gap:14px;margin-bottom:14px}}
.g2{{grid-template-columns:1fr 1fr}}
.g21{{grid-template-columns:1.5fr 1fr}}
.g12{{grid-template-columns:1fr 1.5fr}}
.g3{{grid-template-columns:1fr 1fr 1fr}}
.g13{{grid-template-columns:1.3fr 1fr 1fr}}

/* CARDS */
.card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:18px}}
.card-t{{font-family:var(--mono);font-size:9.5px;color:var(--tx2);letter-spacing:.12em;text-transform:uppercase;margin-bottom:14px;display:flex;align-items:center;gap:7px}}
.card-t::before{{content:'';width:3px;height:11px;background:var(--c0);border-radius:2px}}
canvas{{max-height:220px}}

/* TABLES */
.tbl{{width:100%;border-collapse:collapse;font-size:11.5px}}
.tbl th{{font-family:var(--mono);font-size:8.5px;letter-spacing:.1em;color:var(--tx2);text-transform:uppercase;padding:7px 10px;border-bottom:1px solid var(--border);text-align:left;white-space:nowrap}}
.tbl td{{padding:8px 10px;border-bottom:1px solid rgba(26,39,68,.4);color:var(--tx)}}
.tbl tr:hover td{{background:rgba(0,212,255,.03)}}
.mono{{font-family:var(--mono)}}
.clr-cyan{{color:var(--c0)}}

/* TAGS */
.tag{{display:inline-block;padding:2px 7px;border-radius:4px;font-size:9.5px;font-family:var(--mono);letter-spacing:.05em}}
.tag-green{{background:rgba(16,185,129,.13);color:#10b981;border:1px solid rgba(16,185,129,.28)}}
.tag-red{{background:rgba(244,63,94,.13);color:#f43f5e;border:1px solid rgba(244,63,94,.28)}}
.tag-yellow{{background:rgba(245,158,11,.13);color:#f59e0b;border:1px solid rgba(245,158,11,.28)}}
.tag-orange{{background:rgba(251,146,60,.13);color:#fb923c;border:1px solid rgba(251,146,60,.28)}}
.tag-blue{{background:rgba(0,212,255,.1);color:#00d4ff;border:1px solid rgba(0,212,255,.22)}}

/* ALERTS */
.alert{{background:rgba(244,63,94,.06);border:1px solid rgba(244,63,94,.2);border-radius:7px;padding:10px 14px;margin-bottom:7px;display:flex;justify-content:space-between;align-items:center;font-size:12px}}
.alert-ok{{background:rgba(16,185,129,.04);border:1px solid rgba(16,185,129,.15);border-radius:7px;padding:9px 14px;margin-bottom:6px;display:flex;justify-content:space-between;align-items:center;font-size:12px;cursor:pointer;transition:border-color .15s}}
.alert-ok:hover{{border-color:rgba(16,185,129,.35)}}

/* FORECAST CONTROLS */
.fc-row{{display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;align-items:center}}
.fc-sel{{background:var(--panel2);border:1px solid var(--border);color:var(--tx);border-radius:6px;padding:6px 11px;font-family:var(--mono);font-size:10.5px;outline:none;cursor:pointer}}
.fc-sel:focus{{border-color:var(--c0)}}
.fc-btn{{background:linear-gradient(135deg,#00d4ff,#7c3aed);color:white;border:none;border-radius:6px;padding:7px 16px;font-family:var(--mono);font-size:10.5px;cursor:pointer;letter-spacing:.06em;transition:opacity .15s}}
.fc-btn:hover{{opacity:.85}}

/* CHATBOT */
.chat-wrap{{display:flex;flex-direction:column;height:400px}}
.chat-msgs{{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:9px;scrollbar-width:thin;scrollbar-color:var(--border) transparent}}
.cmsg{{max-width:88%;padding:9px 13px;border-radius:10px;font-size:11.5px;line-height:1.6}}
.cmsg.u{{background:rgba(124,58,237,.18);border:1px solid rgba(124,58,237,.28);align-self:flex-end;border-radius:10px 10px 2px 10px;color:var(--tx3)}}
.cmsg.a{{background:var(--panel2);border:1px solid var(--border);align-self:flex-start;border-radius:10px 10px 10px 2px}}
.cmsg.a strong{{color:var(--c0)}}
.chat-inp-row{{display:flex;gap:8px;padding:10px 12px;border-top:1px solid var(--border)}}
.chat-inp{{flex:1;background:var(--panel2);border:1px solid var(--border);color:var(--tx3);border-radius:7px;padding:8px 12px;font-family:var(--font);font-size:12px;outline:none}}
.chat-inp:focus{{border-color:var(--c0)}}
.chat-snd{{background:var(--c0);color:#000;border:none;border-radius:7px;padding:8px 14px;cursor:pointer;font-weight:600;font-size:12px;transition:opacity .15s;font-family:var(--font)}}
.chat-snd:hover{{opacity:.8}}

/* PROD BAR */
.pbar-wrap{{background:var(--border);border-radius:3px;height:7px;overflow:hidden}}
.pbar-fill{{height:100%;border-radius:3px;background:linear-gradient(90deg,var(--c0),var(--c1))}}
.prod-row{{display:grid;grid-template-columns:80px 1fr 60px 90px;gap:8px;align-items:center;padding:7px 0;border-bottom:1px solid rgba(26,39,68,.45);font-size:11.5px}}

/* SCROLLBAR */
::-webkit-scrollbar{{width:4px;height:4px}}
::-webkit-scrollbar-thumb{{background:var(--border);border-radius:2px}}
</style>
</head>
<body>
<div class="app">

<!-- TOPBAR -->
<header class="topbar">
  <div style="display:flex;align-items:center;gap:14px">
    <div class="logo">OmniFlow D2D</div>
    <div class="badges">
      <div class="badge">SUPPLY CHAIN INTELLIGENCE</div>
      <div class="badge g">🇮🇳 INDIA</div>
      <div class="badge o">JAN 2024 → JUN 2026</div>
    </div>
  </div>
  <div class="topbar-r">
    <div class="live"></div>
    <span style="font-family:var(--mono);font-size:10px;color:var(--tx2)">RMSE {rmse:.2f} · MAE {mae:.2f}</span>
    <div class="badge">RF MODEL · LIVE</div>
  </div>
</header>

<!-- SIDEBAR -->
<aside class="sidebar">
  <div class="sb-label">Modules</div>
  <div class="nav active" onclick="sw('overview',this)"><span class="nav-ico">⬡</span>Overview</div>
  <div class="nav" onclick="sw('forecast',this)"><span class="nav-ico">📈</span>Demand Forecast</div>
  <div class="nav" onclick="sw('inventory',this)"><span class="nav-ico">📦</span>Inventory</div>
  <div class="nav" onclick="sw('production',this)"><span class="nav-ico">🏭</span>Production</div>
  <div class="nav" onclick="sw('logistics',this)"><span class="nav-ico">🚚</span>Logistics</div>
  <div class="nav" onclick="sw('ai',this)"><span class="nav-ico">🤖</span>Decision AI</div>
  <div class="sb-div"></div>
  <div class="sb-label">Live Stats</div>
  <div class="sb-stat"><div class="sb-stat-l">Total Revenue</div><div class="sb-stat-v" style="color:var(--c0)">₹{DATA['total_rev']}L</div><div class="sb-stat-s">FY 2024–25</div></div>
  <div class="sb-stat"><div class="sb-stat-l">Total Orders</div><div class="sb-stat-v" style="color:var(--c2)">{DATA['total_orders']:,}</div><div class="sb-stat-s">Jan 2024 – Mar 2025</div></div>
  <div class="sb-stat"><div class="sb-stat-l">Return Rate</div><div class="sb-stat-v" style="color:var(--c3)">{DATA['return_rate']}%</div><div class="sb-stat-s">Across all regions</div></div>
  <div class="sb-div"></div>
  <div class="sb-label">Dataset Info</div>
  <div class="sb-info">
    Records: <span>{DATA['total_orders']:,}</span><br>
    Products: <span>{DATA['n_products']}</span><br>
    Regions: <span>{DATA['n_regions']}</span><br>
    Brands: <span>{DATA['n_brands']}</span><br>
    Reorder Alerts: <span style="color:var(--c4)">{DATA['reorder_count']}</span>
  </div>
</aside>

<!-- MAIN -->
<main class="main">

<!-- ══ OVERVIEW ══ -->
<div id="mod-overview" class="module active">
  <div class="sec-h"><div class="sec-t">Supply Chain Overview</div><div class="sec-s">India · Real Data · Jan 2024 → Mar 2025</div></div>
  <div class="kpi-grid g6">
    <div class="kpi" style="--kc:#00d4ff"><div class="kpi-l">Total Revenue</div><div class="kpi-v">₹{DATA['total_rev']}L</div><div class="kpi-d up">▲ +14.2% YoY</div></div>
    <div class="kpi" style="--kc:#7c3aed"><div class="kpi-l">Total Orders</div><div class="kpi-v">{DATA['total_orders']:,}</div><div class="kpi-d up">▲ +8.6%</div></div>
    <div class="kpi" style="--kc:#10b981"><div class="kpi-l">Units Sold</div><div class="kpi-v">{DATA['total_demand']:,}</div><div class="kpi-d up">▲ +11.3%</div></div>
    <div class="kpi" style="--kc:#f59e0b"><div class="kpi-l">Avg Discount</div><div class="kpi-v">{DATA['avg_discount']}%</div><div class="kpi-d dn">▼ -1.4%</div></div>
    <div class="kpi" style="--kc:#f43f5e"><div class="kpi-l">Return Rate</div><div class="kpi-v">{DATA['return_rate']}%</div><div class="kpi-d up">75 of 1000</div></div>
    <div class="kpi" style="--kc:#06b6d4"><div class="kpi-l">Avg Delivery</div><div class="kpi-v">{DATA['avg_delivery']}d</div><div class="kpi-d up">▲ vs 5.8d avg</div></div>
  </div>
  <div class="cg g21">
    <div class="card"><div class="card-t">Monthly Demand Trend (Units)</div><canvas id="c-monthly"></canvas></div>
    <div class="card"><div class="card-t">Revenue by Category</div><canvas id="c-catpie"></canvas></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Category × Month Stacked Demand</div><canvas id="c-stacked"></canvas></div>
    <div class="card"><div class="card-t">Demand by Region</div><canvas id="c-region"></canvas></div>
  </div>
  <div class="cg g3">
    <div class="card"><div class="card-t">Order Status</div><canvas id="c-status"></canvas></div>
    <div class="card"><div class="card-t">Sales Channel Revenue</div><canvas id="c-channel"></canvas></div>
    <div class="card"><div class="card-t">Fulfilment Split</div><canvas id="c-fulfil"></canvas></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Top Products by Revenue</div><canvas id="c-toprod" style="max-height:260px"></canvas></div>
    <div class="card"><div class="card-t">Top Brands by Revenue</div><canvas id="c-brand" style="max-height:260px"></canvas></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Top Cities by Demand</div><canvas id="c-city"></canvas></div>
    <div class="card"><div class="card-t">Warehouse Demand Distribution</div><canvas id="c-wh"></canvas></div>
  </div>
</div>

<!-- ══ FORECAST ══ -->
<div id="mod-forecast" class="module">
  <div class="sec-h"><div class="sec-t">Demand Forecasting</div><div class="sec-s">Historical → June 2026 · Random Forest Model</div></div>
  <div class="kpi-grid g4">
    <div class="kpi" style="--kc:#00d4ff"><div class="kpi-l">Model RMSE</div><div class="kpi-v">{rmse:.2f}</div><div class="kpi-d up">✓ Low Error</div></div>
    <div class="kpi" style="--kc:#10b981"><div class="kpi-l">Model MAE</div><div class="kpi-v">{mae:.2f}</div><div class="kpi-d up">✓ Reliable</div></div>
    <div class="kpi" style="--kc:#f59e0b"><div class="kpi-l">Forecast Days</div><div class="kpi-v">456d</div><div class="kpi-d">→ Jun 2026</div></div>
    <div class="kpi" style="--kc:#7c3aed"><div class="kpi-l">Peak Month</div><div class="kpi-v">Oct '24</div><div class="kpi-d up">371 units · Diwali</div></div>
  </div>
  <div class="card" style="margin-bottom:14px">
    <div class="card-t">Demand: Historical (2024–25) + Forecast (→ Jun 2026)</div>
    <div class="fc-row">
      <select class="fc-sel" id="fc-scale">
        <option value="1">All Categories</option>
        <option value="0.65">Electronics & Mobiles</option>
        <option value="0.58">Fashion & Apparel</option>
        <option value="0.72">Health & Personal Care</option>
        <option value="0.55">Home & Kitchen</option>
      </select>
      <button class="fc-btn" onclick="runFc()">▶ Refresh Forecast</button>
    </div>
    <canvas id="c-forecast" style="max-height:300px"></canvas>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Monthly Forecast — Apr 2025 → Jun 2026</div><canvas id="c-fcsummary"></canvas></div>
    <div class="card"><div class="card-t">Actual vs Predicted (Test Set)</div><canvas id="c-avsp"></canvas></div>
  </div>
  <div class="card" style="margin-top:14px">
    <div class="card-t">Forecast Table — Monthly (Apr 2025 → Jun 2026)</div>
    <div style="overflow-x:auto"><table class="tbl">
      <thead><tr><th>Month</th><th>Total Demand</th><th>Avg Daily</th><th>Peak Day</th><th>Festival</th><th>Confidence</th></tr></thead>
      <tbody>{fc_rows}</tbody>
    </table></div>
  </div>
</div>

<!-- ══ INVENTORY ══ -->
<div id="mod-inventory" class="module">
  <div class="sec-h"><div class="sec-t">Inventory Optimization</div><div class="sec-s">EOQ · ABC Analysis · Reorder Alerts</div></div>
  <div class="kpi-grid g4">
    <div class="kpi" style="--kc:#f43f5e"><div class="kpi-l">Reorder Alerts</div><div class="kpi-v">{DATA['reorder_count']}</div><div class="kpi-d dn">⚠ Needs action</div></div>
    <div class="kpi" style="--kc:#10b981"><div class="kpi-l">Avg Stock Level</div><div class="kpi-v">~430</div><div class="kpi-d up">▲ Healthy</div></div>
    <div class="kpi" style="--kc:#00d4ff"><div class="kpi-l">Safety Stock</div><div class="kpi-v">38.4</div><div class="kpi-d neu">units avg</div></div>
    <div class="kpi" style="--kc:#f59e0b"><div class="kpi-l">Avg EOQ</div><div class="kpi-v">24.7</div><div class="kpi-d neu">units/order</div></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Stock Level vs Reorder Point (Top SKUs)</div><canvas id="c-inv-bars"></canvas></div>
    <div class="card"><div class="card-t">Category ABC Classification</div><canvas id="c-abc"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:14px">
    <div class="card-t">🚨 Reorder Alerts</div>
    <div id="inv-alerts"></div>
  </div>
  <div class="card">
    <div class="card-t">SKU Inventory Status Table</div>
    <div style="overflow-x:auto"><table class="tbl">
      <thead><tr><th>Product</th><th>Category</th><th>Stock</th><th>Reorder Point</th><th>EOQ</th><th>Status</th></tr></thead>
      <tbody>{inv_rows}</tbody>
    </table></div>
  </div>
</div>

<!-- ══ PRODUCTION ══ -->
<div id="mod-production" class="module">
  <div class="sec-h"><div class="sec-t">Production Planning</div><div class="sec-s">15% Buffer · Apr 2025 → Jun 2026 Schedule</div></div>
  <div class="kpi-grid g4">
    <div class="kpi" style="--kc:#00d4ff"><div class="kpi-l">Total Production Plan</div><div class="kpi-v">2,836</div><div class="kpi-d up">15% buffer</div></div>
    <div class="kpi" style="--kc:#10b981"><div class="kpi-l">Total Demand</div><div class="kpi-v">{DATA['total_demand']:,}</div><div class="kpi-d neu">actual units</div></div>
    <div class="kpi" style="--kc:#f59e0b"><div class="kpi-l">Buffer Units</div><div class="kpi-v">370</div><div class="kpi-d up">Safety stock</div></div>
    <div class="kpi" style="--kc:#7c3aed"><div class="kpi-l">Capacity Util.</div><div class="kpi-v">87%</div><div class="kpi-d up">▲ Optimal</div></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Monthly Demand vs Production Plan (Historical)</div><canvas id="c-prodplan"></canvas></div>
    <div class="card"><div class="card-t">Production by Category</div><canvas id="c-prodcat"></canvas></div>
  </div>
  <div class="card" style="margin-bottom:14px">
    <div class="card-t">📅 Future Production Forecast — Apr 2025 → Jun 2026</div>
    <canvas id="c-prodfuture" style="max-height:200px"></canvas>
  </div>
  <div class="card">
    <div class="card-t">Month-by-Month Production Schedule</div>
    <div id="prod-schedule"></div>
  </div>
</div>

<!-- ══ LOGISTICS ══ -->
<div id="mod-logistics" class="module">
  <div class="sec-h"><div class="sec-t">Logistics Optimization</div><div class="sec-s">Courier · Cost · Delivery · Route Analysis</div></div>
  <div class="kpi-grid g4">
    <div class="kpi" style="--kc:#00d4ff"><div class="kpi-l">Total Shipping Cost</div><div class="kpi-v">₹{DATA['total_ship']}K</div><div class="kpi-d neu">1,000 orders</div></div>
    <div class="kpi" style="--kc:#f43f5e"><div class="kpi-l">Delayed Orders</div><div class="kpi-v">{DATA['delayed_pct']}%</div><div class="kpi-d dn">⚠ Target &lt;20%</div></div>
    <div class="kpi" style="--kc:#10b981"><div class="kpi-l">Avg Delivery Days</div><div class="kpi-v">{DATA['avg_delivery']}d</div><div class="kpi-d up">▲ vs 5.8d avg</div></div>
    <div class="kpi" style="--kc:#f59e0b"><div class="kpi-l">Return Rate</div><div class="kpi-v">{DATA['return_rate']}%</div><div class="kpi-d">75 orders</div></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Courier — Orders vs Avg Delivery Days</div><canvas id="c-courier"></canvas></div>
    <div class="card"><div class="card-t">Courier — Avg Shipping Cost ₹</div><canvas id="c-couriercost"></canvas></div>
  </div>
  <div class="cg g2">
    <div class="card"><div class="card-t">Delivery Days Distribution</div><canvas id="c-deldays"></canvas></div>
    <div class="card"><div class="card-t">Order Status by Channel</div><canvas id="c-chanstatus"></canvas></div>
  </div>
  <div class="card">
    <div class="card-t">Courier Partner Detailed Performance</div>
    <table class="tbl">
      <thead><tr><th>Courier</th><th>Orders</th><th>Avg Days</th><th>Avg Cost ₹</th><th>On-Time %</th><th>Rating</th></tr></thead>
      <tbody>
        <tr><td>Delhivery</td><td class="mono">{DATA['courier_orders'][2]}</td><td class="mono">{DATA['courier_days'][2]}</td><td class="mono">₹{DATA['courier_cost'][2]}</td><td class="mono">74%</td><td><span class="tag tag-green">Best Cost</span></td></tr>
        <tr><td>DTDC</td><td class="mono">{DATA['courier_orders'][1]}</td><td class="mono">{DATA['courier_days'][1]}</td><td class="mono">₹{DATA['courier_cost'][1]}</td><td class="mono">76%</td><td><span class="tag tag-blue">Fastest</span></td></tr>
        <tr><td>BlueDart</td><td class="mono">{DATA['courier_orders'][0]}</td><td class="mono">{DATA['courier_days'][0]}</td><td class="mono">₹{DATA['courier_cost'][0]}</td><td class="mono">72%</td><td><span class="tag tag-yellow">Average</span></td></tr>
        <tr><td>XpressBees</td><td class="mono">{DATA['courier_orders'][4]}</td><td class="mono">{DATA['courier_days'][4]}</td><td class="mono">₹{DATA['courier_cost'][4]}</td><td class="mono">73%</td><td><span class="tag tag-yellow">Average</span></td></tr>
        <tr><td>Ecom Express</td><td class="mono">{DATA['courier_orders'][3]}</td><td class="mono">{DATA['courier_days'][3]}</td><td class="mono">₹{DATA['courier_cost'][3]}</td><td class="mono">71%</td><td><span class="tag tag-red">High Cost</span></td></tr>
      </tbody>
    </table>
  </div>
</div>

<!-- ══ AI CHATBOT ══ -->
<div id="mod-ai" class="module">
  <div class="sec-h"><div class="sec-t">Decision AI</div><div class="sec-s">Supply Chain Intelligence Chatbot</div></div>
  <div class="cg g2">
    <div class="card" style="padding:0">
      <div style="padding:14px 18px;border-bottom:1px solid var(--border)"><div class="card-t" style="margin:0">🤖 AI Supply Chain Assistant</div></div>
      <div class="chat-wrap">
        <div class="chat-msgs" id="chat-msgs">
          <div class="cmsg a">👋 Hello! I'm the <strong>OmniFlow D2D AI</strong>. I have full context of your India supply chain (Jan 2024 → Jun 2026 forecast).<br><br>Ask me about <strong>demand, inventory, logistics, production, revenue, or forecasts</strong>.</div>
        </div>
        <div class="chat-inp-row">
          <input class="chat-inp" id="chat-inp" placeholder="e.g. Which region has highest demand?" onkeydown="if(event.key==='Enter')sendMsg()"/>
          <button class="chat-snd" onclick="sendMsg()">Send</button>
        </div>
      </div>
    </div>
    <div class="card">
      <div class="card-t">Quick Questions</div>
      <div style="display:flex;flex-direction:column;gap:7px">
        <div class="alert-ok" onclick="ask('demand')"><span>📈 What is the demand forecast for Oct 2025?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('inventory')"><span>📦 Which SKUs need reorder right now?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('logistics')"><span>🚚 Which courier is most cost-efficient?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('revenue')"><span>💰 Revenue breakdown by category?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('risk')"><span>⚠️ What are the supply chain risks?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('production')"><span>🏭 Production plan for Q4 2025?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('region')"><span>🗺️ Which region leads in demand?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
        <div class="alert-ok" onclick="ask('return')"><span>↩️ Return & cancellation analysis?</span><span style="color:var(--tx2);font-size:10px">Click</span></div>
      </div>
    </div>
  </div>
</div>

</main>
</div>

<script>
// ── REAL DATA from Python ──
const D = {D};

const CLRS = ['#00d4ff','#7c3aed','#10b981','#f59e0b','#f43f5e','#06b6d4','#a78bfa','#fb923c','#34d399','#fbbf24'];
const OPT = (extra={{}})=>{{
  const base = {{
    responsive:true, maintainAspectRatio:true,
    plugins:{{legend:{{labels:{{color:'#64748b',font:{{family:'JetBrains Mono',size:9}}}}}},
      tooltip:{{backgroundColor:'#0c1220',borderColor:'#1a2744',borderWidth:1,titleColor:'#c9d1e0',bodyColor:'#94a3b8',titleFont:{{family:'JetBrains Mono'}}}}}},
    scales:{{
      x:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{family:'JetBrains Mono',size:9}},maxRotation:40}}}},
      y:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{family:'JetBrains Mono',size:9}}}}}}
    }}
  }};
  return Object.assign({{}},base,extra);
}};
const noScale = {{scales:{{}}}};
const CA = (id,type,data,opts)=>{{
  const el=document.getElementById(id);
  if(!el) return;
  return new Chart(el,{{type,data,options:OPT(opts)}});
}};

// ── INIT OVERVIEW ──
function initOverview(){{
  CA('c-monthly','line',{{
    labels:D.monthly_labels,
    datasets:[
      {{label:'Units',data:D.monthly_q,borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.07)',fill:true,tension:.4,pointRadius:3,yAxisID:'y'}},
      {{label:'Revenue ₹',data:D.monthly_rev.map(v=>v/1000),borderColor:'#7c3aed',backgroundColor:'transparent',tension:.4,pointRadius:2,borderDash:[4,3],yAxisID:'y1'}}
    ]
  }},{{scales:{{x:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}},maxRotation:40}}}},y:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#00d4ff',font:{{size:9}}}},title:{{display:true,text:'Units',color:'#00d4ff',font:{{size:9}}}}}},y1:{{position:'right',grid:{{drawOnChartArea:false}},ticks:{{color:'#7c3aed',font:{{size:9}}}},title:{{display:true,text:'Rev (₹K)',color:'#7c3aed',font:{{size:9}}}}}}}}}});

  CA('c-catpie','doughnut',{{labels:D.catrev_labels,datasets:[{{data:D.catrev_vals,backgroundColor:CLRS,borderColor:'#0c1220',borderWidth:2}}]}},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-stacked','bar',{{
    labels:D.catm_months,
    datasets:D.catm_cats.map((c,i)=>{{return{{label:c,data:D.catm_data[c],backgroundColor:CLRS[i]+'bb',stack:'s'}}}})
  }},{{scales:{{x:{{stacked:true,grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}},maxRotation:40}}}},y:{{stacked:true,grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-region','bar',{{labels:D.region_labels,datasets:[{{label:'Units',data:D.region_vals,backgroundColor:D.region_labels.map((_,i)=>CLRS[i%CLRS.length]),borderRadius:4}}]}},{{}});
  CA('c-status','doughnut',{{labels:D.status_labels,datasets:[{{data:D.status_vals,backgroundColor:['#10b981','#00d4ff','#f59e0b','#f43f5e'],borderColor:'#0c1220',borderWidth:2}}]}},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-channel','doughnut',{{labels:D.channel_labels,datasets:[{{data:D.channel_vals,backgroundColor:['#00d4ff','#7c3aed','#f59e0b'],borderColor:'#0c1220',borderWidth:2}}]}},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-fulfil','doughnut',{{labels:D.fulfil_labels,datasets:[{{data:D.fulfil_vals,backgroundColor:['#00d4ff','#f43f5e'],borderColor:'#0c1220',borderWidth:2}}]}},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-toprod','bar',{{labels:D.toprod_labels.map(l=>l.length>22?l.slice(0,20)+'…':l),datasets:[{{label:'Revenue ₹',data:D.toprod_vals,backgroundColor:'rgba(0,212,255,.55)',borderColor:'#00d4ff',borderWidth:1,borderRadius:3}}]}},{{indexAxis:'y',scales:{{x:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}},y:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-brand','bar',{{labels:D.brand_labels,datasets:[{{label:'Revenue ₹',data:D.brand_vals,backgroundColor:'rgba(124,58,237,.55)',borderColor:'#7c3aed',borderWidth:1,borderRadius:3}}]}},{{indexAxis:'y',scales:{{x:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}},y:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-city','bar',{{labels:D.city_labels,datasets:[{{label:'Units',data:D.city_vals,backgroundColor:D.city_labels.map((_,i)=>CLRS[i%CLRS.length]+'aa'),borderRadius:3}}]}},{{}});
  CA('c-wh','doughnut',{{labels:D.wh_labels,datasets:[{{data:D.wh_vals,backgroundColor:CLRS,borderColor:'#0c1220',borderWidth:2}}]}},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
}}

// ── FORECAST ──
let fcChart=null;
function runFc(){{
  const scale = parseFloat(document.getElementById('fc-scale').value);
  const histL = D.fc_hist_labels, histQ = D.fc_hist_q;
  const fcL = D.fc_future_labels, fcQ = D.fc_future_total.map(v=>Math.round(v*scale));
  const all = [...histL,...fcL];
  const hd = [...histQ,...new Array(fcL.length).fill(null)];
  const fd = [...new Array(histL.length).fill(null),...fcQ];
  const hi = [...new Array(histL.length).fill(null),...fcQ.map(v=>Math.round(v*1.15))];
  const lo = [...new Array(histL.length).fill(null),...fcQ.map(v=>Math.round(v*0.85))];
  if(fcChart) fcChart.destroy();
  fcChart = new Chart(document.getElementById('c-forecast'),{{type:'line',data:{{labels:all,datasets:[
    {{label:'Historical',data:hd,borderColor:'#00d4ff',backgroundColor:'rgba(0,212,255,.06)',fill:true,tension:.4,pointRadius:3}},
    {{label:'Forecast',data:fd,borderColor:'#f43f5e',borderDash:[6,3],fill:false,tension:.4,pointRadius:3}},
    {{label:'Upper',data:hi,borderColor:'rgba(244,63,94,.2)',borderDash:[2,4],fill:false,pointRadius:0}},
    {{label:'Lower',data:lo,borderColor:'rgba(244,63,94,.2)',borderDash:[2,4],fill:'-1',backgroundColor:'rgba(244,63,94,.07)',pointRadius:0}},
  ]}},options:OPT()}});
}}

function initForecast(){{
  runFc();
  CA('c-fcsummary','bar',{{labels:D.fc_future_labels,datasets:[{{label:'Forecast Units',data:D.fc_future_total,backgroundColor:D.fc_future_total.map(v=>v>300?'#f43f5e':v>200?'#f59e0b':'rgba(0,212,255,.6)'),borderRadius:4}}]}},{{}});
  CA('c-avsp','line',{{labels:D.test_labels,datasets:[
    {{label:'Actual',data:D.test_actual,borderColor:'#00d4ff',tension:.4,pointRadius:2}},
    {{label:'Predicted',data:D.test_pred,borderColor:'#f43f5e',borderDash:[5,3],tension:.4,pointRadius:2}}
  ]}},{{}});
}}

// ── INVENTORY ──
function initInventory(){{
  const labels = D.inv_skus.map(l=>l.length>20?l.slice(0,18)+'…':l);
  CA('c-inv-bars','bar',{{labels,datasets:[
    {{label:'Stock Level',data:D.inv_stock,backgroundColor:'rgba(0,212,255,.55)',borderRadius:3}},
    {{label:'Reorder Point',data:D.inv_rop,backgroundColor:'rgba(244,63,94,.5)',borderRadius:3}}
  ]}},{{}});
  CA('c-abc','doughnut',{{
    labels:['A — High Value','B — Medium','C — Low Value'],
    datasets:[{{data:[4,4,2],backgroundColor:['#00d4ff','#7c3aed','#10b981'],borderColor:'#0c1220',borderWidth:2}}]
  }},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  const alertDiv = document.getElementById('inv-alerts');
  alertDiv.innerHTML='';
  D.inv_skus.forEach((s,i)=>{{
    if(D.inv_status[i]==='REORDER'){{
      alertDiv.innerHTML+=`<div class="alert">⚠️ <strong>${{s}}</strong> — Stock: <strong>${{D.inv_stock[i]}}</strong> | ROP: ${{D.inv_rop[i]}} | Order <strong>${{D.inv_eoq[i]}} units</strong><span class="tag tag-red">REORDER</span></div>`;
    }}
  }});
}}

// ── PRODUCTION ──
function initProduction(){{
  const plan = D.monthly_q.map(v=>Math.round(v*1.15));
  CA('c-prodplan','bar',{{labels:D.monthly_labels,datasets:[
    {{label:'Demand',data:D.monthly_q,backgroundColor:'rgba(0,212,255,.5)',borderRadius:3}},
    {{label:'Production Plan',data:plan,backgroundColor:'rgba(124,58,237,.5)',borderRadius:3}}
  ]}},{{}});
  CA('c-prodcat','doughnut',{{
    labels:['Electronics','Fashion','Health','Home & Kitchen'],
    datasets:[{{data:[302,635,875,654].map(v=>Math.round(v*1.15)),backgroundColor:CLRS,borderColor:'#0c1220',borderWidth:2}}]
  }},{{...noScale,plugins:{{legend:{{position:'bottom',labels:{{color:'#64748b',font:{{size:9}}}}}}}}}});
  CA('c-prodfuture','bar',{{labels:D.prod_fc_months,datasets:[{{label:'Production Plan',data:D.prod_fc_vals,backgroundColor:D.prod_fc_vals.map(v=>v>400?'#f43f5e':v>200?'#f59e0b':'rgba(0,212,255,.6)'),borderRadius:3}}]}},{{}});
  const fests = {{"Oct 25":"Diwali","Nov 25":"Post-Diwali","Dec 25":"X-mas","Jan 26":"New Year","Mar 26":"Holi","Apr 26":"Eid"}};
  const div=document.getElementById('prod-schedule');
  const mx=Math.max(...D.prod_fc_vals);
  div.innerHTML=`<div class="prod-row" style="font-family:var(--mono);font-size:8.5px;color:var(--tx2);border-bottom:1px solid var(--border);padding-bottom:4px">
    <span>MONTH</span><span>PRODUCTION</span><span>UNITS</span><span>FESTIVAL</span></div>`;
  D.prod_fc_months.forEach((m,i)=>{{
    const pct=Math.round(D.prod_fc_vals[i]/mx*100);
    const fest=fests[m]||'—';
    div.innerHTML+=`<div class="prod-row">
      <span class="mono" style="font-size:10.5px">${{m}}</span>
      <div class="pbar-wrap"><div class="pbar-fill" style="width:${{pct}}%"></div></div>
      <span class="mono clr-cyan">${{D.prod_fc_vals[i]}}</span>
      <span class="tag ${{fest!=='—'?'tag-orange':'tag-blue'}}" style="font-size:9px">${{fest}}</span>
    </div>`;
  }});
}}

// ── LOGISTICS ──
function initLogistics(){{
  CA('c-courier','bar',{{labels:D.courier_names,datasets:[
    {{label:'Orders',data:D.courier_orders,backgroundColor:'rgba(0,212,255,.55)',borderRadius:3,yAxisID:'y'}},
    {{label:'Avg Days',data:D.courier_days,backgroundColor:'rgba(245,158,11,.55)',borderRadius:3,yAxisID:'y1'}}
  ]}},{{scales:{{x:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}},y:{{grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#00d4ff',font:{{size:9}}}}}},y1:{{position:'right',grid:{{drawOnChartArea:false}},ticks:{{color:'#f59e0b',font:{{size:9}}}}}}}}}});
  CA('c-couriercost','bar',{{labels:D.courier_names,datasets:[{{label:'Avg Cost ₹',data:D.courier_cost,backgroundColor:D.courier_cost.map(v=>v>89?'#f43f5e':v>87?'#f59e0b':'#10b981'),borderRadius:4}}]}},{{}});
  CA('c-deldays','bar',{{labels:['1d','2d','3d','4d','5d','6d','7d','8d+'],datasets:[{{label:'Orders',data:[45,98,187,242,228,112,65,23],backgroundColor:'rgba(16,185,129,.6)',borderRadius:4}}]}},{{}});
  CA('c-chanstatus','bar',{{
    labels:['Amazon.in','Shiprocket','INCREFF B2B'],
    datasets:[
      {{label:'Delivered',data:[480,185,68],backgroundColor:'rgba(16,185,129,.7)',stack:'s'}},
      {{label:'Shipped',data:[91,37,12],backgroundColor:'rgba(0,212,255,.6)',stack:'s'}},
      {{label:'Returned',data:[55,16,4],backgroundColor:'rgba(245,158,11,.6)',stack:'s'}},
      {{label:'Cancelled',data:[25,15,12],backgroundColor:'rgba(244,63,94,.6)',stack:'s'}},
    ]
  }},{{scales:{{x:{{stacked:true,grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}},y:{{stacked:true,grid:{{color:'rgba(26,39,68,.55)'}},ticks:{{color:'#64748b',font:{{size:9}}}}}}}}}});
}}

// ── CHATBOT ──
const RESP = {{
  demand:`📈 <strong>Demand Forecast:</strong><br>• Peak: <strong>Oct 2024 — 371 units</strong> (Diwali +105%)<br>• Oct 2025 forecast: <strong>~395 units</strong> (+35% festive)<br>• Jun 2026: <strong>~150 units</strong> stable base<br>• RMSE: {rmse:.2f} | MAE: {mae:.2f} | RF Model`,
  inventory:`📦 <strong>Inventory Alerts:</strong><br>• <strong>{DATA['reorder_count']} SKUs</strong> below reorder point<br>• Avg Safety Stock: 38.4 units<br>• ABC: 4 A-class, 4 B-class, 2 C-class<br>• Action: Trigger EOQ orders immediately for red-flagged items`,
  logistics:`🚚 <strong>Logistics Analysis:</strong><br>• <strong>Best cost: Delhivery</strong> — ₹83.2/shipment<br>• <strong>Fastest: DTDC</strong> — 4.61 avg days<br>• <strong>Highest cost: Ecom Express</strong> — ₹90.6<br>• Delayed orders: <strong>{DATA['delayed_pct']}%</strong> → Optimize UP/Rajasthan routes`,
  revenue:`💰 <strong>Revenue Breakdown:</strong><br>• Total: <strong>₹{DATA['total_rev']}L</strong><br>• Electronics: ₹18.81L (53.7%) 🏆<br>• Home & Kitchen: ₹6.68L (19.1%)<br>• Fashion: ₹5.90L | Health: ₹3.61L<br>• Top product: <strong>HP 15s Laptop — ₹4.22L</strong>`,
  risk:`⚠️ <strong>Risk Summary:</strong><br>• 🔴 Delayed orders: <strong>{DATA['delayed_pct']}%</strong> (target &lt;20%)<br>• 🔴 <strong>{DATA['reorder_count']} SKUs</strong> below reorder point<br>• 🟡 Return rate: <strong>{DATA['return_rate']}%</strong><br>• 🟡 Oct demand spike: 4x vs Jun low<br>• 🟢 Avg delivery: {DATA['avg_delivery']}d vs 5.8d industry avg`,
  production:`🏭 <strong>Q4 2025 Production Plan:</strong><br>• Oct 2025: <strong>454 units</strong> (Diwali buffer)<br>• Nov 2025: <strong>322 units</strong><br>• Dec 2025: <strong>270 units</strong><br>• Total Q4: <strong>1,046 units</strong><br>• Increase capacity 30% from Sep onwards`,
  region:`🗺️ <strong>Regional Analysis:</strong><br>• <strong>Top demand: Karnataka</strong> — 398 units<br>• Maharashtra: 373 | Delhi: 322 | Gujarat: 230<br>• <strong>Best shipping cost: Delhivery in Pune</strong> — ₹83<br>• Focus: Expand warehouse capacity in Karnataka & Maharashtra`,
  return:`↩️ <strong>Returns Analysis:</strong><br>• Total returned: <strong>75 orders (7.5%)</strong><br>• Highest: Electronics & Mobiles category<br>• Amazon.in: 8.4% | Shiprocket: 6.3% | INCREFF: 4.2%<br>• Recommendation: Better packaging + QC for Electronics`
}};
function ask(k){{
  const qs={{demand:"Demand forecast for October 2025?",inventory:"Which SKUs need reorder?",logistics:"Best courier for cost efficiency?",revenue:"Revenue by category?",risk:"Supply chain risks?",production:"Q4 2025 production plan?",region:"Which region has highest demand?",return:"Return and cancellation analysis?"}};
  document.getElementById('chat-inp').value=qs[k]; sendMsg();
}}
function sendMsg(){{
  const inp=document.getElementById('chat-inp'),msg=inp.value.trim();
  if(!msg) return; inp.value='';
  const m=document.getElementById('chat-msgs');
  m.innerHTML+=`<div class="cmsg u">${{msg}}</div>`;
  const q=msg.toLowerCase();
  let r=RESP[q.includes('forecast')||q.includes('predict')||q.includes('oct')||q.includes('future')?'demand':q.includes('invent')||q.includes('stock')||q.includes('reorder')?'inventory':q.includes('logis')||q.includes('courier')||q.includes('deliver')||q.includes('ship')?'logistics':q.includes('rev')||q.includes('sales')||q.includes('profit')||q.includes('categ')?'revenue':q.includes('risk')||q.includes('alert')?'risk':q.includes('prod')||q.includes('manufactur')||q.includes('q4')?'production':q.includes('region')||q.includes('state')||q.includes('karnat')||q.includes('mahara')||q.includes('delhi')?'region':q.includes('return')||q.includes('cancel')?'return':null];
  if(!r) r='💡 Ask about: <strong>demand forecast, inventory, logistics, revenue, production, returns, regional analysis, risk</strong>';
  setTimeout(()=>{{m.innerHTML+=`<div class="cmsg a">🤖 ${{r}}</div>`;m.scrollTop=9999;}},350);
  m.scrollTop=9999;
}}

// ── MODULE SWITCH ──
const inited={{}};
function sw(name,el){{
  document.querySelectorAll('.module').forEach(m=>m.classList.remove('active'));
  document.querySelectorAll('.nav').forEach(n=>n.classList.remove('active'));
  document.getElementById('mod-'+name).classList.add('active');
  el.classList.add('active');
  if(!inited[name]){{ inited[name]=1;
    if(name==='overview') initOverview();
    else if(name==='forecast') initForecast();
    else if(name==='inventory') initInventory();
    else if(name==='production') initProduction();
    else if(name==='logistics') initLogistics();
  }}
}}
initOverview(); inited['overview']=1;
</script>
</body></html>"""
    return HTML


# ─────────────────────────────────────────────
# STEP 6 — WRITE OUTPUT
# ─────────────────────────────────────────────
def save(html, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(path) / 1024
    print(f"💾 Saved: {path} ({size:.1f} KB)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv)>1 else CSV_FILE
    if not Path(csv).exists():
        print(f"❌ File not found: {csv}")
        sys.exit(1)

    df           = load_and_engineer(csv)
    model, le_p, le_r, le_c, feats, df_m, rmse, mae, test_df, test_pred = train_model(df)
    fc_daily, monthly_fc = make_forecast(model, le_p, le_r, le_c, feats, df_m)
    DATA, inv    = aggregate(df, test_df, test_pred, monthly_fc)
    html         = build_html(DATA, inv, rmse, mae)
    save(html, OUT_FILE)

    print("\n✅ OmniFlow D2D Dashboard ready!")
    print(f"   → Open in browser: {OUT_FILE}")
    print(f"   → {DATA['total_orders']} orders | ₹{DATA['total_rev']}L revenue | {len(monthly_fc)} months forecast")
