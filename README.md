# ⟳ OmniFlow-D2D — DEPLOYMENT READY
### End-to-End Data Science Application for Amazon India Supply Chain Intelligence

> Data-to-Decision: From raw Amazon sales data to AI-powered supply chain insights

---

## 📌 Dataset
Unified dataset merging two Kaggle sources:
- **Amazon India Sales 2025** — `allenclose/amazon-india-sales-2025-analysis`
- **Amazon Sales 2025 (Shiprocket/INCREFF)** — `zahidmughal2343/amazon-sales-2025`

Single cleaned dataset with 5,000 orders across 20 products, 10 regions, Jan 2024–Feb 2025.

---

## 🚀 Modules

| Module | Description |
|--------|-------------|
| 📊 Overview Dashboard | KPIs, time-series trends, category & region breakdown |
| 📈 Demand Forecasting | SARIMAX/ARIMA forecasting with RMSE/NRMSE evaluation |
| 🏭 Inventory Optimization | Safety Stock, Reorder Point, EOQ per product |
| ⚙️ Production Planning | Demand vs capacity, shortage alerts, utilization |
| 🚚 Logistics Optimization | Region routing, courier analysis, cost simulation |
| 🤖 AI Decision Intelligence | Context-aware chatbot using all module outputs |

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** — single-app UI
- **Pandas / NumPy** — data processing
- **Statsmodels (SARIMAX)** — time series forecasting
- **Plotly** — interactive visualizations
- **Scikit-learn** — fallback regression

---

## ⚡ Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run app.py
```

---

## 📁 Structure

```
omniflow_d2d/
├── app.py              # Main Streamlit application (all modules)
├── requirements.txt
└── README.md
```

---

## 🔑 Key Design Decisions
- **ONE dataset** across ALL 6 modules — no splitting
- **ONE app file** — modular functions, single Streamlit app
- Forecast results cached with `@st.cache_data` for performance
- Chatbot is context-aware — pulls live computed values from all modules
- Dark industrial UI with Space Mono + Sora typography

---

## 💬 Chatbot Example Queries
- *"Which product has highest demand next month?"*
- *"What is the reorder point for Monitor?"*
- *"Which region needs more logistics support?"*
- *"Show KPI summary"*
- *"EOQ for Smart Watch"*
- *"Which courier performs best?"*
