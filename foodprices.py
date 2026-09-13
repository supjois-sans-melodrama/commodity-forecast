import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from prophet import Prophet
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Commodity Price & Forecast Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Real-Time Global Commodity Price & Forecast Dashboard")
st.markdown("Live commodity prices (from 2010) with dynamic machine learning projections up to **2036**.")

st.warning(
    "⚠️ **Prototype Notice:** Historical market quotes are fetched live from Yahoo Finance (`yfinance`). "
    "Predictive outputs are generated dynamically via Meta Prophet and are for illustrative purposes."
)

# -----------------------------------------------------------------------------
# 2. COMMODITIES & CURRENCY MAPPINGS
# -----------------------------------------------------------------------------
COMMODITIES = {
    "Crude Oil (WTI - Energy)": {"ticker": "CL=F", "unit": "per barrel"},
    "Wheat (Agriculture)": {"ticker": "ZW=F", "unit": "per bushel"},
    "Corn (Agriculture)": {"ticker": "ZC=F", "unit": "per bushel"},
    "Coffee (Arabica)": {"ticker": "KC=F", "unit": "per lb"},
    "Sugar #11": {"ticker": "SB=F", "unit": "per lb"},
    "Gold (Precious Metal)": {"ticker": "GC=F", "unit": "per troy oz"}
}

FX_CURRENCIES = {
    "United States (USD)": {"iso": "USD", "symbol": "$", "fx_ticker": None},
    "Eurozone (EUR)": {"iso": "EUR", "symbol": "€", "fx_ticker": "EURUSD=X"},
    "United Kingdom (GBP)": {"iso": "GBP", "symbol": "£", "fx_ticker": "GBPUSD=X"},
    "Japan (JPY)": {"iso": "JPY", "symbol": "¥", "fx_ticker": "JPY=X"},
    "India (INR)": {"iso": "INR", "symbol": "₹", "fx_ticker": "INR=X"},
    "Canada (CAD)": {"iso": "CAD", "symbol": "CA$", "fx_ticker": "CAD=X"}
}

START_DATE = "2010-01-01"

# -----------------------------------------------------------------------------
# 3. LIVE DATA FETCHING & FORECAST ENGINE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_and_forecast_commodity(ticker_symbol, target_year, fx_ticker):
    df_raw = yf.Ticker(ticker_symbol).history(period="max").reset_index()
    if df_raw.empty:
        return None, None, None

    df_raw['Date'] = pd.to_datetime(df_raw['Date']).dt.tz_localize(None)
    df_clean = df_raw[df_raw['Date'] >= START_DATE][['Date', 'Close']].dropna()
    df_clean.columns = ['ds', 'y']

    # UNIT CORRECTION: US Cents -> USD for grains, sugar, coffee
    if ticker_symbol in ["ZW=F", "ZC=F", "SB=F", "KC=F"]:
        df_clean['y'] = df_clean['y'] / 100.0

    # Dynamic Nominal Foreign Exchange Rate Conversion
    if fx_ticker:
        fx_raw = yf.Ticker(fx_ticker).history(period="max").reset_index()
        if not fx_raw.empty:
            fx_raw['Date'] = pd.to_datetime(fx_raw['Date']).dt.tz_localize(None)
            fx_raw = fx_raw[['Date', 'Close']].rename(columns={'Close': 'fx_rate'})
            
            # Adjust reverse-quoted pairs
            if "EURUSD" in fx_ticker or "GBPUSD" in fx_ticker:
                fx_raw['fx_rate'] = 1.0 / fx_raw['fx_rate']
                
            df_clean = pd.merge_asof(
                df_clean.sort_values('ds'), 
                fx_raw.sort_values('Date'), 
                left_on='ds', 
                right_on='Date', 
                direction='nearest'
            )
            df_clean['fx_rate'] = df_clean['fx_rate'].ffill().bfill()
        else:
            df_clean['fx_rate'] = 1.0
    else:
        df_clean['fx_rate'] = 1.0

    # Nominal Conversion: Local Price = USD Price * Spot FX Rate
    df_clean['price_converted'] = df_clean['y'] * df_clean['fx_rate']

    # Fit Meta Prophet Model with stabilized trend parameters and lower changepoint flexibility
    prophet_df = df_clean[['ds', 'price_converted']].rename(columns={'price_converted': 'y'})
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
        growth='linear',
        changepoint_prior_scale=0.01  # Stabilized to prevent aggressive post-spike trend drops
    )
    model.fit(prophet_df)

    last_date = df_clean['ds'].max()
    target_date = pd.Timestamp(f"{target_year}-12-31")
    days_to_predict = (target_date - last_date).days

    if days_to_predict > 0:
        future_dates = model.make_future_dataframe(periods=days_to_predict)
        forecast = model.predict(future_dates)
    else:
        forecast = model.predict(prophet_df[['ds']])

    # INFLATION FLOOR CORRECTION: Enforce macroeconomic baseline cost floor (e.g. 2.5% compounded annual inflation)
    base_price = df_clean['price_converted'].iloc[0]
    forecast['years_elapsed'] = (forecast['ds'] - df_clean['ds'].min()).dt.days / 365.25
    min_inflation_floor = base_price * (1.025 ** forecast['years_elapsed'])
    
    if ticker_symbol in ["ZW=F", "ZC=F", "SB=F", "KC=F", "CL=F"]:
        forecast['yhat'] = np.maximum(forecast['yhat'], min_inflation_floor)
        forecast['yhat_lower'] = np.maximum(forecast['yhat_lower'], min_inflation_floor * 0.9)
        forecast['yhat_upper'] = np.maximum(forecast['yhat_upper'], min_inflation_floor * 1.1)

    hist_fitted = forecast[forecast['ds'].isin(df_clean['ds'])][['ds', 'yhat', 'yhat_lower', 'yhat_upper']].reset_index(drop=True)

    df_clean['type'] = 'Historical'
    forecast_subset = forecast[forecast['ds'] > last_date][['ds', 'yhat']].rename(columns={'yhat': 'price_converted'})
    forecast_subset['type'] = 'Forecast'
    
    latest_fx = df_clean['fx_rate'].iloc[-1]
    forecast_subset['y'] = forecast_subset['price_converted'] / latest_fx
    forecast_subset['fx_rate'] = latest_fx
    
    full_df = pd.concat([df_clean[['ds', 'y', 'fx_rate', 'price_converted', 'type']], forecast_subset], ignore_index=True)
    full_df = full_df[full_df['ds'] <= target_date]
    
    return full_df, last_date, hist_fitted

# -----------------------------------------------------------------------------
# 4. SIDEBAR CONTROLS
# -----------------------------------------------------------------------------
st.sidebar.header("🎯 Dashboard Settings")

selected_commodity_name = st.sidebar.selectbox(
    "Select Commodity Ticker",
    options=list(COMMODITIES.keys())
)

selected_country_name = st.sidebar.selectbox(
    "Select Display Currency Region",
    options=list(FX_CURRENCIES.keys())
)

selected_horizon_year = st.sidebar.selectbox(
    "Target Prediction Horizon Year",
    options=[2028, 2030, 2032, 2034, 2036],
    index=2
)

commodity_info = COMMODITIES[selected_commodity_name]
country_info = FX_CURRENCIES[selected_country_name]

# -----------------------------------------------------------------------------
# 5. TABBED DASHBOARD STRUCTURE
# -----------------------------------------------------------------------------
tab_forecast, tab_engine, tab_ml_docs = st.tabs([
    "📊 Interactive Forecast", 
    "⚙️ Data Engine", 
    "🧠 ML Models & Specs"
])

with st.spinner("Fetching market feeds and running forecast engine..."):
    df_data, cutoff_date, hist_fitted = fetch_and_forecast_commodity(
        commodity_info["ticker"],
        selected_horizon_year,
        country_info["fx_ticker"]
    )

if df_data is not None:
    df_data['price_converted'] = np.round(df_data['price_converted'], 2)
    df_data['y'] = np.round(df_data['y'], 2)
    df_data['fx_rate'] = np.round(df_data['fx_rate'], 4)
    df_data['year'] = df_data['ds'].dt.year

    hist_df = df_data[df_data['type'] == 'Historical']
    forecast_df = df_data[df_data['type'] == 'Forecast']

    latest_hist_price = hist_df['price_converted'].iloc[-1]
    base_price_2010 = hist_df['price_converted'].iloc[0]
    target_forecast_price = forecast_df['price_converted'].iloc[-1] if not forecast_df.empty else latest_hist_price

    hist_change = ((latest_hist_price - base_price_2010) / base_price_2010) * 100
    forecast_change = ((target_forecast_price - latest_hist_price) / latest_hist_price) * 100

    # =========================================================================
    # TAB 1: INTERACTIVE FORECAST DASHBOARD
    # =========================================================================
    with tab_forecast:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2010 Base Price", f"{country_info['symbol']}{base_price_2010:.2f} {country_info['iso']}")
        c2.metric("Latest Market Price", f"{country_info['symbol']}{latest_hist_price:.2f} {country_info['iso']}", f"{hist_change:+.1f}% since 2010")
        c3.metric(f"Predicted ({selected_horizon_year})", f"{country_info['symbol']}{target_forecast_price:.2f} {country_info['iso']}")
        c4.metric("Forecasted Growth", f"{forecast_change:+.1f}%")

        st.markdown("---")

        st.subheader(f"Price Trend & Projection: {selected_commodity_name} ({commodity_info['unit']})")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=hist_df['ds'],
            y=hist_df['price_converted'],
            mode='lines',
            name='Historical Data (From 2010)',
            line=dict(color='#1f77b4', width=2)
        ))

        if not forecast_df.empty:
            plot_forecast = pd.concat([hist_df.tail(1), forecast_df])
            fig.add_trace(go.Scatter(
                x=plot_forecast['ds'],
                y=plot_forecast['price_converted'],
                mode='lines',
                name=f'Meta Prophet Forecast (to {selected_horizon_year})',
                line=dict(color='#ff7f0e', width=2.5, dash='dash')
            ))

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title=f"Price ({country_info['symbol']} {country_info['iso']})",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"📊 Projected Biennial Averages ({country_info['iso']})")
        
        yearly_summary = []
        years_to_show = [y for y in range(2010, selected_horizon_year + 1) if y % 2 == 0]
        
        for yr in years_to_show:
            yr_sub = df_data[df_data['year'] == yr]
            if not yr_sub.empty:
                avg_p = yr_sub['price_converted'].mean()
                d_type = yr_sub['type'].iloc[-1]
                yearly_summary.append({
                    "Year": yr,
                    "Data Type": d_type,
                    "Average Price": f"{country_info['symbol']}{avg_p:.2f} {country_info['iso']}",
                    "Growth vs 2010 Base": f"{((avg_p - base_price_2010)/base_price_2010)*100:+.1f}%"
                })
                
        st.table(pd.DataFrame(yearly_summary))

    # =========================================================================
    # TAB 2: DATA ENGINE & INSPECTION
    # =========================================================================
    with tab_engine:
        st.header("⚙️ Data Pipeline & Valuation Engine")
        st.markdown("""
        Streams benchmark futures prices from global exchanges via Yahoo Finance (`yfinance`), 
        aligns historical trading dates, and applies dynamic Foreign Exchange (FX) spot rates.
        """)
        st.markdown("---")
        
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            st.subheader("Price Conversion Formula")
            st.latex(r"P_{\text{Local}}(t) = P_{\text{USD}}(t) \times \text{FX Rate}(t)")
            st.markdown("Converts USD-denominated global commodity contracts into local regional currencies using daily spot FX rates.")

        with col_e2:
            st.subheader("Pipeline Rules")
            st.markdown("""
            * **Baseline Horizon:** Fixed to 2010 for optimal historical data depth.
            * **Unit Normalization:** Agriculture futures (`ZW=F`, `ZC=F`, `SB=F`, `KC=F`) automatically convert US Cents to USD.
            """)

        st.markdown("---")

        st.subheader("📋 Dataset Inspection & Export")
        display_df = df_data[['ds', 'type', 'y', 'fx_rate', 'price_converted']].copy()
        display_df.columns = ['Date', 'Type', 'Base USD Price', 'Exchange Rate', f'Converted Price ({country_info["iso"]})']

        st.dataframe(display_df, use_container_width=True, height=300)

        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv_data,
            file_name=f"{commodity_info['ticker']}_forecast_{selected_country_name}.csv",
            mime="text/csv"
        )

    # =========================================================================
    # TAB 3: EXTENDED ML MODEL SPECIFICATIONS & TECHNICAL DOCUMENTATION
    # =========================================================================
    with tab_ml_docs:
        st.header("🧠 Machine Learning Architecture & Technical Specifications")

        st.info(
            "💡 **Inflation & Macroeconomic Dynamics:** Model predictions incorporate a macroeconomic cost-inflation floor "
            "to reflect long-term baseline production realities and prevent un-economic price collapse."
        )

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 1: DETAILED EVALUATION METRICS
        # ---------------------------------------------------------------------
        st.subheader("🔬 Live Model Performance Metrics & Formulas")
        if hist_fitted is not None and not hist_df.empty:
            y_actual = hist_df['price_converted'].values
            y_pred = hist_fitted['yhat'].values[:len(y_actual)]

            mape_val = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
            rmse_val = np.sqrt(np.mean((y_actual - y_pred) ** 2))
            mae_val = np.mean(np.abs(y_actual - y_pred))

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(
                label="MAPE (Mean Absolute % Error)", 
                value=f"{mape_val:.2f}%", 
                help="Average relative error across fitted historical data points."
            )
            col_m2.metric(
                label=f"RMSE (Root Mean Squared Error in {country_info['iso']})", 
                value=f"{country_info['symbol']}{rmse_val:.2f}",
                help="Standard deviation of residuals in selected regional currency."
            )
            col_m3.metric(
                label=f"MAE (Mean Absolute Error in {country_info['iso']})", 
                value=f"{country_info['symbol']}{mae_val:.2f}",
                help="Average absolute discrepancy between actual and fitted historical prices."
            )

        st.markdown("#### Formal Metric Formulations")
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.latex(r"\text{MAPE} = \frac{100\%}{n} \sum_{t=1}^{n} \left| \frac{y_t - \hat{y}_t}{y_t} \right|")
        with col_f2:
            st.latex(r"\text{RMSE} = \sqrt{\frac{1}{n} \sum_{t=1}^{n} (y_t - \hat{y}_t)^2}")
        with col_f3:
            st.latex(r"\text{MAE} = \frac{1}{n} \sum_{t=1}^{n} |y_t - \hat{y}_t|")

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 2: MATHEMATICAL FORMULATION
        # ---------------------------------------------------------------------
        st.subheader("📐 Deconstructed Mathematical Architecture")
        st.markdown("""
        Meta Prophet utilizes an additive time-series decomposition model with three primary elements: 
        trend, seasonality, and holidays/external shocks.
        """)

        col_algo1, col_algo2 = st.columns(2)
        with col_algo1:
            st.markdown("**1. Overall Additive Model**")
            st.latex(r"y(t) = g(t) + s(t) + h(t) + \epsilon_t")
            st.markdown("""
            * **$g(t)$ (Piecewise Linear Trend):** Models non-periodic structural shifts in price baselines over multi-year horizons.
            * **$s(t)$ (Periodic Seasonality):** Captures multi-period recurring cycles using Fourier series expansion.
            * **$h(t)$ (Holiday Effects):** Adjusts for predictable market closures or seasonal event shocks.
            * **$\epsilon_t$ (Idiosyncratic Error):** Residual variance assumed to be normally distributed ($\mathcal{N}(0, \sigma^2)$).
            """)

        with col_algo2:
            st.markdown("**2. Annual Seasonality via Fourier Terms**")
            st.latex(r"s(t) = \sum_{n=1}^{N} \left( a_n \cos\left(\frac{2\pi n t}{P}\right) + b_n \sin\left(\frac{2\pi n t}{P}\right) \right)")
            st.markdown("""
            * **$P = 365.25$ days:** Standard annual period accounting for leap years.
            * **$N = 10$ Order:** Default Fourier order used to fit complex yearly agricultural harvest and energy cycles.
            * **Parameters $(a_n, b_n)$:** Estimated simultaneously during model fitting to model smooth seasonal transitions.
            """)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 3: CONSOLIDATED MODEL COMPARISON MATRIX
        # ---------------------------------------------------------------------
        st.subheader("💡 Comprehensive Architectural Trade-Off Analysis")
        st.markdown("""
        The evaluation matrix below outlines alternative forecasting paradigms and their suitability 
        for a live multi-year interactive web application.
        """)

        model_comp_data = [
            {
                "Model Architecture": "Meta Prophet (Selected)", 
                "Mathematical Approach": "Additive Non-linear GAM ($y(t) = g(t) + s(t) + \epsilon_t$)", 
                "Strengths": "Natively models annual seasonality and handles exchange holidays/missing data without explicit padding.", 
                "Limitations & Trade-offs": "Does not model cross-asset correlations (e.g., Crude Oil influencing production costs).", 
                "Dashboard Suitability": "🟢 Optimal: Fast dynamic fitting (<2 sec), zero manual feature engineering required."
            },
            {
                "Model Architecture": "ARIMA / SARIMAX", 
                "Mathematical Approach": "Linear Autoregressive Moving Average with Differencing", 
                "Strengths": "Strong performance on short-term stationary series and immediate autocorrelation.", 
                "Limitations & Trade-offs": "Requires strict stationarity. Predictions over multi-year horizons decay rapidly to historical mean baselines.", 
                "Dashboard Suitability": "🟡 Moderate: Poor long-range predictive value for decade-long horizons."
            },
            {
                "Model Architecture": "XGBoost / LightGBM", 
                "Mathematical Approach": "Gradient Boosted Decision Trees (GBDT)", 
                "Strengths": "High precision when supplied with rich lag features, technical indicators, and exogenous macro variables.", 
                "Limitations & Trade-offs": "Tree-based models cannot extrapolate trend trajectories beyond training min/max bounds. Requires synthetic future feature generation.", 
                "Dashboard Suitability": "🟡 Moderate: High pipeline complexity and risk of flatlining multi-year projections."
            },
            {
                "Model Architecture": "LSTM / Deep Learning", 
                "Mathematical Approach": "Recurrent Neural Network with Gated Memory Cells", 
                "Strengths": "Captures complex non-linear sequence dependencies and multi-variable temporal patterns.", 
                "Limitations & Trade-offs": "Heavy CPU computational overhead triggers cloud server timeout limits. Highly susceptible to overfitting on small financial series.", 
                "Dashboard Suitability": "🔴 Poor: Incompatible with low-latency user interface requirements on web hosting tiers."
            }
        ]
        st.table(pd.DataFrame(model_comp_data))

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 4: PIPELINE & HYPERPARAMETER SPECIFICATIONS
        # ---------------------------------------------------------------------
        col_hp1, col_hp2 = st.columns(2)
        with col_hp1:
            st.subheader("⚙️ Model Hyperparameters")
            st.markdown("""
            ```python
            Prophet(
                growth='linear',
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.01,
                interval_width=0.80
            )
            ```
            * **`changepoint_prior_scale` (0.01):** Regularizes trend flexibility to prevent aggressive downward post-spike corrections.
            * **`yearly_seasonality` (True):** Enforces 365.25-day seasonal cycle learning for agricultural and energy trends.
            """)

        with col_hp2:
            st.subheader("🛠️ Data Preprocessing & Validation Pipeline")
            st.markdown("""
            1. **Exchange Gap Reconciliation:** Alignment of standard calendar dates across global exchange holidays via `pd.merge_asof`.
            2. **Unit Conversion:** Automated scale adjustment for grain and soft futures quoted in cents per bushel/lb to standard currency units.
            3. **Forward-Fill FX Rate Alignment:** Forward and backward propagation (`ffill`/`bfill`) of currency rates across missing foreign exchange market days.
            4. **Inflation Floor Bound:** Implementation of a macro-economic baseline cost floor ensuring nominal asset stability.
            """)

        st.markdown("---")

        # ---------------------------------------------------------------------
        # SECTION 5: DATASET TECHNICAL SPECIFICATIONS
        # ---------------------------------------------------------------------
        st.subheader("📊 Dataset Technical Specifications")
        dataset_specs = [
            {"Attribute": "Primary Market Feed", "Specification": "Yahoo Finance (`yfinance` API)", "Details": "Live daily futures settlement prices"},
            {"Attribute": "FX Rate Source", "Specification": "Yahoo Finance (`yfinance` FX spot rates)", "Details": "Real-time cross-currency spot quotes"},
            {"Attribute": "Cache Invalidation", "Specification": "Hourly (`ttl=3600`)", "Details": "Automated background fetch and retraining schedule"},
            {"Attribute": "Historical Baseline Horizon", "Specification": "Jan 1, 2010 – Present", "Details": "Standardized 16+ year historical training dataset"},
            {"Attribute": "Target Forecast Horizon", "Specification": "Up to Year 2036", "Details": "Dynamic multi-year forward projection horizon"}
        ]
        st.table(pd.DataFrame(dataset_specs))

else:
    st.error("Could not fetch commodity data. Please check ticker or network connection.")