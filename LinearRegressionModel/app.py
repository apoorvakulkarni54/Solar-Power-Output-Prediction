import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import os


# ============================================================
# MODEL SETUP
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model():
    model_path = os.path.join(BASE_DIR, "model.pkl")
    return joblib.load(model_path)


# Load trained model
model = load_model()


@st.cache_data
def load_data():
    try:
        return pd.read_csv(
            os.path.join(BASE_DIR, "solar_power_output.csv")
        )
    except FileNotFoundError:
        return None


df = load_data()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(irr):
    prediction = model.predict(
        pd.DataFrame(
            {
                "solar_irradiance": [irr]
            }
        )
    )[0]

    return max(float(prediction), 0.0)


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="Solar Power Predictor",
    page_icon="☀️",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .hero {
        background: linear-gradient(
            135deg,
            #ff9a3c 0%,
            #ffcf5c 50%,
            #4facfe 100%
        );
        padding: 20px 25px;
        border-radius: 18px;
        color: white;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 20px;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.3rem;
        color: white;
    }

    .hero p {
        margin: 5px 0 0 0;
        font-size: 1.1rem;
    }

    .prediction-card {
        background: #fff8ec;
        border-left: 6px solid #ff9a3c;
        padding: 15px 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        text-align: center;
    }

    .prediction-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e8740c;
        margin-bottom: 5px;
    }

    .prediction-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #e8740c;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HERO BANNER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div style="font-size:55px;">☀️</div>
        <div>
            <h1>Solar Power Output Predictor</h1>
            <p>
                Simple Linear Regression · Predict panel output from sunlight
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3 = st.tabs(
    [
        "🔮 Predict",
        "📊 Data Insights",
        "📁 Batch Predict"
    ]
)


# ============================================================
# TAB 1 — PREDICTION
# ============================================================

with tab1:

    st.markdown(
        "#### ✍️ Enter values and click **Predict**"
    )

    # --------------------------------------------------------
    # INPUT FORM
    # --------------------------------------------------------

    with st.form("predict_form"):

        f1, f2 = st.columns(2)

        irr = f1.number_input(
            "☀️ Solar irradiance (W/m²)",
            min_value=0.0,
            max_value=1500.0,
            value=600.0,
            step=10.0
        )

        panels = f2.number_input(
            "🔲 Number of panels",
            min_value=1,
            max_value=100,
            value=10
        )

        sun_hours = f1.number_input(
            "🕒 Peak sun hours per day",
            min_value=1.0,
            max_value=12.0,
            value=5.5,
            step=0.5
        )

        price = f2.number_input(
            "💰 Electricity price (₹ per kWh)",
            min_value=1.0,
            max_value=30.0,
            value=8.0,
            step=0.5
        )

        submitted = st.form_submit_button(
            "⚡ Predict Solar Power Output",
            type="primary",
            width="stretch"
        )


    # --------------------------------------------------------
    # SAVE INPUTS AFTER BUTTON CLICK
    # --------------------------------------------------------

    if submitted:

        st.session_state["inputs"] = (
            irr,
            panels,
            sun_hours,
            price
        )


    # --------------------------------------------------------
    # BEFORE PREDICT
    # --------------------------------------------------------

    if "inputs" not in st.session_state:

        st.info(
            "👆 Enter the values above and press **Predict** "
            "to see the results."
        )


    # --------------------------------------------------------
    # AFTER PREDICT
    # --------------------------------------------------------

    else:

        irr, panels, sun_hours, price = (
            st.session_state["inputs"]
        )

        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        pred = predict(irr)

        total_kw = pred * panels / 1000

        daily_kwh = total_kw * sun_hours


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-title">
                    Predicted Solar Power Output
                </div>

                <div class="prediction-value">
                    {pred:.2f} W
                </div>

                <div>
                    per panel at {irr:.0f} W/m²
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


        # ====================================================
        # TRAINING RANGE WARNING
        # ====================================================

        if irr < 104 or irr > 999:

            st.warning(
                "⚠️ This irradiance is outside the training range "
                "(104–999 W/m²), so the prediction is an estimate "
                "beyond the data."
            )


        # ====================================================
        # METRICS
        # ====================================================

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "⚡ Output per panel",
            f"{pred:.1f} W"
        )

        c2.metric(
            "🔋 Total system power",
            f"{total_kw:.2f} kW"
        )

        c3.metric(
            "📅 Daily energy",
            f"{daily_kwh:.2f} kWh"
        )

        c4.metric(
            "💰 Monthly savings",
            f"₹{daily_kwh * 30 * price:,.0f}"
        )


        # ====================================================
        # SOLAR IMAGE + GAUGE
        # ====================================================

        left, right = st.columns(2)


        # ----------------------------------------------------
        # SOLAR IMAGE
        # ----------------------------------------------------

        with left:

            st.markdown(
                "#### 🌤️ Live Solar Scene"
            )

            # Real solar panel image
            st.image(
                "https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1200&q=80",
                use_container_width=True
            )


            if irr < 300:

                st.info(
                    "🌥️ Low sunlight: cloudy or early morning conditions."
                )

            elif irr < 700:

                st.warning(
                    "⛅ Moderate sunlight: decent generation."
                )

            else:

                st.success(
                    "☀️ Strong sunlight: peak generation!"
                )


        # ----------------------------------------------------
        # OUTPUT GAUGE
        # ----------------------------------------------------

        with right:

            st.markdown(
                "#### 🎯 Output Gauge"
            )

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=pred,

                    number={
                        "suffix": " W"
                    },

                    delta={
                        "reference": 283,
                        "suffix": " vs avg"
                    },

                    gauge={
                        "axis": {
                            "range": [0, 600]
                        },

                        "bar": {
                            "color": "#e8740c"
                        },

                        "steps": [
                            {
                                "range": [0, 150],
                                "color": "#dbeafe"
                            },

                            {
                                "range": [150, 350],
                                "color": "#fde68a"
                            },

                            {
                                "range": [350, 600],
                                "color": "#fdba74"
                            }
                        ]
                    }
                )
            )

            gauge.update_layout(
                height=320,
                margin=dict(
                    t=30,
                    b=10
                )
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )


        # ====================================================
        # REGRESSION GRAPH
        # ====================================================

        st.markdown(
            "#### 📈 Where your prediction sits on the regression line"
        )


        xs = np.linspace(
            0,
            1200,
            100
        )


        line = model.predict(
            pd.DataFrame(
                {
                    "solar_irradiance": xs
                }
            )
        )


        fig = go.Figure()


        # Historical data

        if df is not None:

            fig.add_trace(
                go.Scatter(
                    x=df["solar_irradiance"],
                    y=df["solar_power_output"],
                    mode="markers",
                    name="Historical data",

                    marker=dict(
                        color="#60a5fa",
                        size=6,
                        opacity=0.5
                    )
                )
            )


        # Regression line

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=line,
                mode="lines",
                name="Regression line",

                line=dict(
                    color="red",
                    width=3
                )
            )
        )


        # User's prediction

        fig.add_trace(
            go.Scatter(
                x=[irr],
                y=[pred],
                mode="markers",
                name="Your input",

                marker=dict(
                    color="gold",
                    size=20,
                    symbol="star",

                    line=dict(
                        color="black",
                        width=2
                    )
                )
            )
        )


        fig.update_layout(
            xaxis_title="Solar irradiance (W/m²)",
            yaxis_title="Output (W)",
            height=420
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ====================================================
        # GENERATION THROUGH THE DAY
        # ====================================================

        st.markdown(
            "#### 🕒 Estimated generation through the day"
        )


        hours = np.arange(
            6,
            19
        )


        hourly_irr = (
            irr *
            np.sin(
                np.pi *
                (hours - 6) /
                12
            )
        )


        hourly_out = [
            predict(h) * panels / 1000
            for h in hourly_irr
        ]


        day = px.area(
            x=hours,
            y=hourly_out,

            labels={
                "x": "Hour of day",
                "y": "System output (kW)"
            },

            color_discrete_sequence=[
                "#ff9a3c"
            ]
        )


        day.update_layout(
            height=320
        )


        st.plotly_chart(
            day,
            use_container_width=True
        )


        # ====================================================
        # MODEL EQUATION
        # ====================================================

        st.markdown(
            "### 🧮 Model Equation"
        )

        st.markdown(
            f"**Output = {model.intercept_:.3f} + "
            f"{model.coef_[0]:.4f} × Irradiance**"
        )

        st.markdown(
            f"Every extra **100 W/m²** of sunlight adds about "
            f"**{model.coef_[0] * 100:.0f} W** per panel."
        )


# ============================================================
# TAB 2 — DATA INSIGHTS
# ============================================================

with tab2:

    if df is None:

        st.error(
            "Place `solar_power_output.csv` next to app.py "
            "to see data insights."
        )

    else:

        k1, k2, k3, k4 = st.columns(4)


        k1.metric(
            "Records",
            len(df)
        )


        k2.metric(
            "Avg output",
            f"{df['solar_power_output'].mean():.0f} W"
        )


        k3.metric(
            "Max output",
            f"{df['solar_power_output'].max():.0f} W"
        )


        k4.metric(
            "Correlation (irr ↔ output)",
            f"{df['solar_irradiance'].corr(df['solar_power_output']):.3f}"
        )


        # ----------------------------------------------------
        # UNIVARIATE
        # ----------------------------------------------------

        st.markdown(
            "#### 📊 Univariate: distribution of a column"
        )


        col = st.selectbox(
            "Choose a column",
            df.columns,
            index=min(4, len(df.columns) - 1)
        )


        histogram = px.histogram(
            df,
            x=col,
            nbins=25,
            marginal="box",
            color_discrete_sequence=[
                "#ff9a3c"
            ]
        )


        st.plotly_chart(
            histogram,
            use_container_width=True
        )


        # ----------------------------------------------------
        # BIVARIATE
        # ----------------------------------------------------

        st.markdown(
            "#### 🔗 Bivariate: feature vs output"
        )


        feature_columns = [
            c for c in df.columns
            if c != "solar_power_output"
        ]


        feat = st.selectbox(
            "Choose a feature",
            feature_columns,
            index=min(2, len(feature_columns) - 1)
        )


        scatter = px.scatter(
            df,
            x=feat,
            y="solar_power_output"
        )


        st.plotly_chart(
            scatter,
            use_container_width=True
        )


        # ----------------------------------------------------
        # MULTIVARIATE
        # ----------------------------------------------------

        st.markdown(
            "#### 🌡️ Multivariate: correlation heatmap"
        )


        numeric_df = df.select_dtypes(
            include=np.number
        )


        heatmap = px.imshow(
            numeric_df.corr().round(2),
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )


        st.plotly_chart(
            heatmap,
            use_container_width=True
        )


        # ----------------------------------------------------
        # KEY INSIGHTS
        # ----------------------------------------------------

        st.markdown(
            "### 💡 Key Insights"
        )


        st.markdown(
            """
            - Solar irradiance is the main driver of output.
            - Output is approximately proportional to irradiance.
            - Temperature, humidity and wind speed have comparatively less effect.
            """
        )


# ============================================================
# TAB 3 — BATCH PREDICTION
# ============================================================

with tab3:

    st.write(
        "Upload a CSV with a **`solar_irradiance`** column."
    )


    file = st.file_uploader(
        "CSV file",
        type="csv"
    )


    if file is not None:

        data = pd.read_csv(file)


        if "solar_irradiance" not in data.columns:

            st.error(
                "Column `solar_irradiance` not found."
            )

        else:

            data["predicted_output"] = (
                model.predict(
                    data[["solar_irradiance"]]
                )
            )

            data["predicted_output"] = (
                data["predicted_output"].clip(lower=0)
            )


            st.success(
                f"✅ Predicted {len(data)} rows"
            )


            st.dataframe(
                data,
                use_container_width=True
            )


            batch_chart = px.scatter(
                data,
                x="solar_irradiance",
                y="predicted_output",
                color="predicted_output",
                color_continuous_scale="YlOrRd"
            )


            st.plotly_chart(
                batch_chart,
                use_container_width=True
            )


            st.download_button(
                "⬇️ Download predictions",

                data.to_csv(
                    index=False
                ).encode(),

                "predictions.csv",

                "text/csv"
            )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Built with Streamlit · Simple Linear Regression model"
)
