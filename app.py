import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_curve, auc,
    precision_score, recall_score, f1_score, classification_report
)

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide", page_icon="❤️")

# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Heart_Disease_Prediction.csv")
    df.columns = [c.strip() for c in df.columns]
    return df

df = load_data()

CHEST_PAIN_MAP = {"ASY": "Asymptomatic", "NAP": "Non-anginal pain",
                   "ATA": "Atypical angina", "TA": "Typical angina"}
EKG_MAP = {0: "Normal", 1: "ST-T abnormality", 2: "LV hypertrophy"}
SLOPE_MAP = {1: "Upsloping", 2: "Flat", 3: "Downsloping"}
THAL_MAP = {3: "Normal", 6: "Fixed defect", 7: "Reversible defect"}

st.title("❤️ Heart Disease Dashboard")
st.caption("270 patient records — explore the data or train a model to predict heart disease risk.")

tab_explore, tab_predict = st.tabs(["🔍 Explore", "🤖 Predict"])

# =================================================================
# TAB 1: EXPLORE
# =================================================================
with tab_explore:
    st.sidebar.header("Filters")
    age_range = st.sidebar.slider(
        "Age range", int(df["Age"].min()), int(df["Age"].max()),
        (int(df["Age"].min()), int(df["Age"].max()))
    )
    gender_sel = st.sidebar.multiselect(
        "Gender", options=df["Gender"].unique().tolist(),
        default=df["Gender"].unique().tolist()
    )
    cp_sel = st.sidebar.multiselect(
        "Chest pain type", options=df["Chest pain type"].unique().tolist(),
        default=df["Chest pain type"].unique().tolist(),
        format_func=lambda x: CHEST_PAIN_MAP.get(x, x)
    )
    disease_sel = st.sidebar.multiselect(
        "Heart Disease status", options=df["Heart Disease"].unique().tolist(),
        default=df["Heart Disease"].unique().tolist()
    )

    fdf = df[
        (df["Age"].between(*age_range)) &
        (df["Gender"].isin(gender_sel)) &
        (df["Chest pain type"].isin(cp_sel)) &
        (df["Heart Disease"].isin(disease_sel))
    ]

    st.markdown(f"**Showing {len(fdf)} of {len(df)} patients**")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patients", len(fdf))
    presence_pct = (fdf["Heart Disease"].eq("Presence").mean() * 100) if len(fdf) else 0
    c2.metric("Disease Presence %", f"{presence_pct:.1f}%")
    c3.metric("Avg Age", f"{fdf['Age'].mean():.1f}" if len(fdf) else "—")
    c4.metric("Avg Cholesterol", f"{fdf['Cholesterol'].mean():.0f}" if len(fdf) else "—")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Age Distribution by Disease Status")
        fig = px.histogram(
            fdf, x="Age", color="Heart Disease", barmode="overlay",
            nbins=20, opacity=0.7,
            color_discrete_map={"Presence": "#e74c3c", "Absence": "#3498db"}
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Chest Pain Type vs Heart Disease")
        cp_df = fdf.copy()
        cp_df["Chest pain type"] = cp_df["Chest pain type"].map(CHEST_PAIN_MAP)
        fig2 = px.histogram(
            cp_df, x="Chest pain type", color="Heart Disease", barmode="group",
            color_discrete_map={"Presence": "#e74c3c", "Absence": "#3498db"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.subheader("Cholesterol vs Max Heart Rate")
        fig3 = px.scatter(
            fdf, x="Cholesterol", y="Max HR", color="Heart Disease",
            size="Age", hover_data=["Age", "BP"],
            color_discrete_map={"Presence": "#e74c3c", "Absence": "#3498db"}
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Gender Breakdown")
        gender_df = fdf.groupby(["Gender", "Heart Disease"]).size().reset_index(name="Count")
        fig4 = px.bar(
            gender_df, x="Gender", y="Count", color="Heart Disease", barmode="group",
            color_discrete_map={"Presence": "#e74c3c", "Absence": "#3498db"}
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Correlation Heatmap (numeric features)")
    numeric_cols = fdf.select_dtypes(include=[np.number]).columns.tolist()
    corr = fdf[numeric_cols].corr()
    fig5 = px.imshow(
        corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", zmin=-1, zmax=1
    )
    st.plotly_chart(fig5, use_container_width=True)

    with st.expander("View filtered raw data"):
        st.dataframe(fdf, use_container_width=True)

# =================================================================
# TAB 2: PREDICT
# =================================================================
with tab_predict:
    st.subheader("Train a Model")

    left, right = st.columns([1, 2])

    with left:
        model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
        test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
        random_state = st.number_input("Random seed", value=42, step=1)
        train_btn = st.button("Train model", type="primary")

    # Prepare features
    model_df = df.copy()
    model_df["Gender"] = model_df["Gender"].map({"MALE": 1, "FEMALE": 0})
    model_df["Exercise angina"] = model_df["Exercise angina"].map({"PRESENCE": 1, "ABSENCE": 0})
    model_df = pd.get_dummies(model_df, columns=["Chest pain type"], prefix="CP")
    model_df["target"] = model_df["Heart Disease"].map({"Presence": 1, "Absence": 0})
    model_df = model_df.drop(columns=["Heart Disease"])

    feature_cols = [c for c in model_df.columns if c != "target"]
    X = model_df[feature_cols]
    y = model_df["target"]

    if "trained" not in st.session_state:
        st.session_state.trained = False

    if train_btn:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=y
        )
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, random_state=int(random_state))
        else:
            model = RandomForestClassifier(n_estimators=200, random_state=int(random_state))

        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        st.session_state.trained = True
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
        st.session_state.y_test = y_test
        st.session_state.y_pred = y_pred
        st.session_state.y_proba = y_proba
        st.session_state.model_choice = model_choice

    if st.session_state.trained:
        y_test = st.session_state.y_test
        y_pred = st.session_state.y_pred
        y_proba = st.session_state.y_proba

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        with right:
            st.markdown(f"**Model: {st.session_state.model_choice}**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{acc:.2%}")
            m2.metric("Precision", f"{prec:.2%}")
            m3.metric("Recall", f"{rec:.2%}")
            m4.metric("F1 Score", f"{f1:.2%}")

        st.divider()
        col_cm, col_roc = st.columns(2)

        with col_cm:
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(
                cm, text_auto=True, aspect="auto",
                x=["Predicted Absence", "Predicted Presence"],
                y=["Actual Absence", "Actual Presence"],
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_roc:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                          name=f"ROC (AUC = {roc_auc:.3f})"))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                          line=dict(dash="dash", color="gray"),
                                          name="Random"))
            fig_roc.update_layout(xaxis_title="False Positive Rate",
                                   yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

        if st.session_state.model_choice == "Random Forest":
            st.subheader("Feature Importance")
            importances = pd.Series(
                st.session_state.model.feature_importances_,
                index=st.session_state.feature_cols
            ).sort_values(ascending=False)
            fig_imp = px.bar(importances, orientation="h")
            fig_imp.update_layout(showlegend=False, yaxis_title="", xaxis_title="Importance")
            st.plotly_chart(fig_imp, use_container_width=True)

        st.divider()
        st.subheader("🩺 Try a Live Prediction")
        st.caption("Enter patient details to get a prediction from the trained model.")

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            in_age = st.number_input("Age", 20, 90, 55)
            in_gender = st.selectbox("Gender", ["MALE", "FEMALE"])
            in_bp = st.number_input("Resting BP", 80, 220, 130)
        with p2:
            in_chol = st.number_input("Cholesterol", 100, 600, 240)
            in_fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
            in_ekg = st.selectbox("EKG results", [0, 1, 2], format_func=lambda x: EKG_MAP[x])
        with p3:
            in_maxhr = st.number_input("Max Heart Rate", 60, 220, 150)
            in_angina = st.selectbox("Exercise Angina", ["ABSENCE", "PRESENCE"])
            in_stdep = st.number_input("ST Depression", 0.0, 7.0, 1.0, 0.1)
        with p4:
            in_slope = st.selectbox("Slope of ST", [1, 2, 3], format_func=lambda x: SLOPE_MAP[x])
            in_vessels = st.selectbox("Number of vessels fluro", [0, 1, 2, 3])
            in_thal = st.selectbox("Thallium", [3, 6, 7], format_func=lambda x: THAL_MAP[x])

        in_cp = st.selectbox("Chest pain type", ["ASY", "NAP", "ATA", "TA"],
                              format_func=lambda x: CHEST_PAIN_MAP[x])

        if st.button("Predict"):
            row = {
                "Age": in_age,
                "Gender": 1 if in_gender == "MALE" else 0,
                "BP": in_bp,
                "Cholesterol": in_chol,
                "FBS over 120": in_fbs,
                "EKG results": in_ekg,
                "Max HR": in_maxhr,
                "Exercise angina": 1 if in_angina == "PRESENCE" else 0,
                "ST depression": in_stdep,
                "Slope of ST": in_slope,
                "Number of vessels fluro": in_vessels,
                "Thallium": in_thal,
            }
            for cp_opt in ["ASY", "NAP", "ATA", "TA"]:
                row[f"CP_{cp_opt}"] = 1 if in_cp == cp_opt else 0

            input_df = pd.DataFrame([row])
            input_df = input_df.reindex(columns=st.session_state.feature_cols, fill_value=0)
            input_scaled = st.session_state.scaler.transform(input_df)
            pred = st.session_state.model.predict(input_scaled)[0]
            proba = st.session_state.model.predict_proba(input_scaled)[0][1]

            if pred == 1:
                st.error(f"⚠️ Prediction: **Heart Disease Present** (probability: {proba:.1%})")
            else:
                st.success(f"✅ Prediction: **No Heart Disease** (probability of presence: {proba:.1%})")
    else:
        st.info("Click **Train model** on the left to get started.")
