import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Titanic Survival Predictor",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Load shared artifacts once
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_shared():
    """Load scaler and feature list — shared by both models."""
    try:
        scaler   = joblib.load("titanic_scaler.pkl")
        features = joblib.load("titanic_features.pkl")
        return scaler, features
    except FileNotFoundError as e:
        st.error(
            f"Required file not found: {e}\n\n"
            "Run all cells in the notebook first to generate the .pkl files, "
            "then place them in the same folder as app.py."
        )
        st.stop()


@st.cache_resource
def load_lr():
    try:
        return joblib.load("logistic_model.pkl")
    except FileNotFoundError as e:
        st.error(f"logistic_model.pkl not found: {e}")
        st.stop()


@st.cache_resource
def load_dt():
    try:
        return joblib.load("decision_tree_model.pkl")
    except FileNotFoundError as e:
        st.error(f"decision_tree_model.pkl not found: {e}")
        st.stop()


scaler, FEATURES = load_shared()

# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.title("Titanic Survival Predictor")
st.markdown(
    "Enter the passenger's details, select a model, and click **Predict** to "
    "see whether the model estimates they would have survived."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Model Selection")

MODEL_OPTIONS = {
    "Logistic Regression": {
        "loader"       : load_lr,
        "needs_scaling": True,
        "description"  : (
            "A linear model that estimates survival probability using weighted feature "
            "coefficients. Stable, interpretable, and works well after feature scaling."
        ),
    },
    "Decision Tree": {
        "loader"       : load_dt,
        "needs_scaling": False,
        "description"  : (
            "A tree-based model that splits passengers using a series of if-else rules. "
            "Captures non-linear patterns and requires no feature scaling."
        ),
    },
}

selected_name = st.selectbox(
    "Choose the model to use for prediction:",
    options=list(MODEL_OPTIONS.keys()),
)

st.caption(MODEL_OPTIONS[selected_name]["description"])

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Passenger input form
# ─────────────────────────────────────────────────────────────────────────────

st.subheader("Passenger Information")

col_left, col_right = st.columns(2, gap="large")

with col_left:

    pclass = st.selectbox(
        "Passenger Class",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 — First Class",
            2: "2 — Second Class",
            3: "3 — Third Class",
        }[x],
        help="First class was the most expensive and had the highest survival rate.",
    )

    sex = st.selectbox(
        "Sex",
        options=["female", "male"],
        format_func=str.capitalize,
    )

    age = st.slider(
        "Age (years)",
        min_value=0,
        max_value=80,
        value=30,
        step=1,
        help="Children under 13 are flagged separately by the model.",
    )

    fare = st.number_input(
        "Ticket Fare (GBP)",
        min_value=0.0,
        max_value=520.0,
        value=32.0,
        step=0.50,
        format="%.2f",
        help="Typical ranges: 3rd class 5-20, 2nd class 10-30, 1st class 30-500.",
    )

with col_right:

    embarked = st.selectbox(
        "Port of Embarkation",
        options=["S", "C", "Q"],
        format_func=lambda x: {
            "S": "S — Southampton",
            "C": "C — Cherbourg",
            "Q": "Q — Queenstown",
        }[x],
        help="Cherbourg had a higher share of first-class passengers.",
    )

    sibsp = st.number_input(
        "Siblings / Spouses Aboard",
        min_value=0,
        max_value=8,
        value=0,
        step=1,
        help="Number of siblings or spouses travelling with the passenger.",
    )

    parch = st.number_input(
        "Parents / Children Aboard",
        min_value=0,
        max_value=6,
        value=0,
        step=1,
        help="Number of parents or children travelling with the passenger.",
    )

    # Live-updated derived values
    family_size_live = int(sibsp) + int(parch) + 1
    st.markdown("---")
    st.markdown("**Derived at prediction time**")
    st.markdown(f"- Family size: **{family_size_live}**")
    st.markdown(f"- Travelling alone: **{'Yes' if family_size_live == 1 else 'No'}**")
    st.markdown(f"- Child (age < 13): **{'Yes' if age < 13 else 'No'}**")

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Feature construction — mirrors notebook preprocessing exactly
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_vector(pclass, sex, age, fare, sibsp, parch, embarked):
    """
    Reproduce the exact preprocessing pipeline from the notebook:

      sex_enc         : female=0, male=1
      log_fare        : np.log1p(fare)  — corrects right skew
      family_size     : sibsp + parch + 1
      is_child        : 1 if age < 13 else 0
      has_cabin       : 0  (unknown at inference time)
      age_was_missing : 0  (user provides age directly)
      alone_enc       : 1 if family_size == 1 else 0
      emb_Q           : 1 if embarked == 'Q' else 0
      emb_S           : 1 if embarked == 'S' else 0
      (Cherbourg is the dropped reference category)
    """
    sex_enc         = 0 if sex == "female" else 1
    log_fare        = np.log1p(fare)
    family_size     = int(sibsp) + int(parch) + 1
    is_child        = 1 if age < 13 else 0
    has_cabin       = 0
    age_was_missing = 0
    alone_enc       = 1 if family_size == 1 else 0
    emb_Q           = 1 if embarked == "Q" else 0
    emb_S           = 1 if embarked == "S" else 0

    row = pd.DataFrame([{
        "pclass"          : int(pclass),
        "sex_enc"         : sex_enc,
        "age"             : float(age),
        "log_fare"        : log_fare,
        "has_cabin"       : has_cabin,
        "age_was_missing" : age_was_missing,
        "alone_enc"       : alone_enc,
        "is_child"        : is_child,
        "family_size"     : family_size,
        "emb_Q"           : emb_Q,
        "emb_S"           : emb_S,
    }])

    return row[FEATURES]   # enforce exact column order from training

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────

predict_col, _ = st.columns([1, 2])
with predict_col:
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:

    # Load the selected model
    model_config = MODEL_OPTIONS[selected_name]
    model        = model_config["loader"]()
    input_df     = build_feature_vector(pclass, sex, age, fare, sibsp, parch, embarked)

    # Apply scaling only for Logistic Regression
    if model_config["needs_scaling"]:
        X_in = scaler.transform(input_df)
    else:
        X_in = input_df.values

    prediction    = model.predict(X_in)[0]
    probabilities = model.predict_proba(X_in)[0]
    survived_prob = probabilities[1]
    died_prob     = probabilities[0]

    st.divider()
    st.subheader("Prediction Result")
    st.caption(f"Model used: {selected_name}")

    # ── Main verdict ──────────────────────────────────────────────────────────
    res_col, metric_col = st.columns([3, 1])

    with res_col:
        if prediction == 1:
            st.success("This passenger would have **Survived**.")
            st.markdown(
                f"The {selected_name} model estimates survival "
                f"with a confidence of **{survived_prob:.1%}**."
            )
        else:
            st.error("This passenger would **Not Have Survived**.")
            st.markdown(
                f"The {selected_name} model estimates non-survival "
                f"with a confidence of **{died_prob:.1%}**."
            )

    with metric_col:
        st.metric("Survived",        f"{survived_prob:.1%}")
        st.metric("Did Not Survive", f"{died_prob:.1%}")

    # ── Probability bar chart ─────────────────────────────────────────────────
    st.markdown("**Probability Breakdown**")
    chart_df = pd.DataFrame(
        {"Probability": {"Did Not Survive": died_prob, "Survived": survived_prob}}
    )
    st.bar_chart(chart_df)

    # ── Passenger profile context ─────────────────────────────────────────────
    st.markdown("**Passenger Profile Context**")
    context = []
    if sex == "female":
        context.append("Female passengers had a ~74% survival rate overall.")
    else:
        context.append("Male passengers had only a ~19% survival rate overall.")
    if pclass == 1:
        context.append("First-class passengers survived at ~63%.")
    elif pclass == 2:
        context.append("Second-class passengers survived at ~47%.")
    else:
        context.append("Third-class passengers survived at only ~24%.")
    if age < 13:
        context.append("Children under 13 had the highest survival rate (~58%).")
    fs = int(sibsp) + int(parch) + 1
    if fs == 1:
        context.append("Solo travelers fared slightly below the dataset average.")
    elif 2 <= fs <= 4:
        context.append("Small family size (2-4) is associated with better survival.")
    else:
        context.append("Very large families had lower-than-average survival rates.")
    for line in context:
        st.markdown(f"- {line}")

    # ── Raw feature vector ────────────────────────────────────────────────────
    with st.expander("View raw feature vector sent to the model"):
        disp = input_df.copy()
        disp.index = ["Value"]
        st.dataframe(disp.T.rename(columns={"Value": "Input Value"}),
                     use_container_width=True)
        st.caption(
            "log_fare = log(fare + 1) to correct right skew. "
            "has_cabin = 0 because cabin is unknown at inference time. "
            "emb_Q / emb_S are one-hot encoded from embarked "
            "(Cherbourg is the dropped reference category). "
            f"Scaling applied: {'Yes' if model_config['needs_scaling'] else 'No'}."
        )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "Built with the Titanic dataset from seaborn. "
    "Models trained in the Mohamed Tamer Titanic notebook. "
    "For educational purposes only."
)
