
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

st.set_page_config(page_title="Telco Customer Churn - Streamlit", layout="wide")

@st.cache_data
def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Clean typical Telco churn dataset quirks
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

def plot_bar_counts(df: pd.DataFrame, col: str):
    counts = df[col].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_title(f"Count by {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    st.pyplot(fig)

def build_pipeline(df: pd.DataFrame, target_col: str):
    y = df[target_col].astype(str) if df[target_col].dtype != "int64" else df[target_col]
    X = df.drop(columns=[target_col])

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    # Logistic Regression as a strong baseline
    clf = LogisticRegression(max_iter=200, n_jobs=None)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])

    return X, y, pipe

def evaluate(model, X_test, y_test):
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        # Some models may not support predict_proba
        pass
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, pos_label="Yes" if "Yes" in np.unique(y_test) else 1),
        "Recall": recall_score(y_test, y_pred, pos_label="Yes" if "Yes" in np.unique(y_test) else 1),
        "F1": f1_score(y_test, y_pred, pos_label="Yes" if "Yes" in np.unique(y_test) else 1),
    }
    if proba is not None:
        # If labels are strings, assume "Yes" is positive; otherwise positive=1
        pos_label = "Yes" if "Yes" in np.unique(y_test) else 1
        # Map y_test to binary for ROC AUC if needed
        if isinstance(pos_label, str):
            y_test_bin = (pd.Series(y_test) == pos_label).astype(int)
            metrics["ROC-AUC"] = roc_auc_score(y_test_bin, proba)
        else:
            metrics["ROC-AUC"] = roc_auc_score(y_test, proba)
    cm = confusion_matrix(y_test, y_pred, labels=[
        "No" if "No" in np.unique(y_test) else 0,
        "Yes" if "Yes" in np.unique(y_test) else 1
    ])
    return metrics, cm

# -------------------- UI --------------------
st.title("📊 Telco Customer Churn (Streamlit UI)")
st.caption("Drop your CSV or use the default Telco dataset file name in the same folder.")

with st.sidebar:
    st.header("Settings")
    csv_source = st.radio("How to load data?", ["Use default file name", "Upload CSV"])
    default_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    uploaded_file = None
    if csv_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    target_col = st.text_input("Target column (label)", value="Churn")
    run_train = st.checkbox("Train model", value=True)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

# Load data
if csv_source == "Upload CSV" and uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    try:
        df = load_data(default_path)
    except Exception as e:
        st.error(f"Failed to load default file '{default_path}'. Upload a CSV instead.\n\n{e}")
        st.stop()

st.subheader("1) Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

st.subheader("2) Basic EDA")
left, right = st.columns(2)
with left:
    # Target distribution
    if target_col in df.columns:
        plot_bar_counts(df, target_col)
    else:
        st.warning(f"Target column '{target_col}' not found in data.")
with right:
    # Example: Contract distribution if present
    if "Contract" in df.columns:
        plot_bar_counts(df, "Contract")
    else:
        st.info("Column 'Contract' not found; skip demo chart.")

st.subheader("3) Train / Evaluate (Logistic Regression)")
if run_train:
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' does not exist in the dataset.")
        st.stop()
    X, y, pipe = build_pipeline(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(np.unique(y)) > 1 else None
    )
    with st.spinner("Training model..."):
        pipe.fit(X_train, y_train)
    metrics, cm = evaluate(pipe, X_test, y_test)
    st.write("**Metrics:**")
    st.json({k: round(v, 4) for k, v in metrics.items()})
    st.write("**Confusion Matrix** (rows=true label, cols=pred label):")
    st.write(cm)

    st.subheader("4) Try a Prediction")
    # Build inputs dynamically for a single-row prediction UI
    with st.form("predict_form"):
        st.caption("Enter feature values (or pick a row from the dataset below).")
        inputs = {}
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                val = st.number_input(col, value=float(X[col].median()) if pd.notna(X[col].median()) else 0.0)
                inputs[col] = val
            else:
                # Use the most frequent category as default
                mode = X[col].mode().iloc[0] if not X[col].mode().empty else ""
                val = st.text_input(col, value=str(mode))
                inputs[col] = val
        submitted = st.form_submit_button("Predict")
        if submitted:
            x_df = pd.DataFrame([inputs])
            proba = None
            try:
                proba = pipe.predict_proba(x_df)[:, 1][0]
            except Exception:
                pass
            pred = pipe.predict(x_df)[0]
            st.success(f"Prediction: **{pred}**" + (f" | Probability of churn: **{proba:.3f}**" if proba is not None else ""))

    # Optional: pick a row and auto-fill
    with st.expander("Pick a row from the dataset to auto-fill the form"):
        idx = st.number_input("Row index", min_value=0, max_value=len(X)-1, value=0, step=1)
        st.write("Selected row features:")
        st.write(X.iloc[idx:idx+1])

else:
    st.info("Tick 'Train model' in the sidebar to build a quick baseline model.")
