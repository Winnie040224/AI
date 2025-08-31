
import os
import io
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Telco Churn — Full ML UI", layout="wide")

# ----------------- Utils -----------------
@st.cache_data
def load_data(csv_src):
    if hasattr(csv_src, "read"):
        df = pd.read_csv(csv_src)
    else:
        df = pd.read_csv(csv_src)
    # common telco cleanup
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    return df

def ensure_binary_series(y):
    """Map typical 'Yes'/'No' to 1/0. If already numeric, keep it."""
    y_s = pd.Series(y).copy()
    uniques = pd.unique(y_s)
    if set(map(str, uniques)) == set(["Yes", "No"]) or set(uniques) == set(["Yes", "No"]):
        y_s = (y_s == "Yes").astype(int)
    elif y_s.dtype == "bool":
        y_s = y_s.astype(int)
    return y_s

def iqr_outlier_summary(df, num_cols):
    rows = []
    for col in num_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        below = (df[col] < lower).sum()
        above = (df[col] > upper).sum()
        rows.append({
            "column": col,
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lower, "upper_bound": upper,
            "below_count": int(below), "above_count": int(above),
            "total_outliers": int(below + above)
        })
    return pd.DataFrame(rows).sort_values("total_outliers", ascending=False)

def clip_outliers_iqr(df, num_cols):
    df2 = df.copy()
    for col in num_cols:
        q1 = df2[col].quantile(0.25)
        q3 = df2[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df2[col] = df2[col].clip(lower, upper)
    return df2

def build_preprocessor(X, scale_numeric=True, encode_categorical=True):
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    transformers = []
    if scale_numeric and numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    elif numeric_cols:
        # pass-through numeric when not scaling
        transformers.append(("num", "passthrough", numeric_cols))

    if encode_categorical and categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    elif categorical_cols:
        transformers.append(("cat", "passthrough", categorical_cols))

    if transformers:
        preprocessor = ColumnTransformer(transformers=transformers)
    else:
        preprocessor = "passthrough"
    return preprocessor, numeric_cols, categorical_cols

def evaluate(model, X_test, y_test, positive_label=1):
    proba = None
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass
    y_pred = model.predict(X_test)

    if isinstance(y_test, pd.Series):
        yt = y_test
    else:
        yt = pd.Series(y_test)

    if yt.dtype != int and yt.dtype != float:
        yt = ensure_binary_series(yt)

    metrics = {
        "Accuracy": accuracy_score(yt, y_pred),
        "Precision": precision_score(yt, y_pred, zero_division=0),
        "Recall": recall_score(yt, y_pred, zero_division=0),
        "F1": f1_score(yt, y_pred, zero_division=0),
    }
    if proba is not None:
        try:
            metrics["ROC-AUC"] = roc_auc_score(yt, proba)
        except Exception:
            pass
    cm = confusion_matrix(yt, y_pred, labels=[0, 1])
    return metrics, cm, proba, y_pred

def plot_confusion_matrix(cm, title="Confusion Matrix"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0,1])
    ax.set_yticks([0,1])
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center')
    st.pyplot(fig)

def plot_metric_bars(results, metric_name):
    # results: dict model_name -> metrics dict
    labels = list(results.keys())
    values = [results[m].get(metric_name, np.nan) for m in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(f"{metric_name} Comparison")
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    st.pyplot(fig)

def plot_roc_curves(roc_dict):
    # roc_dict: name -> (fpr, tpr)
    fig, ax = plt.subplots()
    for name, (fpr, tpr) in roc_dict.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1], [0,1], linestyle="--")
    ax.set_title("ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# ----------------- Sidebar -----------------
with st.sidebar:
    st.header("Data & Settings")
    csv_choice = st.radio("How to load data?", ["Use default path", "Upload CSV"])
    default_paths = [
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv"  # fallback for this session
    ]
    default_path = next((p for p in default_paths if os.path.exists(p)), default_paths[0])
    uploaded = None
    if csv_choice == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv"])

    target_col = st.text_input("Target column", value="Churn")
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", value=42, step=1)

    st.markdown("---")
    st.subheader("Transform Options")
    scale_numeric = st.checkbox("Scale numeric features", value=True)
    encode_categorical = st.checkbox("One-Hot encode categorical", value=True)

    st.markdown("---")
    use_smote = st.checkbox("Apply SMOTE to training set (handle class imbalance)", value=True)

# ----------------- Load Data -----------------
if csv_choice == "Upload CSV" and uploaded is not None:
    df = load_data(uploaded)
else:
    try:
        df = load_data(default_path)
    except Exception as e:
        st.error(f"Failed to load default file '{default_path}'. Upload a CSV instead.\n\n{e}")
        st.stop()

st.title("📊 Telco Churn — End-to-End ML (EDA → Outliers → Chi2 → Transform → SMOTE → Models → Comparison)")

st.subheader("1) Dataset Preview")
st.write("Shape:", df.shape)
st.dataframe(df.head(20), use_container_width=True)

if target_col not in df.columns:
    st.error(f"Target column '{target_col}' not found in dataset.")
    st.stop()

# Basic columns
y_raw = df[target_col]
X_raw = df.drop(columns=[target_col])

num_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

tabs = st.tabs(["EDA", "Outliers", "Chi-Square Test", "Transform & Split", "SMOTE", "Train & Compare"])

# ----------------- EDA -----------------
with tabs[0]:
    left, right = st.columns(2)
    with left:
        st.write("Target distribution")
        counts = y_raw.astype(str).value_counts()
        fig, ax = plt.subplots()
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel(target_col)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    with right:
        if "Contract" in df.columns:
            st.write("Contract distribution")
            fig2, ax2 = plt.subplots()
            df["Contract"].value_counts().plot(kind="bar", ax=ax2)
            ax2.set_xlabel("Contract")
            ax2.set_ylabel("Count")
            st.pyplot(fig2)
        else:
            st.info("Column 'Contract' not found; skip demo chart.")

# ----------------- Outliers -----------------
with tabs[1]:
    st.write("IQR-based outlier detection on numeric features")
    if num_cols:
        summary = iqr_outlier_summary(df, num_cols)
        st.dataframe(summary, use_container_width=True)
        st.caption("Use the button below to clip outliers to [Q1-1.5*IQR, Q3+1.5*IQR]")
        if st.button("Apply IQR clipping (in-memory)"):
            df[:] = clip_outliers_iqr(df, num_cols)
            st.success("Outliers clipped using IQR bounds.")
    else:
        st.info("No numeric columns detected.")

# ----------------- Chi-Square Test -----------------
with tabs[2]:
    st.write("Feature relevance (categorical) via Chi-Square Test against target")
    y_bin = ensure_binary_series(y_raw)
    if len(pd.unique(y_bin)) != 2:
        st.warning("Chi-square test expects a binary target; mapping failed or not binary.")
    else:
        # One-hot encode only categorical columns for chi2
        if cat_cols:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            X_ohe = ohe.fit_transform(X_raw[cat_cols])
            feature_names = ohe.get_feature_names_out(cat_cols)
            # MinMax scale to satisfy chi2 non-negativity on numeric parts (if we also include numeric)
            # Here we use only categorical OHE (already non-negative), so scaling is optional.
            chi2_vals, p_vals = chi2(X_ohe, y_bin)
            chi_df = pd.DataFrame({
                "feature": feature_names,
                "chi2": chi2_vals,
                "p_value": p_vals
            }).sort_values("p_value")
            st.dataframe(chi_df.head(30), use_container_width=True)
            fig3, ax3 = plt.subplots()
            topn = chi_df.head(15).set_index("feature")["chi2"]
            topn.plot(kind="bar", ax=ax3)
            ax3.set_title("Top features by Chi2 (lower p-values are more significant)")
            ax3.set_ylabel("Chi2")
            ax3.set_xticklabels(ax3.get_xticklabels(), rotation=30, ha="right")
            st.pyplot(fig3)
        else:
            st.info("No categorical columns to test.")

# ----------------- Transform & Split -----------------
with tabs[3]:
    st.write("Configure transformations and split the dataset")
    preprocessor, num_cols2, cat_cols2 = build_preprocessor(
        X_raw, scale_numeric=scale_numeric, encode_categorical=encode_categorical
    )
    st.code(f"Numeric cols: {num_cols2}\nCategorical cols: {cat_cols2}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, ensure_binary_series(y_raw), test_size=test_size, random_state=random_state,
        stratify=ensure_binary_series(y_raw) if len(pd.unique(ensure_binary_series(y_raw)))==2 else None
    )
    st.write("Train size:", X_train.shape, " | Test size:", X_test.shape)

# ----------------- SMOTE -----------------
with tabs[4]:
    st.write("Apply SMOTE (train set only)")
    if use_smote:
        before_counts = pd.Series(y_train).value_counts()
        st.write("Class balance (before):")
        st.write(before_counts)

        # Create a light preprocessing to ensure all-numeric for SMOTE (OHE + scale to 0-1)
        pre_for_smote = ColumnTransformer([
            ("num", MinMaxScaler(), num_cols2),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols2)
        ])
        X_train_enc = pre_for_smote.fit_transform(X_train)
        sm = SMOTE(random_state=random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train_enc, y_train)
        after_counts = pd.Series(y_train_res).value_counts()
        st.write("Class balance (after SMOTE):")
        st.write(after_counts)
        st.caption("Note: SMOTE applied on encoded+scaled train features internally. Models will use full pipeline below.")
    else:
        st.info("SMOTE is disabled in sidebar.")

# ----------------- Train & Compare -----------------
with tabs[5]:
    st.write("Select models, tune hyperparameters, train and compare")
    m1, m2, m3 = st.columns(3)
    with m1:
        use_knn = st.checkbox("K-Nearest Neighbours", value=True)
        knn_k = st.slider("K (neighbors)", 3, 25, 7, 1)
    with m2:
        use_lr = st.checkbox("Logistic Regression", value=True)
        lr_C = st.slider("C (inverse regularization)", 0.01, 5.0, 1.0, 0.01)
        lr_max_iter = st.number_input("Max Iter", value=300, step=50)
    with m3:
        use_rf = st.checkbox("Random Forest", value=True)
        rf_n = st.slider("n_estimators", 50, 500, 200, 10)
        rf_depth = st.slider("max_depth (0=auto)", 0, 50, 0, 1)

    # Build base preprocessor for models
    preprocessor, _, _ = build_preprocessor(
        X_raw, scale_numeric=scale_numeric, encode_categorical=encode_categorical
    )

    models = {}
    if use_knn:
        models["KNN"] = Pipeline([
            ("pre", preprocessor),
            ("clf", KNeighborsClassifier(n_neighbors=knn_k))
        ])
    if use_lr:
        models["LogReg"] = Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(C=lr_C, max_iter=lr_max_iter, solver="liblinear"))
        ])
    if use_rf:
        rf = RandomForestClassifier(
            n_estimators=rf_n,
            max_depth=None if rf_depth == 0 else rf_depth,
            random_state=random_state,
            n_jobs=-1
        )
        models["RandomForest"] = Pipeline([("pre", preprocessor), ("clf", rf)])

    results = {}
    roc_data = {}
    trained_models = {}
    if st.button("Train models"):
        for name, pipe in models.items():
            with st.spinner(f"Training {name}..."):
                pipe.fit(X_train, y_train)
                metrics, cm, proba, y_pred = evaluate(pipe, X_test, y_test)
                results[name] = metrics
                trained_models[name] = pipe

                st.markdown(f"### {name} Results")
                st.json({k: round(v, 4) for k, v in metrics.items()})
                st.write("Confusion Matrix (rows=true, cols=pred):")
                st.write(cm)
                plot_confusion_matrix(cm, title=f"{name} — Confusion Matrix")

                if proba is not None:
                    fpr, tpr, _ = roc_curve(y_test, proba)
                    roc_data[name] = (fpr, tpr)

        if results:
            st.markdown("## 📈 Metric Comparisons")
            for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
                if any(metric in m for m in results.values()):
                    plot_metric_bars(results, metric)

        if len(roc_data) >= 1:
            st.markdown("## ROC Curves")
            plot_roc_curves(roc_data)

        if "RandomForest" in trained_models:
            try:
                rf_model = trained_models["RandomForest"]["clf"]
                if hasattr(rf_model, "feature_importances_"):
                    st.markdown("## Random Forest — Feature Importances (Top 20)")
                    # Extract feature names from the preprocessor
                    pre = trained_models["RandomForest"]["pre"]
                    num_features = pre.transformers_[0][2] if pre.transformers_ else []
                    cat_features = []
                    for name_t, trans, cols in pre.transformers_:
                        if name_t == "cat" and hasattr(trans, "get_feature_names_out"):
                            cat_features = list(trans.get_feature_names_out(cols))
                    feature_names = list(num_features) + list(cat_features)
                    importances = rf_model.feature_importances_
                    order = np.argsort(importances)[::-1][:20]
                    fig_imp, ax_imp = plt.subplots()
                    ax_imp.bar(range(len(order)), importances[order])
                    ax_imp.set_xticks(range(len(order)))
                    ax_imp.set_xticklabels([str(feature_names[i]) for i in order], rotation=30, ha="right")
                    ax_imp.set_ylabel("Importance")
                    st.pyplot(fig_imp)
            except Exception as e:
                st.info(f"Could not extract feature importances: {e}")
