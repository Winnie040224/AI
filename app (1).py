# app.py
import os
import io
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 可选：SHAP（可解释性）
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

st.set_page_config(page_title="Telco Churn – Streamlit Prototype", layout="wide")
st.title("📱 Telco Customer Churn – Streamlit Prototype")

RANDOM_STATE = 42

# ============== Utilities ==============
@st.cache_data(show_spinner=False)
def load_data(default_path: str = "WA_Fn-UseC_-Telco-Customer-Churn.csv"):
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        return df
    return None

def clean_transform_base(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    # TotalCharges to numeric first
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # unify blanks to NaN
    df.replace([' ', 'NaN', 'N/A'], np.nan, inplace=True)
    # drop NaN
    df.dropna(inplace=True)

    # normalize "No internet/phone service"
    replace_cols = ['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                    'StreamingTV','StreamingMovies','MultipleLines']
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({'No internet service':'No','No phone service':'No'})

    # 0/1 to Yes/No
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].replace({1:'Yes', 0:'No'})

    # drop obvious non-features if exist
    drop_cols = [c for c in ['customerID','gender','PhoneService'] if c in df.columns]
    if drop_cols:
        df.drop(drop_cols, axis=1, inplace=True)

    # keep only Yes/No target mapping
    if 'Churn' in df.columns:
        df = df[df['Churn'].isin(['Yes','No'])]

    return df

def split_xy(df: pd.DataFrame):
    y = df['Churn'].map({'Yes':1, 'No':0})
    X = df.drop('Churn', axis=1)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    return X, y, cat_cols, num_cols

def make_preprocessor(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

def plot_hist_kde(df, col):
    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.histplot(df[col], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribution of {col}")
    st.pyplot(fig)

def plot_box(df, col):
    fig, ax = plt.subplots(figsize=(6,3.5))
    sns.boxplot(x=df[col], ax=ax)
    ax.set_title(f"Boxplot of {col}")
    st.pyplot(fig)

def plot_corr_heatmap(df_num, drop_col=None):
    corr = df_num.corr(numeric_only=True)
    if drop_col in corr.columns:
        corr = corr.drop(index=drop_col, columns=drop_col)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    st.pyplot(fig)

def plot_count(series, title):
    fig, ax = plt.subplots(figsize=(4.5,3.5))
    series.value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title(title); ax.set_xlabel(series.name); ax.set_ylabel("Count")
    st.pyplot(fig)

def plot_confusion(cm, title):
    fig, ax = plt.subplots(figsize=(4.5,3.8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn','Churn'], yticklabels=['No Churn','Churn'], ax=ax)
    ax.set_title(title); ax.set_ylabel('Actual'); ax.set_xlabel('Predicted')
    st.pyplot(fig)

def metrics_table(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    pre = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan
    return pd.DataFrame([{
        "Accuracy": acc, "Precision": pre, "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    }])

def plot_metrics_bars(report_dict, accuracy, title):
    df_report = pd.DataFrame(report_dict).transpose()
    metrics_df = df_report[['precision','recall','f1-score']].drop(
        ['accuracy','macro avg','weighted avg'], errors='ignore'
    )
    metrics_df = metrics_df.reset_index().rename(columns={'index':'Class'})
    melted = pd.melt(metrics_df, id_vars='Class', value_vars=['precision','recall','f1-score'],
                     var_name='Metric', value_name='Score')
    fig, ax = plt.subplots(figsize=(7.5,4.8))
    sns.barplot(data=melted, x='Class', y='Score', hue='Metric', ax=ax)
    ax.bar('Accuracy', accuracy, color='red', width=0.4, label='Accuracy')
    ax.set_title(f'{title} (Accuracy: {accuracy:.2f})'); ax.set_ylim(0,1); ax.grid(axis='y')
    handles, labels = ax.get_legend_handles_labels()
    if 'Accuracy' not in labels:
        handles.append(plt.Rectangle((0,0),1,1,color='red')); labels.append('Accuracy')
    ax.legend(handles=handles, labels=labels, title='Metric')
    st.pyplot(fig)

def plot_roc(y_true, y_prob, label):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_val = roc_auc_score(y_true, y_prob)
    plt.plot(fpr, tpr, label=f'{label} (AUC={auc_val:.3f})')


# ============== Data source ==============
st.sidebar.header("1) 数据来源 / 上传")
df_default = load_data()
uploaded = st.sidebar.file_uploader("上传 CSV（可替代内置数据）", type=["csv"])

if uploaded is not None:
    df_raw = pd.read_csv(uploaded)
elif df_default is not None:
    df_raw = df_default
else:
    st.warning("未找到默认数据集，请先上传 CSV。")
    st.stop()

st.success(f"已载入数据：{df_raw.shape[0]} 行 × {df_raw.shape[1]} 列")

# ============== Cleaning preview ==============
with st.expander("查看原始数据前几行", expanded=False):
    st.dataframe(df_raw.head())

df = clean_transform_base(df_raw)
st.info(f"清洗后数据：{df.shape[0]} 行 × {df.shape[1]} 列")

# Tabs
tab_eda, tab_pre, tab_train, tab_batch, tab_single, tab_shap = st.tabs([
    "🧭 EDA 可视化", "🧪 预处理检查", "🏋️ 训练与评估", "📤 批量预测导出", "🧍 单条预测", "🔍 可解释性（SHAP）"
])

# ============== EDA Tab ==============
with tab_eda:
    st.subheader("探索性分析（替代你 Jupyter 里的图）")

    # 直方图/KDE + 箱线图
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    # 将 TotalCharges 转数值后，可能在 df 中还是 object? 我们上面已处理
    left, right = st.columns(2)
    with left:
        col_for_hist = st.selectbox("选择列绘制直方图/KDE", options=cols_num, index=cols_num.index('MonthlyCharges') if 'MonthlyCharges' in cols_num else 0)
        plot_hist_kde(df, col_for_hist)
    with right:
        col_for_box = st.selectbox("选择列绘制箱线图", options=cols_num, index=cols_num.index('TotalCharges') if 'TotalCharges' in cols_num else 0)
        plot_box(df, col_for_box)

    # 相关系数热力
    st.markdown("---")
    st.markdown("#### 相关系数热力图")
    df_num = df.select_dtypes(include=[np.number])
    drop_sc = 'SeniorCitizen' if 'SeniorCitizen' in df_num.columns else None
    plot_corr_heatmap(df_num, drop_col=drop_sc)

    # 卡方检验（分类列）
    st.markdown("---")
    st.markdown("#### 卡方检验（分类列 vs. Churn）")
    cat_cols_all = df.select_dtypes(include=['object']).columns.tolist()
    if 'customerID' in cat_cols_all:
        cat_cols_all.remove('customerID')
    if 'Churn' in cat_cols_all:
        cat_cols_all.remove('Churn')

    import scipy.stats as stats
    rows = []
    for col in cat_cols_all:
        contingency = pd.crosstab(df[col], df['Churn'])
        chi2, p, dof, exp = stats.chi2_contingency(contingency)
        rows.append([col, chi2, p, "✅ Significant" if p < 0.05 else "❌ Not significant"])
    chi_df = pd.DataFrame(rows, columns=["Feature","Chi2","p-value","Conclusion"])
    st.dataframe(chi_df)

# ============== Preprocessing Tab ==============
with tab_pre:
    st.subheader("预处理配置预览")
    X, y, cat_cols, num_cols = split_xy(df)
    st.write("**分类特征：**", cat_cols)
    st.write("**数值特征：**", num_cols)
    st.write("**目标分布（Churn）**")
    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(y.value_counts().rename("count"))
    with col2:
        plot_count(y, "Churn 分布（0=No, 1=Yes）")

# ============== Train & Evaluate Tab ==============
with tab_train:
    st.subheader("训练与评估（KNN 无/有 SMOTE、Logistic、RandomForest）")
    X, y, cat_cols, num_cols = split_xy(df)
    preprocessor = make_preprocessor(cat_cols, num_cols)

    test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    # Pipelines
    pipe_knn = Pipeline(steps=[
        ("prep", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    pipe_knn_smote = ImbPipeline(steps=[
        ("prep", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    pipe_lr = Pipeline(steps=[
        ("prep", preprocessor),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])

    pipe_rf = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])

    colA, colB = st.columns(2)
    with colA:
        do_grid = st.checkbox("轻量网格搜索（KNN/LR/RF）", value=False)
    with colB:
        cv_folds = st.number_input("CV 折数（GridSearch）", min_value=3, max_value=10, value=5, step=1)

    results = []
    roc_store = []

    def run_model(name, pipe, param_grid=None):
        if do_grid and param_grid:
            gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv_folds, scoring='f1', n_jobs=-1)
            gs.fit(X_train, y_train)
            model = gs.best_estimator_
            best = gs.best_params_
        else:
            model = pipe.fit(X_train, y_train)
            best = None
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
        mt = metrics_table(y_test, y_pred, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        st.markdown(f"### {name}")
        if best: st.write("Best params:", best)
        st.dataframe(mt.style.format("{:.3f}"))

        plot_confusion(cm, f"{name} – Confusion Matrix")

        rep = classification_report(y_test, y_pred, output_dict=True)
        plot_metrics_bars(rep, mt['Accuracy'].iloc[0], title=name)

        if y_prob is not None:
            roc_store.append((name, y_prob))
        results.append({"Model": name, **mt.iloc[0].to_dict(), "Estimator": model})

    # 参数网格（可选）
    param_knn = {"clf__n_neighbors": [5, 11]}
    param_lr  = {"clf__C": [0.1, 1.0, 10.0]}
    param_rf  = {"clf__max_depth": [None, 12], "clf__n_estimators":[200]}

    # Run all models
    run_model("KNN (No SMOTE)", pipe_knn, param_grid=param_knn)
    run_model("KNN (SMOTE)", pipe_knn_smote, param_grid=param_knn)
    run_model("Logistic Regression", pipe_lr, param_grid=param_lr)
    run_model("Random Forest", pipe_rf, param_grid=param_rf)

    # ROC curves
    if roc_store:
        fig = plt.figure(figsize=(6.5,5))
        for name, yprob in roc_store:
            plot_roc(y_test, yprob, name)
        plt.plot([0,1],[0,1],'k--', label='Random Guess')
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('ROC Curves'); plt.legend(loc='lower right'); plt.grid(True)
        st.pyplot(fig)

    # Comparison table
    if results:
        df_cmp = pd.DataFrame([{k:v for k,v in d.items() if k != "Estimator"} for d in results])
        df_cmp = df_cmp.sort_values(by=["F1-Score","Recall","ROC-AUC"], ascending=False)
        st.markdown("### 模型对比表")
        st.dataframe(df_cmp.style.format({"Accuracy":"{:.3f}","Precision":"{:.3f}",
                                          "Recall":"{:.3f}","F1-Score":"{:.3f}","ROC-AUC":"{:.3f}"}))
        # 保存到 session_state，供其它 tab 使用
        st.session_state["models"] = results
        st.session_state["X_train"] = X_train
        st.session_state["y_train"] = y_train
        st.session_state["X_test"]  = X_test
        st.session_state["y_test"]  = y_test
        st.session_state["preprocessor"] = preprocessor
        st.session_state["cat_cols"] = cat_cols
        st.session_state["num_cols"] = num_cols

# ============== Batch scoring Tab ==============
with tab_batch:
    st.subheader("批量预测并导出 CSV（本周目标客户清单）")
    if "models" not in st.session_state:
        st.warning("请先到『训练与评估』标签页训练模型。")
    else:
        # 选择一个模型
        model_names = [d["Model"] for d in st.session_state["models"]]
        chosen = st.selectbox("选择用于预测的模型", model_names, index=0)
        mdl = [d for d in st.session_state["models"] if d["Model"] == chosen][0]["Estimator"]

        st.markdown("上传待预测 CSV（字段尽量与训练集一致，至少包含你保留的特征列；无需包含 Churn）")
        up = st.file_uploader("上传待预测数据", type=["csv"], key="batch_upload")
        if up:
            df_new = pd.read_csv(up)
            df_new = clean_transform_base(df_new)
            if 'Churn' in df_new.columns:
                df_new = df_new.drop(columns=['Churn'])

            probs = mdl.predict_proba(df_new)[:,1] if hasattr(mdl, "predict_proba") else None
            preds = mdl.predict(df_new)
            out = df_new.copy()
            out['churn_probability'] = probs if probs is not None else preds
            out['churn_pred'] = preds

            st.dataframe(out.head())

            # 下载
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("⬇️ 下载目标客户清单 CSV", data=buf.getvalue(),
                               file_name="target_customers.csv", mime="text/csv")

# ============== Single prediction Tab ==============
with tab_single:
    st.subheader("单条客户信息 → 即时预测")
    if "models" not in st.session_state:
        st.warning("请先到『训练与评估』标签页训练模型。")
    else:
        mdl = [d for d in st.session_state["models"] if d["Model"] == "Random Forest"][0]["Estimator"]
        # 动态生成输入控件
        schema = st.session_state["preprocessor"]
        X_cols = st.session_state["cat_cols"] + st.session_state["num_cols"]

        user_inputs = {}
        # 简化：从 df 的同名列中抽取一个参考选项
        sample_row = df.iloc[0]

        cols1, cols2 = st.columns(2)
        for i, col in enumerate(X_cols):
            if col in st.session_state["cat_cols"]:
                choices = sorted(df[col].dropna().unique().tolist())
                ui = (cols1 if i%2==0 else cols2).selectbox(f"{col}", choices, index=0)
            else:
                default_val = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
                ui = (cols1 if i%2==0 else cols2).number_input(f"{col}", value=float(default_val))
            user_inputs[col] = ui

        # 阈值与成本
        st.markdown("---")
        thr = st.slider("判定阈值（超过此概率即判为 Churn）", 0.1, 0.9, 0.5, 0.01)
        benefit = st.number_input("成功挽留收益（RM）", value=500.0, step=50.0)
        outreach_cost = st.number_input("触达成本（RM）/人", value=50.0, step=10.0)

        if st.button("🔮 预测"):
            x_df = pd.DataFrame([user_inputs])
            prob = mdl.predict_proba(x_df)[:,1][0] if hasattr(mdl, "predict_proba") else float(mdl.predict(x_df)[0])
            pred = int(prob >= thr)
            st.metric("流失概率", f"{prob:.3f}")
            st.write("预测标签：", "⚠️ Churn" if pred==1 else "✅ No Churn")

            # 简单“期望收益”
            expected_profit = (benefit - outreach_cost) if pred==1 else -outreach_cost
            st.write(f"**建议**：{'纳入挽留名单' if pred==1 else '暂不触达'}；预计本客户净效益：RM {expected_profit:.2f}")

# ============== SHAP Tab ==============
with tab_shap:
    st.subheader("SHAP 可解释性（以 Random Forest 为例）")
    if not _HAS_SHAP:
        st.info("未安装 shap，运行：pip install shap")
    elif "models" not in st.session_state:
        st.warning("请先到『训练与评估』标签页训练模型。")
    else:
        # 取 RF 模型与测试集
        rf_list = [d for d in st.session_state["models"] if d["Model"] == "Random Forest"]
        if not rf_list:
            st.warning("请先训练 Random Forest。")
        else:
            rf = rf_list[0]["Estimator"]
            X_test = st.session_state["X_test"]
            y_test = st.session_state["y_test"]

            st.caption("注：OneHot + 树模型 → 使用 TreeExplainer 更快；下方展示全局重要性 + 若干局部解释。")
            try:
                # 取预处理后的 feature names（OneHot 展开）
                prep = rf.named_steps["prep"]
                ohe = prep.named_transformers_["cat"]
                cat_out = ohe.get_feature_names_out(st.session_state["cat_cols"])
                feat_names = list(cat_out) + st.session_state["num_cols"]

                # 变换测试集到模型输入空间
                X_test_trans = prep.transform(X_test)
                # 如果是 RF，TreeExplainer 适配
                explainer = shap.TreeExplainer(rf.named_steps["clf"])
                shap_values = explainer.shap_values(X_test_trans)

                st.markdown("**全局特征重要性（mean |SHAP|）**")
                fig = plt.figure(figsize=(7.5,5))
                shap.summary_plot(shap_values[1], X_test_trans, feature_names=feat_names, show=False)
                st.pyplot(fig, clear_figure=True)

                # 局部解释（展示前 3 条）
                st.markdown("---")
                st.markdown("**局部解释（前 3 个样本）**")
                for i in range(min(3, X_test_trans.shape[0])):
                    st.write(f"样本 #{i}")
                    fig2 = plt.figure(figsize=(7,3.8))
                    shap.force_plot(explainer.expected_value[1], shap_values[1][i,:], 
                                    matplotlib=True, show=False)
                    st.pyplot(fig2, clear_figure=True)

            except Exception as e:
                st.error(f"SHAP 可视化失败：{e}")
                st.info("提示：不同 shap 版本与稀疏/稠密矩阵兼容性可能导致错误；必要时先将输入转为 dense。")
