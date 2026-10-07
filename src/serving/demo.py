import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src import config


st.set_page_config(
    page_title="Sentinel | Fraud Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#101D35"
BLUE = "#326BFF"
TEAL = "#11A88A"
RED = "#D94F67"
MUTED = "#72809A"

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root {{ --navy:{NAVY}; --blue:{BLUE}; --teal:{TEAL}; --muted:{MUTED}; }}
      html, body, [class*="css"] {{ font-family:'DM Sans',sans-serif; }}
      .stApp {{ background:#F5F7FB; color:#17233B; }}
      [data-testid="stHeader"] {{ background:rgba(245,247,251,.88); }}
      [data-testid="stSidebar"] {{ background:#101D35; border-right:1px solid #223250; }}
      [data-testid="stSidebar"] * {{ color:#E8EEF9; }}
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ color:#9EADC6; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label {{ padding:7px 9px; border-radius:9px; }}
      [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{ background:#1B2B49; }}
      h1,h2,h3 {{ font-family:'Manrope',sans-serif; color:#17233B; letter-spacing:-.025em; }}
      h1 {{ font-size:2.05rem!important; }}
      h2 {{ font-size:1.42rem!important; }}
      .eyebrow {{ color:#73819B; font-size:.73rem; font-weight:700; letter-spacing:.14em; text-transform:uppercase; }}
      .page-title {{ font-family:'Manrope',sans-serif; color:#17233B; font-size:2rem; font-weight:800; letter-spacing:-.04em; margin:.15rem 0 .35rem; }}
      .subtle {{ color:#72809A; font-size:.94rem; }}
      .hero {{ background:linear-gradient(115deg,#101D35 0%,#1A3157 62%,#2856A8 100%); border-radius:20px; padding:26px 30px; color:#F4F7FD; margin:.4rem 0 1.3rem; box-shadow:0 12px 28px rgba(20,42,79,.14); }}
      .hero h2 {{ color:white; margin:0 0 7px; }} .hero p {{ color:#C0CDE2; margin:0; }}
      .hero-tag {{ display:inline-block; border:1px solid rgba(255,255,255,.22); border-radius:30px; padding:5px 10px; color:#D9E5FA; font-size:.75rem; margin-bottom:12px; }}
      .section-head {{ display:flex; align-items:center; justify-content:space-between; margin:1.2rem 0 .6rem; }}
      div[data-testid="stMetric"] {{ background:#fff; border:1px solid #E8ECF3; padding:17px 18px; border-radius:15px; box-shadow:0 4px 13px rgba(28,46,78,.035); }}
      div[data-testid="stMetricLabel"] p {{ color:#77849A; font-size:.82rem; }}
      div[data-testid="stMetricValue"] {{ color:#17233B; font-family:'Manrope',sans-serif; font-weight:800; }}
      div[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:16px; border-color:#E8ECF3; }}
      .status-pill {{ display:inline-block; border-radius:30px; padding:5px 10px; font-size:.76rem; font-weight:700; }}
      .status-ok {{ background:#E6F7F2; color:#087D66; }} .status-warn {{ background:#FFF1E7; color:#B75519; }}
      .login-logo {{ font-family:'Manrope',sans-serif; font-size:1.02rem; font-weight:800; color:#F4F6FA; letter-spacing:.04em; }}
      body:has(.login-logo) [data-testid="stAppViewContainer"] {{ background:#575B68; background-image:radial-gradient(ellipse at 50% 42%,rgba(21,42,75,.22),transparent 48%),radial-gradient(#24262B .65px,transparent .65px); background-size:auto,26px 26px; color:#F4F6FA; }}
      body:has(.login-logo) [data-testid="stHeader"] {{ background:transparent; }}
      body:has(.login-logo) [data-testid="stMainBlockContainer"] {{ max-width:430px; margin:7vh auto 0; padding:31px 34px 26px; background:rgba(17,18,21,.97); border:1px solid #26282D; border-radius:18px; box-shadow:0 26px 90px rgba(0,0,0,.48); }}
      body:has(.login-logo) .login-logo {{ text-align:center; margin-bottom:27px; }}
      body:has(.login-logo) .eyebrow {{ color:#6E91C5; text-align:center; }}
      body:has(.login-logo) .page-title {{ color:#F5F6F8; text-align:center; font-size:1.7rem!important; margin-top:6px; }}
      body:has(.login-logo) .subtle {{ color:#9499A4; text-align:center; font-size:.88rem; margin-bottom:19px; }}
      body:has(.login-logo) [data-testid="stTextInput"] label, body:has(.login-logo) [data-testid="stCheckbox"] label {{ color:#B7BBC4; }}
      body:has(.login-logo) input {{ background:#2d313c!important; color:#F4F6FA!important; border:1px solid #292B31!important; border-radius:9px!important; }}
      body:has(.login-logo) input:focus {{ border-color:#2878E5!important; box-shadow:0 0 0 1px #2878E5!important; }}
      body:has(.login-logo) [data-testid="stFormSubmitButton"] button {{ background:#0879E8; border:1px solid #0879E8; border-radius:9px; min-height:2.8rem; margin-top:5px; }}
      body:has(.login-logo) [data-testid="stFormSubmitButton"] button:hover {{ background:#1688F5; border-color:#1688F5; }}
      body:has(.login-logo) .login-foot {{ color:#696F7A; text-align:center; font-size:.73rem; margin-top:19px; }}
      .stButton>button {{ border-radius:10px; font-weight:700; min-height:2.6rem; }}
      .stButton>button[kind="primary"] {{ background:#326BFF; border-color:#326BFF; }}
      [data-testid="stDataFrame"] {{ border:1px solid #E8ECF3; border-radius:12px; overflow:hidden; }}
      footer {{ visibility:hidden; }}
    </style>
    """,
    unsafe_allow_html=True,
)


API = config.API_URL.rstrip("/")


def api_call(method, path, **kwargs):
    headers = {"Authorization": f"Bearer {st.session_state.get('token', '')}"}
    response = requests.request(
        method, API + path, headers=headers, timeout=120, **kwargs
    )
    if response.status_code == 401:
        st.session_state.pop("token", None)
        st.session_state.pop("username", None)
        st.rerun()
    response.raise_for_status()
    return response.json()


def show_login():
    st.markdown('<div class="login-logo">◈ &nbsp; SENTINEL <span style="color:#777D88;font-weight:500">/ FRAUD INTELLIGENCE</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">PFE · MASTER II BDCC</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Welcome back</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtle">Sign in to monitor risk signals and score transactions.</div>', unsafe_allow_html=True)
    with st.form("login"):
        username = st.text_input("Username", placeholder="Your administrator account")
        password = st.text_input("Password", type="password")
        remember = st.checkbox("Keep me signed in")
        submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if submitted:
        try:
            auth = requests.post(
                API + "/auth/login",
                json={"username": username, "password": password, "remember_me": remember},
                timeout=15,
            )
            auth.raise_for_status()
            data = auth.json()
            st.session_state.update(
                token=data["access_token"],
                role=data["role"],
                username=data.get("username", username),
            )
            st.rerun()
        except requests.RequestException:
            st.error("Could not sign in. Check the API and your credentials.")
    st.markdown('<div class="login-foot">SECURE ACCESS · ADAPTIVE FRAUD DETECTION</div>', unsafe_allow_html=True)
    st.stop()


if "token" not in st.session_state:
    show_login()

try:
    health = api_call("GET", "/health")
    status = api_call("GET", "/status")
except requests.RequestException as exc:
    st.error(f"API is unavailable: {exc}")
    st.stop()

metrics = health["train_metrics"]
user_name = st.session_state.get("username", "Analyst")

with st.sidebar:
    st.markdown("<div style='font:800 1.1rem Manrope;color:#fff;padding:8px 4px'>◈ &nbsp; SENTINEL</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.72rem;letter-spacing:.13em;color:#8293B0;margin:0 4px 18px'>FRAUD INTELLIGENCE PLATFORM</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:.72rem;color:#8293B0;margin:0 4px 7px'>WORKSPACE</div>", unsafe_allow_html=True)
    page = st.radio(
        "Navigation",
        ["Overview", "Transaction scoring", "Batch scoring", "Model evaluation", "Model & audit"],
        label_visibility="collapsed",
    )
    st.markdown("<hr style='border-color:#2B3B58;margin:22px 0'>", unsafe_allow_html=True)
    api_state = "● API connected" if health.get("status") == "healthy" else "● API issue"
    st.markdown(f"<div style='font-size:.84rem;color:#C6D3E8'>{api_state}</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:.77rem;color:#8293B0;margin-top:5px'>Signed in as {user_name} · {st.session_state.get('role','')}</div>", unsafe_allow_html=True)
    if health["synthetic"]:
        st.warning("Demo dataset is synthetic")
    else:
        st.success("IEEE-CIS data loaded")
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    if st.button("Sign out", use_container_width=True):
        st.session_state.clear()
        st.rerun()
    st.caption("Adaptive Fraud Detection · Research edition")


def page_header(eyebrow, title, description):
    st.markdown(f'<div class="eyebrow">{eyebrow}</div><div class="page-title">{title}</div><div class="subtle">{description}</div>', unsafe_allow_html=True)


def dataset(version):
    return (
        pd.read_parquet(config.PROC_DIR / "features.parquet"),
        joblib.load(config.MODEL_DIR / "feature_cols.pkl"),
    )


def nice_metric(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (float, int)) else str(value)


if page == "Overview":
    page_header("Risk operations / Overview", "Fraud intelligence overview", "A live view of model readiness, transaction scoring, and feedback activity.")
    model_state = "DRIFT RESPONSE ACTIVE" if health.get("in_drift") else "MODEL READY"
    st.markdown(
        f'<div class="hero"><span class="hero-tag">{model_state}</span><h2>Adaptive detection at a glance</h2><p>Fusion combines gradient-boosted scoring with online learning. Feedback updates the adaptive model after verified labels.</p></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions scored", f"{status['total_requests']:,}", help="Predictions recorded by this API instance.")
    c2.metric("Fusion ROC-AUC", nice_metric(metrics.get("auc_roc")))
    c3.metric("Fusion PR-AUC", nice_metric(metrics.get("auc_pr")))
    c4.metric("Feedback updates", f"{health.get('n_updates', 0):,}", delta=f"{health.get('drift_count', 0)} drift signals")

    left, right = st.columns([1.4, 1], gap="large")
    with left:
        st.markdown("### Model performance")
        comparison = pd.DataFrame([
            {"Model": "Fusion", "PR-AUC": metrics.get("auc_pr", 0), "ROC-AUC": metrics.get("auc_roc", 0), "F1 fraud": metrics.get("f1_fraud", 0)},
            {"Model": "Batch XGBoost", "PR-AUC": metrics.get("batch_test", {}).get("auc_pr", 0), "ROC-AUC": metrics.get("batch_test", {}).get("auc_roc", 0), "F1 fraud": metrics.get("batch_test", {}).get("f1_fraud", 0)},
            {"Model": "River", "PR-AUC": metrics.get("river_test", {}).get("auc_pr", 0), "ROC-AUC": metrics.get("river_test", {}).get("auc_roc", 0), "F1 fraud": metrics.get("river_test", {}).get("f1_fraud", 0)},
        ]).melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.bar(comparison, x="Metric", y="Score", color="Model", barmode="group",
                     color_discrete_sequence=[BLUE, TEAL, "#A7B3C5"], range_y=[0, 1])
        fig.update_layout(template="plotly_white", height=320, margin=dict(l=8,r=8,t=16,b=8), legend_title=None,
                          font=dict(family="DM Sans",color="#52617A"), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(title=None, gridcolor="#EEF1F6")
        fig.update_xaxes(title=None)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("### Runtime health")
        with st.container(border=True):
            st.markdown(f"**Data source**  \n{'IEEE-CIS training data' if not health['synthetic'] else 'Synthetic demonstration fixture'}")
            st.markdown(f"**Model version**  \n`{metrics.get('model_version','unknown')}`")
            st.markdown(f"**Fusion weight · XGBoost**  \n{health.get('alpha', 0):.1%}")
            st.markdown(f"**Decision threshold**  \n{metrics.get('threshold', 0):.3f}")
            st.markdown(f"**Explanation cache**  \n{'Available' if health.get('redis_available') else 'Local computation'}")
        if health["synthetic"]:
            st.warning("Synthetic results are for integration checks only; do not report them as IEEE-CIS metrics.")

    st.markdown("### Evaluation snapshot")
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    fraud_total = sum(cm[1]) if len(cm) > 1 else 0
    fraud_recall = cm[1][1] / fraud_total if fraud_total else 0
    a, b, c = st.columns(3)
    a.metric("Fraud recall at selected threshold", f"{fraud_recall:.1%}")
    a.progress(fraud_recall)
    b.metric("Fraud precision", f"{metrics.get('precision', 0):.1%}")
    b.progress(metrics.get("precision", 0))
    c.metric("Adaptive updates", f"{health.get('n_updates', 0):,}", help="Only explicit verified feedback updates River.")

elif page == "Transaction scoring":
    page_header("Risk operations / Score", "Transaction scoring", "Score a held-out IEEE-CIS transaction with the current fused model and inspect its explanation.")
    try:
        df, feature_cols = dataset(metrics["model_version"])
    except Exception as exc:
        st.error(f"Could not load processed features: {exc}")
        st.stop()
    start = int(metrics["dataset"]["validation_end"])
    upper = max(start, len(df) - 1)
    with st.container(border=True):
        selector, action = st.columns([3, 1])
        idx = selector.number_input("Held-out transaction row", min_value=min(start, upper), max_value=upper, value=min(start, upper), step=1)
        selector.caption("Rows are selected from the chronological test partition; the true label is withheld until after scoring.")
        run = action.button("Score transaction", type="primary", use_container_width=True)
        row = df.iloc[int(idx)]
        show_cols = [c for c in ["TransactionAmt", "TransactionAmt_log", "card1_count", "hour_sin", "hour_cos", "day_of_week"] if c in feature_cols]
        if show_cols:
            st.dataframe(pd.DataFrame([row[show_cols].to_dict()]), use_container_width=True, hide_index=True)
    if run:
        payload = {c: (float(row[c]) if pd.notna(row[c]) else None) for c in feature_cols}
        try:
            st.session_state["scored_result"] = api_call("POST", "/predict", json={"transaction_id": str(uuid.uuid4()), "features": payload})
            st.session_state["scored_true_label"] = int(row["isFraud"])
        except requests.RequestException as exc:
            st.error(f"Scoring failed: {exc}")
    result = st.session_state.get("scored_result")
    if result:
        st.markdown("### Scoring result")
        label_col, score_col, cutoff_col, truth_col = st.columns(4)
        flagged = bool(result["label"])
        label_col.metric("Decision", "FRAUD FLAGGED" if flagged else "CLEARED")
        score_col.metric("Fused score", f"{result['fused_score']:.1%}")
        cutoff_col.metric("Threshold", f"{result['threshold']:.1%}")
        truth_col.metric("Held-out truth", "Fraud" if st.session_state.get("scored_true_label") else "Legitimate")
        scores = pd.DataFrame({"Component": ["XGBoost", "River", "Fusion score"], "Score": [result["xgb_score"], result["river_score"], result["fused_score"]]})
        fig = px.bar(scores, x="Component", y="Score", color="Component", color_discrete_sequence=[BLUE, TEAL, "#F0A64A"], range_y=[0,1])
        fig.add_hline(y=result["threshold"], line_dash="dash", line_color=RED, annotation_text="Decision threshold")
        fig.update_layout(template="plotly_white", height=270, showlegend=False, margin=dict(l=8,r=8,t=18,b=5), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        shap = result.get("shap_top3", [])
        if shap:
            st.markdown("### Main XGBoost explanation signals")
            exp = pd.DataFrame(shap).sort_values("shap_value")
            chart = go.Figure(go.Bar(x=exp["shap_value"], y=exp["feature"], orientation="h",
                                     marker_color=[RED if v > 0 else TEAL for v in exp["shap_value"]]))
            chart.update_layout(template="plotly_white", height=230, margin=dict(l=8,r=8,t=8,b=8),
                                xaxis_title="Contribution to XGBoost log-odds", yaxis_title=None, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(chart, use_container_width=True, config={"displayModeBar":False})
            st.caption("SHAP explains XGBoost log-odds contributions. It does not explain the River or fused score.")
        if st.session_state.get("role") == "admin":
            with st.expander("Submit verified feedback"):
                st.caption("Use only a verified label. Feedback is applied once and updates the online learner.")
                label = st.selectbox("Verified outcome", [0, 1], format_func=lambda x: "Legitimate (0)" if x == 0 else "Fraud (1)", key="feedback_label")
                if st.button("Submit feedback", key="submit_feedback"):
                    try:
                        response = api_call("POST", "/feedback", json={"transaction_id": result["transaction_id"], "ground_truth": label})
                        st.success("Verified feedback recorded.")
                        st.json(response)
                    except requests.RequestException as exc:
                        st.error(f"Feedback failed: {exc}")

elif page == "Batch scoring":
    page_header("Risk operations / Batch", "Batch scoring", "Upload up to 500 processed transactions, review model decisions, then export the results.")
    try:
        _, feature_cols = dataset(metrics["model_version"])
    except Exception as exc:
        st.error(f"Could not load the model feature schema: {exc}")
        st.stop()
    st.markdown("Upload a CSV with one column for every required model feature. Labels such as `isFraud` are not accepted as inputs.")
    upload = st.file_uploader("Choose a feature CSV", type=["csv"], label_visibility="collapsed")
    if upload:
        try:
            data = pd.read_csv(upload)
            missing = sorted(set(feature_cols) - set(data.columns))
            extra = sorted(set(data.columns) - set(feature_cols))
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", f"{len(data):,}")
            m2.metric("Required features", len(feature_cols))
            m3.metric("Schema", "Ready" if not missing and not extra else "Needs attention")
            if missing:
                st.error("Missing columns: " + ", ".join(missing[:20]))
            if extra:
                st.warning("Extra columns will be ignored: " + ", ".join(extra[:20]))
            st.dataframe(data.head(8), use_container_width=True, hide_index=True)
            if not 1 <= len(data) <= 500:
                st.error("The API accepts from 1 to 500 transactions per batch.")
            elif not missing and st.button("Score batch", type="primary"):
                clean = data[feature_cols].where(pd.notna(data[feature_cols]), None)
                payload = [{"features": record} for record in clean.to_dict("records")]
                results = api_call("POST", "/predict/batch", json=payload)
                result_df = pd.DataFrame(results)
                st.markdown("### Batch results")
                flagged = int(result_df["label"].sum()) if "label" in result_df else 0
                q1, q2, q3 = st.columns(3)
                q1.metric("Scored", len(result_df))
                q2.metric("Flagged", flagged)
                q3.metric("Flag rate", f"{flagged / max(len(result_df), 1):.1%}")
                st.dataframe(result_df, use_container_width=True, hide_index=True)
                st.download_button("Download scored results", result_df.to_csv(index=False), "fraud_predictions.csv", "text/csv", type="primary")
        except (ValueError, requests.RequestException) as exc:
            st.error(f"Batch scoring failed: {exc}")

elif page == "Model evaluation":
    page_header("Model governance / Evaluation", "Model evaluation", "Frozen chronological holdout metrics from the latest training run.")
    if health["synthetic"]:
        st.warning("These metrics come from synthetic demo data and are not IEEE-CIS results.")
    else:
        st.info("Metrics below were generated from the IEEE-CIS files currently installed in this project.")
    fusion = metrics
    comparisons = [
        ("Fusion", fusion),
        ("Batch XGBoost", fusion.get("batch_test", {})),
        ("River", fusion.get("river_test", {})),
    ]
    table = pd.DataFrame([{"Model": name, "PR-AUC": m.get("auc_pr"), "ROC-AUC": m.get("auc_roc"), "F1 fraud": m.get("f1_fraud"), "Precision": m.get("precision"), "Recall": m.get("recall")} for name, m in comparisons])
    st.dataframe(table.style.format({c:"{:.3f}" for c in table.columns if c != "Model"}), use_container_width=True, hide_index=True)
    cm = fusion.get("confusion_matrix", [[0,0],[0,0]])
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Fusion confusion matrix")
        heat = go.Figure(go.Heatmap(z=cm, x=["Predicted legitimate", "Predicted fraud"], y=["Actually legitimate", "Actually fraud"],
                                    text=cm, texttemplate="%{text:,}", colorscale=[[0,"#E8F0FF"],[1,BLUE]], showscale=False))
        heat.update_layout(height=350, template="plotly_white", margin=dict(l=8,r=8,t=15,b=8), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(heat, use_container_width=True, config={"displayModeBar":False})
    with right:
        st.markdown("### Dataset and decision setup")
        ds = metrics.get("dataset", {})
        with st.container(border=True):
            st.markdown(f"**Provenance:** {ds.get('provenance','unknown')}")
            st.markdown(f"**Rows:** {ds.get('rows',0):,} &nbsp; · &nbsp; **Features:** {ds.get('features',0)}")
            st.markdown(f"**Train boundary:** {ds.get('train_end',0):,}")
            st.markdown(f"**Validation boundary:** {ds.get('validation_end',0):,}")
            st.markdown(f"**Threshold:** {metrics.get('threshold',0):.4f}")
            st.markdown(f"**Split:** {ds.get('split','')}")
        st.caption("AUC summarizes ranking quality; precision and recall describe decisions at the selected threshold. Class weighting means scores are not calibrated probabilities.")
    if metrics.get("evaluation"):
        st.info(metrics["evaluation"])
    adaptation_path = config.MODEL_DIR / "adaptation_evaluation.json"
    if adaptation_path.exists():
        with st.expander("Predict-then-learn adaptation experiment"):
            try:
                import json
                st.json(json.loads(adaptation_path.read_text(encoding="utf-8")))
            except (ValueError, OSError) as exc:
                st.error(f"Could not read the adaptation report: {exc}")

else:
    page_header("Model governance / Administration", "Model & audit", "Inspect the active model state and review administrator actions.")
    try:
        info = api_call("GET", "/model/info")
        state1, state2, state3, state4 = st.columns(4)
        state1.metric("Model version", info.get("model_version", "—"))
        state2.metric("Fusion alpha", f"{info.get('alpha',0):.1%}")
        state3.metric("Online updates", f"{info.get('n_updates',0):,}")
        state4.metric("Drift alarms", info.get("drift_count",0))
        with st.expander("Model details", expanded=True):
            st.json(info)
    except requests.RequestException as exc:
        st.error(f"Could not load model information: {exc}")
    if st.session_state.get("role") == "admin":
        if st.button("Save online model state", type="primary"):
            try:
                st.success("Online state saved.")
                st.json(api_call("POST", "/model/save"))
            except requests.RequestException as exc:
                st.error(f"Save failed: {exc}")
        with st.expander("Security audit trail"):
            if st.button("Load latest audit events"):
                try:
                    logs = api_call("GET", "/auth/audit-logs?limit=100")
                    st.dataframe(pd.DataFrame(logs), use_container_width=True, hide_index=True)
                except requests.RequestException as exc:
                    st.error(f"Could not load audit events: {exc}")

st.markdown("<div style='border-top:1px solid #E8ECF3;margin-top:30px;padding-top:14px;color:#8A96A9;font-size:.76rem'>Sentinel · Adaptive Fraud Detection Research Platform &nbsp; · &nbsp; XGBoost + River + ADWIN</div>", unsafe_allow_html=True)
