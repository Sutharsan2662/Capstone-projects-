# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings
warnings.filterwarnings("ignore")

# ---------------------------
# Configuration - adjust if needed
# ---------------------------
MODEL_DIR = Path(r"C:\Users\C Sutharsan\Downloads\GUVI class notes AIML\Capstone_project\Project_4\Final\models")
DATA_PATH = "Final_dataset.csv"  # dataset must be in same folder as app or give full path

# ---------------------------
# Utilities
# ---------------------------
def safe_str(s):
    if pd.isna(s):
        return ""
    return str(s).encode("utf-8", errors="ignore").decode("utf-8")

@st.cache_data(show_spinner=False)
def load_csv_safe(file_path: str):
    encodings = ["utf-8", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc, errors="replace")
            df.columns = [safe_str(c) for c in df.columns]
            return df
        except Exception:
            continue
    return None

@st.cache_resource(show_spinner=False)
def load_joblib(path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def ensure_list_unique_preserve_order(seq):
    seen = set()
    res = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res

# ---------------------------
# Load models / assets (exact filenames from your screenshot)
# ---------------------------
load_msgs = []
try:
    # Regression models and assets
    xgb_reg = load_joblib(MODEL_DIR / "XGBoost_model.pkl")
    gb_reg = load_joblib(MODEL_DIR / "GradientBoosting_model.pkl")
    top_features = load_joblib(MODEL_DIR / "top_features.pkl")
    label_encoders_reg = load_joblib(MODEL_DIR / "label_encoders.pkl")
except Exception as e:
    xgb_reg = gb_reg = top_features = label_encoders_reg = None
    load_msgs.append(f"Regression load error: {e}")

try:
    # Classification models and assets
    clf_model = load_joblib(MODEL_DIR / "XGBoost_advanced_classifier.pkl")
    feature_label_encoders = load_joblib(MODEL_DIR / "feature_label_encoders.pkl")
    target_label_encoder = load_joblib(MODEL_DIR / "target_label_encoder.pkl")
    classification_top_features = load_joblib(MODEL_DIR / "classification_top_features.pkl")
except Exception as e:
    clf_model = feature_label_encoders = target_label_encoder = classification_top_features = None
    load_msgs.append(f"Classification load error: {e}")

try:
    # Recommendation assets (these come from your screenshot)
    cf_data = load_pickle(MODEL_DIR / "cf_data.pkl")  # earlier you stored triple (user_type_matrix, user_similarity, user_index_map)
    cbf_data = load_pickle(MODEL_DIR / "cbf_data.pkl")  # (tfidf, feature_matrix, attraction_similarity, attr_index_map)
    # Map unpack carefully
    # cf_data likely contains (user_type_matrix, user_similarity, user_index_map)
    # cbf_data likely contains (tfidf, feature_matrix, attraction_similarity, attr_index_map)
    # We'll not assume order beyond this; code below handles missing components gracefully.
    try:
        user_type_matrix, user_similarity, user_index_map = cf_data
    except Exception:
        user_type_matrix = user_similarity = user_index_map = None
    try:
        tfidf_vec, feature_matrix, attraction_similarity, attr_index_map = cbf_data
    except Exception:
        tfidf_vec = feature_matrix = attraction_similarity = attr_index_map = None
except Exception as e:
    user_type_matrix = user_similarity = user_index_map = None
    tfidf_vec = feature_matrix = attraction_similarity = attr_index_map = None
    load_msgs.append(f"Recommendation load error: {e}")

# Load dataset
df = pd.read_csv(DATA_PATH)
if df is None:
    load_msgs.append("Failed to load Final_dataset.csv (check path / encoding).")

# ---------------------------
# Streamlit Page Setup
# ---------------------------
st.set_page_config(page_title="Tourism Experience Analysis", layout="wide")
st.sidebar.title("Tourism Experience")
st.sidebar.caption("Regression | Classification | Recommendations")

if load_msgs:
    with st.sidebar.expander("Load messages (click)"):
        for m in load_msgs:
            st.error(safe_str(m))

st.title("🏞 Tourism Experience Analysis Platform")
st.write("Use the tabs below to work with Regression, Classification, and Recommendation modules.")

tabs = st.tabs(["📊 Rating Prediction (Regression)", "🎯 Visit Mode (Classification)", "💡 Recommendations", "ℹ️ Info"])

# ---------------------------
# Helper: feature importance extraction (robust)
# ---------------------------
def get_feature_importances_from_model(model, feature_names):
    """
    Try multiple ways to extract feature importances.
    Returns a DataFrame with 'feature' and 'importance' columns.
    """
    importances = None
    # If model is pipeline, try to get final estimator
    try:
        if hasattr(model, "named_steps"):
            # attempt to access final estimator (commonly named 'model' or last step)
            last_step = list(model.named_steps.items())[-1][1]
            estimator = last_step
        else:
            estimator = model
    except Exception:
        estimator = model

    # Try `feature_importances_`
    if hasattr(estimator, "feature_importances_"):
        importances = np.array(estimator.feature_importances_)
    # Try coef_
    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        # if multiclass (coef_ is 2D), take mean of absolute coefs across classes
        if coef.ndim == 1:
            importances = np.abs(coef)
        else:
            importances = np.mean(np.abs(coef), axis=0)
    # else no importances
    if importances is None:
        return None

    # align length
    if len(importances) != len(feature_names):
        # If lengths mismatch, try to reduce or extend safely by mapping by names if available
        # Fallback: pad or trim
        L = min(len(importances), len(feature_names))
        importances = importances[:L]
        feature_names = feature_names[:L]

    fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return fi_df

# ---------------------------
# Tab 0: Regression
# ---------------------------
with tabs[0]:
    st.header("📊 Rating Prediction (Regression Ensemble)")

    if xgb_reg is None or gb_reg is None or top_features is None or label_encoders_reg is None or df is None:
        st.warning("Regression assets or dataset missing. Check sidebar load messages.")
    else:
        st.markdown("Enter inputs for the following top features (these were selected during training).")
        # Build inputs in a form to avoid duplicate widget ids when re-running
        with st.form("reg_form"):
            reg_inputs = {}
            for feat in top_features:
                key = f"reg_{feat}"
                if feat in label_encoders_reg:
                    options = list(label_encoders_reg[feat].classes_)
                    reg_inputs[feat] = st.selectbox(feat, options=options, key=key)
                else:
                    # numeric input
                    reg_inputs[feat] = st.number_input(feat, value=0.0, key=key)
            submit_reg = st.form_submit_button("Predict Rating")

        if submit_reg:
            try:
                input_df = pd.DataFrame([reg_inputs])
                # apply encoders
                for col, le in label_encoders_reg.items():
                    if col in input_df.columns:
                        input_df[col] = le.transform(input_df[col].astype(str).fillna("missing"))

                # Ensure all top_features present & correct ordering
                input_df = input_df[top_features]

                # Predictions
                pred_xgb = xgb_reg.predict(input_df)[0]
                pred_gb = gb_reg.predict(input_df)[0]
                pred_avg = np.mean([pred_xgb, pred_gb])

                # Display predicted values
                c1, c2, c3 = st.columns(3)
                c1.metric("XGBoost Prediction", f"{pred_xgb:.3f}")
                c2.metric("GradientBoosting Prediction", f"{pred_gb:.3f}")
                c3.metric("Ensemble (avg) Prediction", f"{pred_avg:.3f}")

                st.success(f"Predicted Rating (ensemble): **{pred_avg:.3f}**")

                # Evaluate models ONCE on dataset to show metrics
                # Build X from dataset using top_features & label encoders similar to training
                X_full = pd.DataFrame()
                for feat in top_features:
                    if feat in label_encoders_reg:
                        # encode using saved label encoder (if value not seen, map to label - will error; so convert unseen to "missing")
                        vals = df[feat].astype(str).fillna("missing")
                        X_full[feat] = label_encoders_reg[feat].transform(vals)
                    else:
                        # numeric: coerce to numeric then fillna median
                        X_full[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(df[feat].median())

                y_full = pd.to_numeric(df["Rating"], errors="coerce").fillna(df["Rating"].median())

                # Make predictions on full dataset
                try:
                    preds_xgb_full = xgb_reg.predict(X_full)
                    preds_gb_full = gb_reg.predict(X_full)
                    preds_avg_full = (preds_xgb_full + preds_gb_full) / 2.0

                    # Compute metrics for ensemble
                    r2 = r2_score(y_full, preds_avg_full)
                    rmse = mean_squared_error(y_full, preds_avg_full, squared=False)
                    mae = mean_absolute_error(y_full, preds_avg_full)

                    m1, m2, m3 = st.columns(3)
                    m1.metric("R² (ensemble)", f"{r2:.3f}")
                    m2.metric("RMSE (ensemble)", f"{rmse:.3f}")
                    m3.metric("MAE (ensemble)", f"{mae:.3f}")
                except Exception as e:
                    st.warning(f"Couldn't compute dataset metrics for regression: {safe_str(e)}")

                # Feature importance plots
                st.subheader("Feature Importance (XGBoost model)")
                fi_xgb = get_feature_importances_from_model(xgb_reg, top_features)
                if fi_xgb is not None:
                    fig, ax = plt.subplots(figsize=(8, max(3, len(fi_xgb) * 0.4)))
                    sns.barplot(x="importance", y="feature", data=fi_xgb, ax=ax)
                    ax.set_title("Feature Importance (XGBoost)")
                    st.pyplot(fig)
                else:
                    st.info("XGBoost model does not expose feature importances in a standard way.")

                # correlation heatmap between top features and Rating
                st.subheader("Top Features Correlation (heatmap)")
                try:
                    corr_df = X_full.copy()
                    corr_df["target"] = y_full
                    corr = corr_df.corr()
                    fig2, ax2 = plt.subplots(figsize=(7,7))
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
                    st.pyplot(fig2)
                except Exception as e:
                    st.info("Unable to build correlation heatmap: " + safe_str(e))

            except Exception as e:
                st.error("Prediction failed: " + safe_str(e))

# ---------------------------
# Tab 1: Classification
# ---------------------------
with tabs[1]:
    st.header("🎯 Visit Mode Prediction (Classification)")

    if clf_model is None or feature_label_encoders is None or target_label_encoder is None or df is None:
        st.warning("Classification assets or dataset missing. Check sidebar messages.")
    else:
        st.markdown("Provide values for categorical & numeric features used by classifier.")
        # Build input form using saved feature encoders list
        clf_cat_features = list(feature_label_encoders.keys())
        inferred_feats = []
        # If classification_top_features exists (list), use that order
        if classification_top_features is not None:
            all_clf_feats = classification_top_features
        else:
            all_clf_feats = clf_cat_features  # fallback

        with st.form("clf_form"):
            clf_inputs = {}
            # categorical first
            for i, feat in enumerate(all_clf_feats):
                key = f"clf_{feat}"
                if feat in feature_label_encoders:
                    options = list(feature_label_encoders[feat].classes_)
                    clf_inputs[feat] = st.selectbox(feat, options=options, key=key)
                else:
                    # numeric fallback
                    clf_inputs[feat] = st.number_input(feat, value=0.0, key=key)
            submit_clf = st.form_submit_button("Predict Visit Mode")

        if submit_clf:
            try:
                input_df = pd.DataFrame([clf_inputs])
                # encode categorical features
                for col, le in feature_label_encoders.items():
                    if col in input_df.columns:
                        input_df[col] = le.transform(input_df[col].astype(str).fillna("missing"))

                # Predict (clf_model may be pipeline)
                pred_encoded = clf_model.predict(input_df)[0]
                try:
                    pred_label = target_label_encoder.inverse_transform([pred_encoded])[0]
                except Exception:
                    pred_label = pred_encoded

                st.success(f"Predicted Visit Mode: **{safe_str(pred_label)}**")

                # Compute classification metrics on dataset (if target present)
                if "VisitMode_y" in df.columns:
                    # Prepare X and y using classification_top_features if available, else try to use feature_label_encoders keys
                    try:
                        clf_feats = classification_top_features if classification_top_features is not None else list(feature_label_encoders.keys())
                        X_full_clf = pd.DataFrame()
                        for feat in clf_feats:
                            if feat in feature_label_encoders:
                                vals = df[feat].astype(str).fillna("missing")
                                X_full_clf[feat] = feature_label_encoders[feat].transform(vals)
                            else:
                                X_full_clf[feat] = pd.to_numeric(df[feat], errors="coerce").fillna(df[feat].median())
                        # y true encoded earlier during training; we need to transform label strings to encoder classes
                        y_true = df["VisitMode_y"].astype(str).fillna("missing")
                        # If target_label_encoder is LabelEncoder, transform; else assume already numeric
                        try:
                            y_true_enc = target_label_encoder.transform(y_true)
                        except Exception:
                            # maybe y_true is already numeric
                            y_true_enc = pd.to_numeric(y_true, errors="coerce").fillna(0).astype(int)

                        y_pred_full = clf_model.predict(X_full_clf)

                        acc = accuracy_score(y_true_enc, y_pred_full)
                        prec = precision_score(y_true_enc, y_pred_full, average="weighted", zero_division=0)
                        rec = recall_score(y_true_enc, y_pred_full, average="weighted", zero_division=0)
                        f1 = f1_score(y_true_enc, y_pred_full, average="weighted", zero_division=0)

                        a1, a2, a3, a4 = st.columns(4)
                        a1.metric("Accuracy", f"{acc:.3f}")
                        a2.metric("Precision", f"{prec:.3f}")
                        a3.metric("Recall", f"{rec:.3f}")
                        a4.metric("F1 Score", f"{f1:.3f}")
                    except Exception as e:
                        st.info("Unable to compute classification metrics on dataset: " + safe_str(e))
                else:
                    st.info("No VisitMode_y column in dataset to compute metrics.")

                # Feature importance for classifier (if available)
                st.subheader("Classification Feature Importance")
                # We will try to use classification_top_features or keys from feature_label_encoders
                feat_names_for_clf = classification_top_features if classification_top_features is not None else (list(feature_label_encoders.keys()))
                fi_clf = get_feature_importances_from_model(clf_model, feat_names_for_clf)
                if fi_clf is not None:
                    figc, axc = plt.subplots(figsize=(8, max(3, len(fi_clf) * 0.4)))
                    sns.barplot(x="importance", y="feature", data=fi_clf, ax=axc)
                    axc.set_title("Feature Importance (Classification)")
                    st.pyplot(figc)

                    # heatmap between top classification features and target (if in df)
                    if set(feat_names_for_clf).intersection(df.columns):
                        try:
                            tmp = pd.DataFrame()
                            for f in feat_names_for_clf:
                                if f in df.columns:
                                    tmp[f] = pd.to_numeric(df[f], errors="coerce")
                            if "VisitMode_y" in df.columns:
                                tmp["target"] = df["VisitMode_y"]
                            if tmp.shape[1] >= 2:
                                figh, axh = plt.subplots(figsize=(7,7))
                                sns.heatmap(tmp.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axh)
                                st.pyplot(figh)
                        except Exception as e:
                            st.info("Can't draw classification heatmap: " + safe_str(e))
                else:
                    st.info("Feature importance not available for classifier.")

            except Exception as e:
                st.error("Classification prediction failed: " + safe_str(e))

# ---------------------------
# Tab 2: Recommendations
# ---------------------------
with tabs[2]:
    st.header("💡 Recommendation System (CF / CBF / Hybrid)")

    if df is None:
        st.warning("Dataset not loaded; recommendations unavailable.")
    else:
        # helper functions (safe)
        def recommend_cf(user_id, top_n=5):
            try:
                if user_index_map is None or user_similarity is None or user_type_matrix is None:
                    return []
                if user_id not in user_index_map:
                    return []
                idx = user_index_map[user_id]
                sim_arr = user_similarity[idx].toarray()[0] if hasattr(user_similarity[idx], "toarray") else user_similarity[idx]
                sim_scores = list(enumerate(sim_arr))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:]
                similar_users = [user_type_matrix.index[i] for i, _ in sim_scores[:10]]
                similar_users_data = df[df["UserId"].isin(similar_users)]
                avg_ratings = similar_users_data.groupby("AttractionType")["Rating"].mean()
                top_attraction_types = avg_ratings.sort_values(ascending=False).head(top_n).index.tolist()
                # pick attractions for those types, ensure unique attractions and exact count top_n
                candidates = df[df["AttractionType"].isin(top_attraction_types)]["Attraction"].unique().tolist()
                return candidates[:top_n]
            except Exception:
                return []

        def recommend_cbf(user_id, top_n=5):
            try:
                if attr_index_map is None or attraction_similarity is None:
                    return []
                user_data = df[df["UserId"] == user_id]
                if user_data.empty:
                    return []
                visited_types = user_data["AttractionType"].tolist()
                # build similarity scores
                sim_scores = np.zeros(attraction_similarity.shape[0])
                for atype in visited_types:
                    if atype in attr_index_map:
                        idx = attr_index_map[atype]
                        sim_scores += attraction_similarity[idx].toarray()[0]
                # zero out visited indices
                visited_idx = [attr_index_map[atype] for atype in visited_types if atype in attr_index_map]
                for vi in visited_idx:
                    if 0 <= vi < len(sim_scores):
                        sim_scores[vi] = -np.inf
                # get top indices
                top_indices = np.argsort(sim_scores)[::-1]
                # filter out -inf and pick top_n
                picks = []
                for i in top_indices:
                    if sim_scores[i] == -np.inf:
                        continue
                    # map index back to attraction row(s)
                    # In your original cbf code attr_index_map maps AttractType to index of df — we used the same approach
                    row = df.iloc[i]
                    picks.append(row["Attraction"])
                    if len(picks) >= top_n:
                        break
                # ensure uniqueness and exact count (if available)
                picks = ensure_list_unique_preserve_order(picks)[:top_n]
                return [safe_str(x) for x in picks]
            except Exception:
                return []

        def recommend_hybrid(user_id, top_n=5):
            cf_recs = recommend_cf(user_id, top_n=top_n * 2)
            cbf_recs = recommend_cbf(user_id, top_n=top_n * 2)
            combined = ensure_list_unique_preserve_order(cf_recs + cbf_recs)
            return combined[:top_n]

        # UI controls
        left, right = st.columns([3,1])
        with left:
            use_select = st.checkbox("Select existing user id", value=True, key="rec_select_user")
            if use_select:
                user_ids = df["UserId"].dropna().unique().tolist()
                selected_user = st.selectbox("Select User ID", user_ids, key="rec_user_selectbox")
            else:
                selected_user = st.text_input("Enter User ID", key="rec_user_text")

            method = st.radio("Method", ["Hybrid", "Collaborative Filtering", "Content-Based Filtering"], index=0, key="rec_method")
            top_n = st.number_input("Number of recommendations", min_value=1, max_value=20, value=5, key="rec_number")
            get_btn = st.button("Get Recommendations", key="rec_get_btn")
        with right:
            st.markdown("**User history sample**")
            if selected_user:
                st.write("History count:", len(df[df["UserId"] == selected_user]))
            else:
                st.write("No user selected")

        if get_btn:
            if not selected_user:
                st.warning("Please select or enter a user id.")
            else:
                if method == "Collaborative Filtering":
                    recs = recommend_cf(selected_user, top_n=int(top_n))
                elif method == "Content-Based Filtering":
                    recs = recommend_cbf(selected_user, top_n=int(top_n))
                else:
                    recs = recommend_hybrid(selected_user, top_n=int(top_n))

                # If fewer than requested recs found, explicitly inform
                if not recs:
                    st.info("No recommendations found for this user (cold-start / user not in model).")
                else:
                    st.subheader(f"Top {min(len(recs), int(top_n))} recommendations (method: {method})")
                    for i, r in enumerate(recs, 1):
                        st.write(f"{i}. {safe_str(r)}")
                    if len(recs) < int(top_n):
                        st.info(f"Only {len(recs)} unique recommendations available.")

# ---------------------------
# Info Tab
# ---------------------------
with tabs[3]:
    st.header("ℹ️ Info & Notes")
    st.markdown("""
    **Files used (models / pickles)**  
    - Regression: `XGBoost_model.pkl`, `GradientBoosting_model.pkl`, `top_features.pkl`, `label_encoders.pkl`  
    - Classification: `XGBoost_advanced_classifier.pkl`, `feature_label_encoders.pkl`, `target_label_encoder.pkl`, `classification_top_features.pkl`  
    - Recommendations: `cf_data.pkl`, `cbf_data.pkl` (and their internal maps)  
    - Dataset: `Final_dataset.csv` (must contain Rating and VisitMode_y where applicable)

    **Notes**
    - All widgets have unique keys to avoid StreamlitDuplicateElementId errors.
    - Feature importance extraction tries multiple methods; some models/pipelines may hide importances.
    - Recommendation functions return exactly `top_n` items if available, else fewer with a message.
    - If you want improved UI style, we can add images, colors, or CSS-like tweaks.
    """)

# End of app.py
