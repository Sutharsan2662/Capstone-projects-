import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Tourism ML Platform",
    page_icon="🏞️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .recommendation-item {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 0.8rem;
        margin: 0.3rem 0;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
MODEL_DIR = Path(r"C:\Users\C Sutharsan\Downloads\GUVI class notes AIML\Capstone_project\Project_4\Final\models")
FINAL_DATA_PATH = Path(r"C:\Users\C Sutharsan\Downloads\GUVI class notes AIML\Capstone_project\Project_4\Final\Final_dataset.csv")
ENHANCED_DATA_PATH = Path(r"C:\Users\C Sutharsan\Downloads\GUVI class notes AIML\Capstone_project\Project_4\Final\enhanced_dataset.csv")

# Utility functions
@st.cache_data
def load_final_dataset():
    """Load and cache the final dataset (for regression and recommendations)"""
    try:
        # Try the specified path first
        if FINAL_DATA_PATH.exists():
            df = pd.read_csv(FINAL_DATA_PATH)
        else:
            # Fallback to current directory
            df = pd.read_csv("Final_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading Final_dataset.csv: {e}")
        st.info("Please ensure 'Final_dataset.csv' is in the correct location")
        return None

@st.cache_data
def load_enhanced_dataset():
    """Load and cache the enhanced dataset (for classification)"""
    try:
        # Try the specified path first
        if ENHANCED_DATA_PATH.exists():
            df = pd.read_csv(ENHANCED_DATA_PATH)
        else:
            # Fallback to current directory
            df = pd.read_csv("enhanced_dataset.csv")
        return df
    except Exception as e:
        st.error(f"Error loading enhanced_dataset.csv: {e}")
        st.info("Please ensure 'enhanced_dataset.csv' is in the correct location")
        return None

@st.cache_resource
def load_models():
    """Load all models and return them in a dictionary"""
    models = {}
    
    # Regression models
    try:
        models['xgb_reg'] = joblib.load(MODEL_DIR / "XGBoost_model.pkl")
        models['gb_reg'] = joblib.load(MODEL_DIR / "GradientBoosting_model.pkl")
        models['top_features'] = joblib.load(MODEL_DIR / "top_features.pkl")
        models['label_encoders'] = joblib.load(MODEL_DIR / "label_encoders.pkl")
        st.sidebar.success("✅ Regression models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Regression models: {e}")
        models.update({'xgb_reg': None, 'gb_reg': None, 'top_features': None, 'label_encoders': None})
    
    # Classification models
    try:
        models['xgb_clf'] = joblib.load(MODEL_DIR / "XGBoost_advanced_classifier.pkl")
        models['feature_encoders'] = joblib.load(MODEL_DIR / "feature_label_encoder.pkl")
        models['target_encoder'] = joblib.load(MODEL_DIR / "target_label_encoder.pkl")
        models['clf_features'] = joblib.load(MODEL_DIR / "classification_top_features.pkl")
        st.sidebar.success("✅ Classification models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Classification models: {e}")
        models.update({'xgb_clf': None, 'feature_encoders': None, 'target_encoder': None, 'clf_features': None})
    
    # Recommendation models
    try:
        with open(MODEL_DIR / "cf_data.pkl", "rb") as f:
            cf_data = pickle.load(f)
        with open(MODEL_DIR / "cbf_data.pkl", "rb") as f:
            cbf_data = pickle.load(f)
        
        models['user_type_matrix'], models['user_similarity'], models['user_index_map'] = cf_data
        models['tfidf'], models['feature_matrix'], models['attraction_similarity'], models['attr_index_map'] = cbf_data
        st.sidebar.success("✅ Recommendation models loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Recommendation models: {e}")
        models.update({
            'user_type_matrix': None, 'user_similarity': None, 'user_index_map': None,
            'tfidf': None, 'feature_matrix': None, 'attraction_similarity': None, 'attr_index_map': None
        })
    
    return models

# Load data and models
df_final = load_final_dataset()  # For regression and recommendations
df_enhanced = load_enhanced_dataset()  # For classification
models = load_models()

# Main title
st.markdown('<h1 class="main-header">🏞️ Tourism Experience ML Platform</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.selectbox("Choose a task:", ["🏠 Home", "📊 Rating Prediction", "🎯 Visit Mode Prediction", "💡 Recommendations"])

# Home page
if page == "🏠 Home":
    st.markdown("## Welcome to the Tourism Experience ML Platform!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Rating Prediction
        Predict tourism experience ratings using:
        - XGBoost Regressor
        - Gradient Boosting Regressor
        - Ensemble predictions
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Visit Mode Prediction
        Classify visit modes with:
        - Advanced feature encoding
        - XGBoost Classification
        - Performance metrics
        """)
    
    with col3:
        st.markdown("""
        ### 💡 Recommendations
        Get personalized recommendations:
        - Collaborative Filtering
        - Content-Based Filtering
        - Hybrid approach
        """)
    
    if df_final is not None:
        st.markdown("## 📈 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Final Dataset Records", len(df_final))
        with col2:
            st.metric("Enhanced Dataset Records", len(df_enhanced) if df_enhanced is not None else "N/A")
        with col3:
            st.metric("Unique Users", df_final['UserId'].nunique() if 'UserId' in df_final.columns else "N/A")
        with col4:
            st.metric("Avg Rating", f"{df_final['Rating'].mean():.2f}" if 'Rating' in df_final.columns else "N/A")

# Rating Prediction page
elif page == "📊 Rating Prediction":
    st.markdown("## 📊 Tourism Rating Prediction")
    
    if models['xgb_reg'] is None or models['top_features'] is None:
        st.error("Regression models not available. Please check if model files exist.")
    else:
        st.markdown("### Enter feature values for prediction:")
        
        # Create input form
        with st.form("regression_form"):
            inputs = {}
            
            # Create columns for better layout
            n_features = len(models['top_features'])
            cols = st.columns(min(3, n_features))
            
            for i, feature in enumerate(models['top_features']):
                col_idx = i % len(cols)
                
                with cols[col_idx]:
                    if feature in models['label_encoders']:
                        # Categorical feature
                        options = list(models['label_encoders'][feature].classes_)
                        inputs[feature] = st.selectbox(f"{feature}", options, key=f"reg_{feature}")
                    else:
                        # Numerical feature
                        inputs[feature] = st.number_input(f"{feature}", value=0.0, key=f"reg_{feature}")
            
            submit_button = st.form_submit_button("🔮 Predict Rating", use_container_width=True)
        
        if submit_button:
            try:
                # Prepare input data
                input_df = pd.DataFrame([inputs])
                
                # Apply label encoders
                for col, encoder in models['label_encoders'].items():
                    if col in input_df.columns:
                        input_df[col] = encoder.transform(input_df[col].astype(str))
                
                # Make predictions
                xgb_pred = models['xgb_reg'].predict(input_df[models['top_features']])[0]
                gb_pred = models['gb_reg'].predict(input_df[models['top_features']])[0]
                ensemble_pred = (xgb_pred + gb_pred) / 2
                
                # Display results
                st.markdown("### Prediction Results")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f'<div class="metric-card"><h4>XGBoost</h4><h2 style="color: #2c3e50;">{xgb_pred:.3f}</h2></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="metric-card"><h4>Gradient Boosting</h4><h2 style="color: #2c3e50;">{gb_pred:.3f}</h2></div>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<div class="metric-card"><h4>Ensemble</h4><h2 style="color: #2c3e50;">{ensemble_pred:.3f}</h2></div>', unsafe_allow_html=True)
                
                st.markdown(f'<div class="prediction-result">Final Predicted Rating: {ensemble_pred:.3f}</div>', unsafe_allow_html=True)
                
                # Feature importance visualization
                if hasattr(models['xgb_reg'], 'feature_importances_'):
                    st.markdown("### 📊 Feature Importance")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance_df = pd.DataFrame({
                        'feature': models['top_features'],
                        'importance': models['xgb_reg'].feature_importances_
                    }).sort_values('importance', ascending=True)
                    
                    sns.barplot(data=importance_df, x='importance', y='feature', ax=ax)
                    ax.set_title('XGBoost Feature Importance')
                    ax.set_xlabel('Importance')
                    st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Classification page
elif page == "🎯 Visit Mode Prediction":
    st.markdown("## 🎯 Visit Mode Classification")
    
    if models['xgb_clf'] is None or models['feature_encoders'] is None:
        st.error("Classification models not available. Please check if model files exist.")
    elif df_enhanced is None:
        st.error("Enhanced dataset not available. Please check if enhanced_dataset.csv exists.")
    else:
        st.markdown("### Enter feature values for classification:")
        st.info("Using enhanced_dataset.csv for classification")
        
        # Define features outside the form so they're accessible later
        numerical_features = [
            'VisitMonth', 'VisitQuarter', 'VisitYear',
            'continent_mode_prob_Business', 'continent_mode_prob_Couples',
            'user_pct_Couples', 'user_pct_Family', 'user_pct_Friends', 'user_pct_Business',
            'user_travel_diversity', 'attraction_avg_rating_before', 'user_previous_visits',
            'city_popularity', 'user_avg_rating_before', 'user_attraction_compatibility',
            'sin_month', 'cos_month', 'month_mode_prob_Business', 'month_mode_prob_Family',
            'month_mode_prob_Friends', 'month_mode_prob_Couples'
        ]
        categorical_features = [
            'VisitSeason', 'Continent', 'Region', 'Country', 'CityName',
            'AttractionType', 'prev_visit_mode'
        ]
        all_features = numerical_features + categorical_features
        
        with st.form("classification_form"):
            inputs = {}
            
            st.markdown("#### Categorical Features")
            cat_cols = st.columns(3)
            for i, feature in enumerate(categorical_features):
                col_idx = i % 3
                with cat_cols[col_idx]:
                    if feature in models['feature_encoders']:
                        options = list(models['feature_encoders'][feature].classes_)
                        inputs[feature] = st.selectbox(f"{feature}", options, key=f"clf_{feature}")
                    elif feature in df_enhanced.columns:
                        unique_vals = df_enhanced[feature].dropna().unique().tolist()
                        if len(unique_vals) <= 100:
                            inputs[feature] = st.selectbox(f"{feature}", unique_vals, key=f"clf_{feature}")
                        else:
                            inputs[feature] = st.text_input(f"{feature}", key=f"clf_{feature}")
                    else:
                        inputs[feature] = st.text_input(f"{feature}", key=f"clf_{feature}")
            
            st.markdown("#### Numerical Features")
            num_cols = st.columns(4)
            for i, feature in enumerate(numerical_features):
                col_idx = i % 4
                with num_cols[col_idx]:
                    if feature in df_enhanced.columns:
                        mean_val = df_enhanced[feature].mean()
                        min_val = df_enhanced[feature].min()
                        max_val = df_enhanced[feature].max()
                        inputs[feature] = st.number_input(
                            f"{feature}", 
                            value=float(mean_val) if not pd.isna(mean_val) else 0.0,
                            min_value=float(min_val) if not pd.isna(min_val) else 0.0,
                            max_value=float(max_val) if not pd.isna(max_val) else 100.0,
                            step=0.01,
                            key=f"clf_{feature}"
                        )
                    else:
                        inputs[feature] = st.number_input(f"{feature}", value=0.0, step=0.01, key=f"clf_{feature}")
            
            submit_button = st.form_submit_button("Predict Visit Mode", use_container_width=True)
        
        if submit_button:
            try:
                # Prepare input data
                input_df = pd.DataFrame([inputs])
                
                # Apply encoders for categorical features
                for col, encoder in models['feature_encoders'].items():
                    if col in input_df.columns:
                        try:
                            input_df[col] = encoder.transform(input_df[col].astype(str))
                        except Exception:
                            # Handle unseen categories
                            input_df[col] = 0  # Default encoding for unseen values
                
                # Ensure all required features are present
                for feature in all_features: # Corrected from 'features' to 'all_features'
                    if feature not in input_df.columns:
                        input_df[feature] = 0.0
                
                # Reorder columns to match training data
                input_df = input_df[all_features]
                
                # Make prediction
                prediction = models['xgb_clf'].predict(input_df)[0]
                
                # Decode prediction if target encoder exists
                if models['target_encoder'] is not None:
                    try:
                        predicted_label = models['target_encoder'].inverse_transform([prediction])[0]
                    except:
                        predicted_label = prediction
                else:
                    predicted_label = prediction
                
                st.markdown("### Classification Result")
                st.markdown(f'<div class="prediction-result">Predicted Visit Mode: {predicted_label}</div>', unsafe_allow_html=True)
                
                # Prediction probabilities if available
                if hasattr(models['xgb_clf'], 'predict_proba'):
                    try:
                        probabilities = models['xgb_clf'].predict_proba(input_df)[0]
                        classes = models['xgb_clf'].classes_
                        
                        # Decode class names if possible
                        if models['target_encoder'] is not None:
                            try:
                                decoded_classes = models['target_encoder'].inverse_transform(classes)
                            except:
                                decoded_classes = classes
                        else:
                            decoded_classes = classes
                        
                        st.markdown("### Prediction Probabilities")
                        prob_df = pd.DataFrame({
                            'Class': decoded_classes,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        bars = sns.barplot(data=prob_df, x='Probability', y='Class', ax=ax, palette='viridis')
                        ax.set_title('Class Probabilities', fontsize=14, fontweight='bold')
                        ax.set_xlabel('Probability', fontsize=12)
                        ax.set_ylabel('Visit Mode', fontsize=12)
                        
                        # Add probability values on bars
                        for i, (idx, row) in enumerate(prob_df.iterrows()):
                            ax.text(row['Probability'] + 0.01, i, f'{row["Probability"]:.3f}', 
                                   va='center', fontweight='bold')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    except Exception as e:
                        st.info(f"Could not display probabilities: {e}")
                
            except Exception as e:
                st.error(f"Classification failed: {e}")
                st.info("Please check if all required features are available in the enhanced dataset.")
        
        # Show enhanced dataset info
        st.markdown("### 📊 Enhanced Dataset Info")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(df_enhanced))
        with col2:
            st.metric("Features", len(df_enhanced.columns))
        with col3:
            if 'VisitMode_y' in df_enhanced.columns:
                st.metric("Unique Classes", df_enhanced['VisitMode_y'].nunique())
            else:
                st.metric("Unique Classes", "N/A")

# Recommendations page
elif page == "💡 Recommendations":
    st.markdown("## 💡 Tourism Recommendations")
    
    if df_final is None:
        st.error("Final dataset not available for recommendations. Please check if Final_dataset.csv exists.")
    else:
        st.info("Using Final_dataset.csv for recommendations")
        
        # User selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_ids = sorted(df_final['UserId'].unique()) if 'UserId' in df_final.columns else []
            if user_ids:
                selected_user = st.selectbox("Select User ID:", user_ids)
            else:
                selected_user = st.number_input("Enter User ID:", value=1, min_value=1)
        
        with col2:
            num_recommendations = st.slider("Number of recommendations:", 1, 10, 5)
        
        # Recommendation functions
        def get_collaborative_recommendations(user_id, top_n=5):
            try:
                if (models['user_index_map'] is None or 
                    models['user_similarity'] is None or 
                    models['user_type_matrix'] is None):
                    return ["Collaborative filtering model not available"]
                
                # Simple fallback collaborative filtering using dataset directly
                user_data = df_final[df_final['UserId'] == user_id]
                if user_data.empty:
                    return ["User not found in the system"]
                
                # Get user's visited attractions
                user_visited = user_data['Attraction'].tolist()
                
                # Find users with similar preferences (high ratings for same attraction types)
                user_attraction_types = user_data['AttractionType'].unique()
                
                # Get other users who liked similar attraction types
                similar_users_data = df_final[
                    (df_final['AttractionType'].isin(user_attraction_types)) & 
                    (df_final['Rating'] >= 4.0) &  # High ratings only
                    (df_final['UserId'] != user_id)  # Exclude current user
                ]
                
                if similar_users_data.empty:
                    return ["No similar users found"]
                
                # Get top-rated attractions from similar users that current user hasn't visited
                recommendations = (similar_users_data[~similar_users_data['Attraction'].isin(user_visited)]
                                 .groupby('Attraction')['Rating']
                                 .agg(['mean', 'count'])
                                 .query('count >= 2')  # At least 2 ratings
                                 .sort_values('mean', ascending=False)
                                 .head(top_n)
                                 .index.tolist())
                
                return recommendations if recommendations else ["No new attractions found from similar users"]
                
            except Exception as e:
                return [f"Error in collaborative filtering: {str(e)}"]
        
        def get_content_based_recommendations(user_id, top_n=5):
            try:
                user_data = df_final[df_final['UserId'] == user_id]
                if user_data.empty:
                    return ["User has no history in the system"]
                
                # Get user's preferences (high-rated attraction types)
                user_preferences = user_data[user_data['Rating'] >= 4.0]['AttractionType'].unique()
                user_visited = user_data['Attraction'].tolist()
                
                if len(user_preferences) == 0:
                    # If no high ratings, use all visited types
                    user_preferences = user_data['AttractionType'].unique()
                
                # Find highly-rated attractions of similar types that user hasn't visited
                similar_attractions = df_final[
                    (df_final['AttractionType'].isin(user_preferences)) &
                    (~df_final['Attraction'].isin(user_visited)) &
                    (df_final['Rating'] >= 4.0)
                ]
                
                if similar_attractions.empty:
                    # Fallback: get any highly-rated attractions user hasn't visited
                    similar_attractions = df_final[
                        (~df_final['Attraction'].isin(user_visited)) &
                        (df_final['Rating'] >= 4.5)
                    ]
                
                if similar_attractions.empty:
                    return ["No similar attractions found"]
                
                # Get top attractions by average rating
                recommendations = (similar_attractions.groupby('Attraction')['Rating']
                                 .agg(['mean', 'count'])
                                 .sort_values(['mean', 'count'], ascending=[False, False])
                                 .head(top_n)
                                 .index.tolist())
                
                return recommendations if recommendations else ["No content-based recommendations found"]
                
            except Exception as e:
                return [f"Error in content-based filtering: {str(e)}"]
        
        def get_hybrid_recommendations(user_id, top_n=5):
            cf_recs = get_collaborative_recommendations(user_id, top_n)
            cbf_recs = get_content_based_recommendations(user_id, top_n)
            
            # Combine and remove duplicates while preserving order
            combined = []
            seen = set()
            
            for rec_list in [cf_recs, cbf_recs]:
                for rec in rec_list:
                    if rec not in seen and not rec.startswith("Error") and rec != "User not found in the system":
                        combined.append(rec)
                        seen.add(rec)
            
            return combined[:top_n] if combined else ["No recommendations available"]
        
        # Generate recommendations button
        if st.button("🚀 Get Recommendations", use_container_width=True):
            # Show user's history
            user_history = df_final[df_final['UserId'] == selected_user] if 'UserId' in df_final.columns else pd.DataFrame()
            
            if not user_history.empty:
                st.markdown("### 📚 User's History")
                history_summary = user_history.groupby('Attraction')['Rating'].mean().sort_values(ascending=False)
                for attraction, rating in history_summary.head(5).items():
                    st.markdown(f"• **{attraction}** (Rating: {rating:.2f})")
            
            st.markdown("### 🎯 Recommendations")
            
            # Create tabs for different recommendation types
            rec_tabs = st.tabs(["🤝 Collaborative Filtering", "📝 Content-Based", "⚡ Hybrid"])
            
            with rec_tabs[0]:
                cf_recommendations = get_collaborative_recommendations(selected_user, num_recommendations)
                st.markdown("#### Collaborative Filtering Recommendations")
                for i, rec in enumerate(cf_recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">{i}. {rec}</div>', unsafe_allow_html=True)
            
            with rec_tabs[1]:
                cbf_recommendations = get_content_based_recommendations(selected_user, num_recommendations)
                st.markdown("#### Content-Based Recommendations")
                for i, rec in enumerate(cbf_recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">{i}. {rec}</div>', unsafe_allow_html=True)
            
            with rec_tabs[2]:
                hybrid_recommendations = get_hybrid_recommendations(selected_user, num_recommendations)
                st.markdown("#### Hybrid Recommendations")
                for i, rec in enumerate(hybrid_recommendations, 1):
                    st.markdown(f'<div class="recommendation-item">{i}. {rec}</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Tourism Experience ML Platform")