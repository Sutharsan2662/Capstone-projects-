import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🎵 Mood Music Predictor",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🎵 Mood Music Predictor</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">
        Predict your mood based on physiological parameters and get personalized music recommendations!
    </p>
</div>
""", unsafe_allow_html=True)

# Load model and encoder function
@st.cache_resource
def load_models():
    try:
        model = joblib.load("XGBoost_advanced_classifier.pkl")
        mood_encoder = joblib.load("mood_encoder.pkl")
        return model, mood_encoder
    except FileNotFoundError as e:
        st.error(f"⚠️ Could not load model files. Please ensure the model files are in the correct directory.")
        st.stop()

# Load data for visualization
@st.cache_data
def load_data():
    try:
        data = pd.read_csv("mood_music_dataset.csv")
        return data
    except FileNotFoundError:
        st.warning("Dataset file not found. Feature visualization will be limited.")
        return None

# Initialize models and data
model, mood_encoder = load_models()
data = load_data()

# Mood to song mapping
mood_to_song = {
    mood: song for mood, song in zip(
        mood_encoder.classes_,
        ["Anirudh songs 🎉",
         "Calm Acoustic or Soft Piano U1 drugs 🎶",
         "Healing music Endrendrum Raja 🤘",
         "Lo-fi or Jazz to chill out Periya Bhai ☕"]
    )
}

# Time mapping
time_mapping = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}

# Sidebar for inputs
st.sidebar.header("🔧 Input Parameters")
st.sidebar.markdown("---")

# Input widgets
heart_rate = st.sidebar.slider(
    "💓 Heart Rate (bpm)", 
    min_value=50, 
    max_value=120, 
    value=75, 
    step=1,
    help="Your current heart rate in beats per minute"
)

blink_rate = st.sidebar.slider(
    "👀 Blink Rate (per min)", 
    min_value=10, 
    max_value=50, 
    value=20, 
    step=1,
    help="Number of blinks per minute"
)

temperature = st.sidebar.slider(
    "🌡️ Body Temperature (°C)", 
    min_value=35.0, 
    max_value=40.0, 
    value=37.0, 
    step=0.1,
    help="Your body temperature in Celsius"
)

score = st.sidebar.slider(
    "⭐ Score (0-100)", 
    min_value=0, 
    max_value=100, 
    value=50, 
    step=1,
    help="Overall wellness/energy score"
)

time_input = st.sidebar.selectbox(
    "⏰ Time of Day",
    options=["Morning", "Afternoon", "Evening", "Night"],
    index=0,
    help="Current time of the day"
)

# Convert time to numeric
time_of_day = time_mapping[time_input]

# Prediction button
predict_button = st.sidebar.button("🔮 Predict Mood", type="primary")

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📊 Current Input Parameters")
    
    # Display current inputs in a nice format
    input_df = pd.DataFrame({
        'Parameter': ['Heart Rate', 'Blink Rate', 'Body Temperature', 'Score', 'Time of Day'],
        'Value': [f"{heart_rate} bpm", f"{blink_rate} /min", f"{temperature} °C", f"{score}/100", time_input],
        'Status': ['Normal' if 60 <= heart_rate <= 100 else 'Attention',
                  'Normal' if 15 <= blink_rate <= 25 else 'Attention',
                  'Normal' if 36.1 <= temperature <= 37.2 else 'Attention',
                  'Good' if score >= 70 else 'Fair' if score >= 40 else 'Low',
                  'Active' if time_input in ['Morning', 'Afternoon'] else 'Rest']
    })
    
    st.dataframe(input_df, use_container_width=True, hide_index=True)

with col2:
    st.header("🎯 Prediction Results")
    
    if predict_button:
        # Prepare features for prediction
        features = np.array([[heart_rate, blink_rate, temperature, score, time_of_day]])
        
        # Make prediction
        pred_numeric = model.predict(features)[0]
        pred_proba = model.predict_proba(features)[0]
        pred_label = mood_encoder.inverse_transform([pred_numeric])[0]
        
        # Get song recommendation
        song_suggestion = mood_to_song.get(pred_label, "Any genre you enjoy 🎵")
        
        # Display prediction in a beautiful box
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎭 Predicted Mood</h2>
            <h1 style="margin: 1rem 0;">{pred_label}</h1>
            <h3>🎵 Recommended Music</h3>
            <p style="font-size: 1.2rem;">{song_suggestion}</p>
            <p style="margin-top: 1rem; font-size: 0.9rem; opacity: 0.8;">
                Confidence: {max(pred_proba)*100:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability distribution
        st.subheader("📈 Mood Probability Distribution")
        prob_df = pd.DataFrame({
            'Mood': mood_encoder.classes_,
            'Probability': pred_proba * 100
        }).sort_values('Probability', ascending=True)
        
        fig = px.bar(prob_df, x='Probability', y='Mood', orientation='h',
                    title="Confidence for Each Mood",
                    color='Probability',
                    color_continuous_scale='viridis')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# Feature Analysis Section
if data is not None:
    st.markdown("---")
    st.header("📈 Feature Analysis & Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["🔥 Correlation Heatmap", "📊 Feature Distributions"])
    
    with tab1:
        st.subheader("Correlation Heatmap")
        # Create correlation matrix
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            fig, ax = plt.subplots(figsize=(12, 8))
            correlation_matrix = data[numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", 
                       center=0, square=True, ax=ax, fmt='.2f')
            plt.title("Feature Correlation Matrix", fontsize=16, fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Highlight strongest correlations
            st.subheader("🔍 Key Insights")
            # Find strongest positive and negative correlations
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_pairs.append({
                        'Feature 1': correlation_matrix.columns[i],
                        'Feature 2': correlation_matrix.columns[j],
                        'Correlation': correlation_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False).head(5)
            st.dataframe(corr_df, use_container_width=True, hide_index=True)
    
    with tab2:
        st.subheader("Feature Distributions")
        
        # Select features to plot
        numeric_features = ['Heart Rate', 'Blink Rate', 'Skin Temperature', 'Score']
        available_features = [f for f in numeric_features if f in data.columns]
        
        if available_features:
            selected_feature = st.selectbox("Select feature to analyze:", available_features)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribution plot
                fig = px.histogram(data, x=selected_feature, nbins=30,
                                 title=f"Distribution of {selected_feature}",
                                 marginal="box")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Box plot by mood (if mood column exists)
                if 'Mood' in data.columns:
                    fig = px.box(data, x='Mood', y=selected_feature,
                               title=f"{selected_feature} by Mood")
                    fig.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Mood column not available for comparison")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>🎵 Mood Music Predictor | Built with Streamlit & XGBoost</p>
    <p>Enter your physiological parameters and discover your mood with personalized music recommendations!</p>
</div>
""", unsafe_allow_html=True)