"""
Smart Driver Monitoring - Streamlit Demo App
Run with: streamlit run demos/streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Page config
st.set_page_config(
    page_title="Smart Driver Monitoring",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Smart Driver Monitoring Dashboard")
st.markdown("AI & Analytics for Driver Safety and Performance")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["📊 Rating Prediction", "💬 Feedback Analysis", "⚠️ Violation Detection", "📄 Document Check"]
)

# Load models if available
@st.cache_resource
def load_models():
    import joblib
    models = {}
    src_dir = Path(__file__).parent.parent / "src"
    
    try:
        models['rating'] = joblib.load(src_dir / "rating_model.joblib")
    except:
        models['rating'] = None
    
    try:
        models['sentiment_model'] = joblib.load(src_dir / "sentiment_model.joblib")
        models['tfidf'] = joblib.load(src_dir / "tfidf_vectorizer.joblib")
    except:
        models['sentiment_model'] = None
        models['tfidf'] = None
    
    try:
        models['violation'] = joblib.load(src_dir / "violation_model.joblib")
    except:
        models['violation'] = None
    
    return models

models = load_models()


# ============ RATING PREDICTION ============
if page == "📊 Rating Prediction":
    st.header("📊 Driver Rating Prediction")
    st.markdown("Predict driver ratings based on telematics features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Features")
        speed_mean = st.slider("Average Speed (km/h)", 20, 100, 45)
        speed_std = st.slider("Speed Variation (std)", 0, 30, 10)
        hard_brake_count = st.slider("Hard Brake Count", 0, 20, 2)
        overspeed_count = st.slider("Overspeed Count", 0, 15, 1)
        harsh_turn_count = st.slider("Harsh Turn Count", 0, 10, 1)
    
    with col2:
        st.subheader("Additional Info")
        avg_trip_duration = st.slider("Avg Trip Duration (sec)", 300, 3600, 1200)
        total_distance = st.slider("Total Distance (km)", 10, 500, 100)
        trip_count = st.slider("Number of Trips", 5, 50, 20)
        speed_max = st.slider("Max Speed (km/h)", 40, 140, 80)
    
    if st.button("Predict Rating", type="primary"):
        features = np.array([[
            speed_mean, speed_std, speed_max, hard_brake_count,
            overspeed_count, harsh_turn_count, avg_trip_duration,
            total_distance, trip_count
        ]])
        
        if models['rating'] is not None:
            prediction = models['rating'].predict(features)[0]
            rating = round(prediction, 1)
        else:
            # Simple heuristic if model not trained
            rating = 5 - (hard_brake_count * 0.2) - (overspeed_count * 0.3) - ((speed_mean - 40) / 50)
            rating = max(1, min(5, rating))
        
        st.success(f"### Predicted Rating: ⭐ {rating:.1f} / 5.0")
        
        # Show feature impact
        st.subheader("Feature Analysis")
        impact_data = {
            'Feature': ['Speed Mean', 'Hard Brakes', 'Overspeed', 'Harsh Turns'],
            'Value': [speed_mean, hard_brake_count, overspeed_count, harsh_turn_count],
            'Impact': ['🟡 Medium', '🔴 High' if hard_brake_count > 5 else '🟢 Low',
                      '🔴 High' if overspeed_count > 3 else '🟢 Low',
                      '🟡 Medium' if harsh_turn_count > 2 else '🟢 Low']
        }
        st.dataframe(pd.DataFrame(impact_data), hide_index=True)


# ============ FEEDBACK ANALYSIS ============
elif page == "💬 Feedback Analysis":
    st.header("💬 Passenger Feedback Analysis")
    st.markdown("Analyze sentiment from passenger feedback")
    
    feedback_text = st.text_area(
        "Enter passenger feedback:",
        "Driver was very professional and the ride was smooth. Arrived on time!",
        height=100
    )
    
    if st.button("Analyze Sentiment", type="primary"):
        # Try using transformer model
        try:
            from transformers import pipeline
            sentiment_pipe = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            result = sentiment_pipe(feedback_text)[0]
            sentiment = result['label']
            confidence = result['score']
        except:
            # Simple keyword-based fallback
            positive_words = ['great', 'excellent', 'good', 'professional', 'smooth', 'safe', 'clean', 'polite']
            negative_words = ['rude', 'dangerous', 'late', 'dirty', 'aggressive', 'unsafe', 'rough']
            
            text_lower = feedback_text.lower()
            pos_count = sum(1 for w in positive_words if w in text_lower)
            neg_count = sum(1 for w in negative_words if w in text_lower)
            
            if pos_count > neg_count:
                sentiment = "POSITIVE"
                confidence = 0.7 + (pos_count * 0.05)
            elif neg_count > pos_count:
                sentiment = "NEGATIVE"
                confidence = 0.7 + (neg_count * 0.05)
            else:
                sentiment = "NEUTRAL"
                confidence = 0.5
            
            confidence = min(confidence, 0.99)
        
        # Display result
        if sentiment == "POSITIVE":
            st.success(f"### Sentiment: 😊 {sentiment}")
        elif sentiment == "NEGATIVE":
            st.error(f"### Sentiment: 😠 {sentiment}")
        else:
            st.info(f"### Sentiment: 😐 {sentiment}")
        
        st.metric("Confidence", f"{confidence:.1%}")
        
        # Show keywords
        st.subheader("Detected Keywords")
        keywords = {
            'positive': ['professional', 'smooth', 'safe', 'on time', 'great', 'excellent'],
            'negative': ['rude', 'aggressive', 'late', 'unsafe', 'dirty']
        }
        
        found_pos = [w for w in keywords['positive'] if w in feedback_text.lower()]
        found_neg = [w for w in keywords['negative'] if w in feedback_text.lower()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ Positive:", ", ".join(found_pos) if found_pos else "None")
        with col2:
            st.write("❌ Negative:", ", ".join(found_neg) if found_neg else "None")


# ============ VIOLATION DETECTION ============
elif page == "⚠️ Violation Detection":
    st.header("⚠️ Driving Violation Detection")
    st.markdown("Detect violations from telematics data")
    
    st.subheader("Upload Telematics CSV")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        if st.button("Detect Violations", type="primary"):
            # Check for violation indicators
            violations = []
            
            if 'speed' in df.columns:
                overspeed = df[df['speed'] > 80]
                if len(overspeed) > 0:
                    violations.append(f"🚨 Overspeed: {len(overspeed)} instances")
            
            if 'hard_brake' in df.columns:
                hard_brakes = df[df['hard_brake'] == 1]
                if len(hard_brakes) > 0:
                    violations.append(f"🛑 Hard Brakes: {len(hard_brakes)} instances")
            
            if 'harsh_turn' in df.columns:
                harsh_turns = df[df['harsh_turn'] == 1]
                if len(harsh_turns) > 0:
                    violations.append(f"↩️ Harsh Turns: {len(harsh_turns)} instances")
            
            if violations:
                st.warning("### Violations Detected!")
                for v in violations:
                    st.write(v)
            else:
                st.success("### No violations detected! ✅")
    else:
        st.info("Upload a telematics CSV to detect violations")
        
        # Demo with sample data
        st.subheader("Or try with sample data:")
        if st.button("Generate Sample Data"):
            np.random.seed(42)
            sample_df = pd.DataFrame({
                'trip_id': [f"T{i:03d}" for i in range(20)],
                'speed': np.random.normal(50, 20, 20).clip(20, 120),
                'hard_brake': (np.random.rand(20) < 0.15).astype(int),
                'harsh_turn': (np.random.rand(20) < 0.1).astype(int)
            })
            st.dataframe(sample_df)
            
            violations_count = (sample_df['speed'] > 80).sum() + sample_df['hard_brake'].sum()
            st.metric("Total Violations", violations_count)


# ============ DOCUMENT CHECK ============
elif page == "📄 Document Check":
    st.header("📄 Document Forgery Detection")
    st.markdown("Validate driver license documents")
    
    st.subheader("Manual Text Check")
    license_text = st.text_area(
        "Paste OCR-extracted text from license:",
        """DRIVING LICENSE
Name: John Ahmed Smith
License No: UAE-DXB-123456
DOB: 1985-03-15
Expiry: 2026-03-15""",
        height=150
    )
    
    if st.button("Validate Document", type="primary"):
        import re
        
        issues = []
        
        # Check license number
        license_pattern = r'UAE-[A-Z]{3}-\d{6}'
        if not re.search(license_pattern, license_text):
            issues.append("❌ Invalid license number format")
        else:
            st.write("✅ License number format valid")
        
        # Check date format
        date_pattern = r'\d{4}-\d{2}-\d{2}'
        dates = re.findall(date_pattern, license_text)
        
        if not dates:
            issues.append("❌ No valid dates found")
        else:
            for date_str in dates:
                try:
                    year, month, day = map(int, date_str.split('-'))
                    if not (1 <= month <= 12 and 1 <= day <= 31):
                        issues.append(f"❌ Invalid date: {date_str}")
                    else:
                        st.write(f"✅ Valid date: {date_str}")
                except:
                    issues.append(f"❌ Cannot parse date: {date_str}")
        
        # Check for suspicious characters
        suspicious = re.findall(r'[0-9](?=[a-zA-Z])|[a-zA-Z](?=[0-9])', license_text)
        if 'License No' not in license_text and suspicious:
            issues.append("⚠️ Possible character substitution detected")
        
        # Display result
        st.markdown("---")
        if issues:
            st.error("### ⚠️ Potential Forgery Indicators")
            for issue in issues:
                st.write(issue)
            st.metric("Forgery Risk", f"High ({len(issues)} issues)")
        else:
            st.success("### ✅ Document appears valid")
            st.metric("Forgery Risk", "Low")

# Footer
st.markdown("---")
st.markdown("*Smart Driver Monitoring System - AI & Analytics Demo*")
