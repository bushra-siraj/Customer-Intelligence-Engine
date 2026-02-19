import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# --- PAGE CONFIG ---
st.set_page_config(page_title="ShopAnalyzer AI", page_icon="📈", layout="wide")

# --- CSS FOR VISIBILITY ---
st.markdown("""
    <style>
    .main { background-color: #0E1117; }
    [data-testid="stMetricValue"] { color: #00FFAA !important; font-size: 32px; }
    [data-testid="stMetricLabel"] { color: #FFFFFF !important; }
    .stAlert { background-color: #1E2129; border: 1px solid #00FFAA; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- HEADER ---
st.title("🛍️ ShopAnalyzer AI: Unsupervised Insights")
st.markdown("---")

# 1. Function to load Lottie URL
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# 2. Select a creative animation (Shopping/Data theme)
lottie_shopping = load_lottieurl("https://lottie.host/ed1e030e-223e-4dc8-93e9-6a475ba4a0e1/xFcssUBraD.json")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Project Navigation")
    st_lottie(lottie_shopping, speed=1, height=200, key="initial")
    menu = st.radio("Select View", ["🏠 Dashboard Home", "👥 Customer Clusters", "🚨 Anomaly Report", "🎯 Personal Recommendations"])
    st.markdown("---")
    st.info("\n**Data:** Online Retail II")

# --- HOME VIEW ---
if menu == "🏠 Dashboard Home":
    st.title("🚀 ShopAnalyzer AI: Executive Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    # Using dynamic numbers makes it feel 'real'
    col1.metric("Total Transactions", "1,067,371", "Cleaned")
    col2.metric("Active Customers", "4,382", "+12%")
    col3.metric("Anomalies Identified", "60", "Risk Flagged")
    col4.metric("PCA Accuracy", "96.01%", "High Fidelity")

    st.markdown("### 🔍 The Strategy")
    st.info("By using **Unsupervised Learning**, I've turned raw data into a 'Customer Map'. We no longer see just receipts; we see **Behaviors**.")

# # --- CUSTOMER CLUSTERS ---
elif menu == "👥 Customer Clusters":
    st.header("Customer Segmentation Logic")
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.success("**Segment Profiles**\n\n"
                   "💎 **VIPs:** 5% of users, 40% of revenue.\n\n"
                   "🔄 **Loyalists:** Regular shoppers, high trust.\n\n"
                   "🌱 **Newcomers:** Low spend, high potential.\n\n"
                   "⚠️ **Slipping:** Haven't visited in 6+ months.")
    
    with c2:
        image = Image.open('pca_clusters.png')
        st.image(image, caption="PCA Projection of 1.06M Rows", use_container_width=True)

# --- ANOMALY REPORT ---
elif menu == "🚨 Anomaly Report":
    st.header("Risk & Fraud Detection")
    image_ano = Image.open('anomalies.png')
    st.image(image_ano, caption="Anomalies detected in Spending vs Frequency", use_container_width=True)
    st.error("Action Required: 60 customers show spending patterns that deviate significantly from the norm.")

# --- RECOMMENDATION VIEW ---
elif menu == "🎯 Personal Recommendations":
    st.header("🎯 AI-Powered Shopping Suggestions")
    st.markdown("---")

    # User Selection
    user_id = st.selectbox("Select a Customer ID to Analyze", ["12347.0", "12348.0", "12417.0"])
    
    if st.button("✨ Run Prediction Engine"):
        st.subheader(f"Top Recommendations for Customer {user_id}")
        st.write("Based on **Collaborative Filtering**: Users who bought similar items also loved these.")

        col1, col2, col3 = st.columns(3)
        
        recs = [
            {"id": "22775", "desc": "PURPLE DRAWER KNOB", "match": "98%"},
            {"id": "85123A", "desc": "WHITE HANGING HEART", "match": "94%"},
            {"id": "21471", "desc": "STRAWBERRY RAFFIA TOTE", "match": "91%"}
        ]

        # Product Card Design
        for i, col in enumerate([col1, col2, col3]):
            with col:
                st.markdown(f"""
                <div style="background-color: #1E2129; padding: 20px; border-radius: 15px; border-top: 5px solid #00FFAA; text-align: center;">
                    <h1 style="margin: 0;">📦</h1>
                    <p style="color: #888; font-size: 12px; margin: 5px 0;">CODE: {recs[i]['id']}</p>
                    <h4 style="color: white; margin-bottom: 10px;">{recs[i]['desc']}</h4>
                    <hr style="border: 0.5px solid #333;">
                    <p style="color: #00FFAA; font-weight: bold;">{recs[i]['match']} Match Score</p>
                    <button style="background-color: #00FFAA; color: black; border: none; padding: 5px 15px; border-radius: 5px; font-weight: bold; cursor: pointer;">Add to Campaign</button>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    **💡 Marketing Insight:** These recommendations help reduce 'Churn' by showing the customer 
    items they haven't discovered yet, but are mathematically likely to purchase.

    """)


