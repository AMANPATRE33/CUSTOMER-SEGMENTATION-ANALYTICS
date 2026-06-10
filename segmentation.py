import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==========================================
# 1. PAGE CONFIGURATION & THEMING (UI/UX)
# ==========================================
st.set_page_config(
    page_title="Enterprise Customer Segmentation Engine",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Corporate Typography, Structural Spacing, and Component CSS Injections
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        /* Reset Base Typography */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            -webkit-font-smoothing: antialiased;
        }
        
        /* Main Workspace Header Styling (Theme Adaptive) */
        .platform-header {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.03em;
            line-height: 1.2;
            color: var(--text-color, #1E3A8A);
            margin-bottom: 0.5rem;
        }
        
        .platform-subtitle {
            font-size: 1.1rem;
            font-weight: 400;
            color: #6B7280;
            line-height: 1.6;
            margin-bottom: 2.5rem;
        }
        
        /* Metric Component Custom Styling */
        div[data-testid="stMetric"] {
            background-color: rgba(249, 250, 251, 0.05);
            border: 1px solid rgba(229, 231, 235, 0.3);
            padding: 1.25rem 1.5rem;
            border-radius: 8px;
            box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02);
        }
        
        div[data-testid="stMetricLabel"] {
            font-weight: 600;
            color: #9CA3AF;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        div[data-testid="stMetricValue"] {
            font-weight: 700;
            font-size: 1.75rem;
            letter-spacing: -0.02em;
        }
        
        /* Custom UI Action Block Button Styling */
        div.stButton > button {
            background: #1E40AF !important;
            color: #FFFFFF !important;
            font-weight: 600 !important;
            padding: 0.75rem 2rem !important;
            border-radius: 6px !important;
            border: none !important;
            transition: all 0.2s ease-in-out;
            letter-spacing: -0.01em;
        }
        
        div.stButton > button:hover {
            background: #1D4ED8 !important;
            transform: translateY(-1px);
        }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL & SCALER ARTIFACT LOADING
# ==========================================
@st.cache_resource
def load_assets():
    """Cache high-overhead serialization files to maximize platform performance"""
    model_path = "kmeans_model.pkl"
    scaler_path = "scaler.pkl"
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        st.error("Critical Error: Core pipeline assets 'kmeans_model.pkl' and 'scaler.pkl' are missing from root directory.")
        st.stop()

kmeans, scaler = load_assets()

# ==========================================
# 3. GRANULAR CORPORATE INSIGHT DICTIONARY
# ==========================================
persona_map = {
    0: {
        "name": "Older Affluent In-Store VIPs",
        "badge": "Tier 1 Gold Portfolio Asset",
        "desc": "Wealthy retirees characterized by an average age of 69.39, substantial capital capacity ($73,662.58 average gross income), and heavy reliance on brick-and-mortar touchpoints (8.50 average store transactions). Digital discovery footprint remains minimal.",
        "action": "Allocate budget allocation to physical marketing collateral. Deploy high-end direct mail portfolios, assign white-glove customer care channels, and orchestrate private regional store events to capture maximum customer retention values.",
        "color": "#B45309"
    },
    1: {
        "name": "At-Risk Low-Value Digital Browsers",
        "badge": "High-Probability Churn Warning Class",
        "desc": "Highly active web traffic consumers (7.15 average monthly visits) demonstrating near-total capital conversion bottlenecks. Segment displays depressed spending thresholds ($106.66) and an elevated risk recency value averaging 70.91 trailing days.",
        "action": "Strictly suppress expensive offline marketing pipelines. Direct efforts into programmatic web retargeting channels and automated email cart-recovery sequences designed to prompt a conversion before the retention window closes entirely.",
        "color": "#DC2626"
    },
    2: {
        "name": "Tech-Savvy Active Power Users",
        "badge": "Primary E-Commerce Engine Segment",
        "desc": "High-value digital buyers (average age 59.01) maintaining an optimal balance of robust cross-channel capital expenditure ($901.45) and the highest digital purchasing cadence in the framework (7.98 direct web checkouts). Recency holds steady at an active 45.79 days.",
        "action": "Provide digital convenience upgrades. Integrate app-exclusive reward structures, coordinate priority access to digital inventory drops, and run targeted promotions on top-performing product categories.",
        "color": "#059669"
    },
    3: {
        "name": "Aging Low-Engagement Shoppers",
        "badge": "Dormant Demographics Target",
        "desc": "Seniors (average age 67.70) within restricted income ranges ($42,253.58) showing systemic brand disengagement. Spending is limited to basic needs ($173.81) and recency metrics indicate an alarming 65.96 inactive days.",
        "action": "Leverage standard low-overhead baseline communication channels. Build re-engagement messaging around high-utility multi-pack product discounts and staple household necessities to extract low-risk remaining revenue.",
        "color": "#4B5563"
    },
    4: {
        "name": "Prime-Age Elite High-Spenders",
        "badge": "Corporate Crown Jewel Cluster",
        "desc": "Working professionals at peak earning age (46.18) commanding the absolute highest gross baseline revenue potential ($78,149.33 average income) and record-setting average basket values ($1,293.15 spending volume). Physical store execution values lead the matrix at 8.56 transactions.",
        "action": "Prioritize premium retention strategies. Implement complimentary express logistics structures, fast-track premium customer care options, and offer tailored cross-promotions for luxury item tiers to cross-sell premium lines.",
        "color": "#6D28D9"
    },
    5: {
        "name": "Highly Active Budget Deal-Hunters",
        "badge": "High Engagement Cohort",
        "desc": "The company's largest active volume cohort (461 active customers). Despite lower baseline spending capacity ($130.56), this cohort displays unparalleled brand affinity and freshness, leading the framework with an outstanding recency of 20.14 days.",
        "action": "Leverage their constant platform engagement. Roll out high-frequency digital flash sales, implement cart-value threshold volume incentives (e.g., 'Spend $60, Get $15 Off'), and use viral peer referral models to acquire similar customers.",
        "color": "#2563EB"
    }
}

# ==========================================
# 4. APPLICATION INTERFACE DESIGN (UX)
# ==========================================
# Left Sidebar Control Panel - Clean typography without icons
with st.sidebar:
    st.markdown("## Control Hub")
    st.caption("Engine Status: Active (K=6)")
    st.markdown("---")
    
    st.markdown("### Demographics")
    age = st.slider("Age", min_value=18, max_value=100, value=45, step=1)
    income = st.number_input("Annual Income ($)", min_value=0, max_value=300000, value=55000, step=1000)
    recency = st.slider("Recency (Days since last purchase)", min_value=0, max_value=365, value=30, step=1)
    
    st.markdown("---")
    
    st.markdown("### Transaction Channels")
    total_spending = st.number_input("Total Spending Volume ($)", min_value=0, max_value=10000, value=500, step=50)
    web_purchases = st.number_input("Web Purchases Count", min_value=0, max_value=50, value=4, step=1)
    store_purchases = st.number_input("Store Purchases Count", min_value=0, max_value=50, value=5, step=1)
    web_visits = st.slider("Monthly Web Visits Count", min_value=0, max_value=30, value=5, step=1)

# Central Main Panel Layout
st.markdown("<div class='platform-header'>CustomerEdge Intelligence Platform</div>", unsafe_allow_html=True)
st.markdown("<div class='platform-subtitle'>Configure target customer behavioral features on the sidebar panel to generate structural cohort classifications and recommended marketing strategies.</div>", unsafe_allow_html=True)

predict_btn = st.button("Run Real-Time Segment Allocation Pipeline", use_container_width=True)

# ==========================================
# 5. INFERENCE PIPELINE & SYSTEM REPORTING
# ==========================================
if predict_btn:
    # Build dataframe for structural model feature vector mapping
    raw_input_data = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "Total_Spending": total_spending,
        "NumWebPurchases": web_purchases,
        "NumStorePurchases": store_purchases,
        "NumWebVisitsMonth": web_visits,
        "Recency": recency
    }])
    
    # Run preprocessing scaling transformation
    scaled_input_data = scaler.transform(raw_input_data)
    
    # Model inference allocation execution
    predicted_cluster = kmeans.predict(scaled_input_data)[0]
    allocated_profile = persona_map[predicted_cluster]
    
    st.markdown("---")
    st.markdown("### Model Inference Output Summary")
    
    # Render Structural Analytics Layout Grid
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric(label="Assigned Cluster ID", value=f"Segment {predicted_cluster}")
    with m_col2:
        st.metric(label="Strategic Asset Class", value=allocated_profile["badge"])
    with m_col3:
        st.metric(label="Active Parameter Recency", value=f"{recency} Days")
        
    # Standard Executive HTML Reporting Card Block
    st.markdown(f"""
        <div style="background-color: rgba(249, 250, 251, 0.02); 
                    padding: 2.2rem; 
                    border-radius: 8px; 
                    border: 1px solid rgba(229, 231, 235, 0.15);
                    border-left: 5px solid {allocated_profile['color']}; 
                    margin-top: 1.8rem;">
            <h2 style="margin-top:0; color: {allocated_profile['color']}; font-weight:700; font-size:1.5rem; letter-spacing:-0.03em;">
                {allocated_profile['name']}
            </h2>
            <p style="font-size: 1.05rem; line-height: 1.65; margin-top:0.8rem; margin-bottom:1.5rem; opacity: 0.9;">
                <strong>Statistical Persona Profile:</strong> {allocated_profile['desc']}
            </p>
            <div style="background-color: rgba(255, 255, 255, 0.03); 
                        padding: 1.25rem 1.5rem; 
                        border-radius: 6px; 
                        border: 1px solid rgba(229, 231, 235, 0.1);
                        box-shadow: 0 1px 2px 0 rgba(0,0,0,0.01);">
                <p style="font-size: 1.05rem; margin-bottom: 0; line-height:1.6;">
                    <span style="color: {allocated_profile['color']}; font-weight: 700; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; display: block; margin-bottom: 0.25rem;">
                        Recommended Go-To-Market Operations Strategy
                    </span>
                    {allocated_profile['action']}
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("System status idle. Adjust operational customer metrics in the control hub panel and execute pipeline mapping above.")
