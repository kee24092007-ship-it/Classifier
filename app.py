import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ==========================================
# PREMIUM LIGHT UI/UX DESIGN THEME CONFIG
# ==========================================
st.set_page_config(
    page_title="NexusOS Workspace // Intelligence Center", 
    page_icon="", 
    layout="wide"
)

# Custom Global Stylesheet Injector (Clean Premium Light Theme with Dark Text)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Light Background Reset */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Sidebar Light Theme Customization */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0 !important;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h2 {
        color: #0F172A !important;
    }

    /* Telemetry KPI Badges (Light Variant) */
    .kpi-container {
        background: #FFFFFF;
        padding: 24px;
        border-radius: 18px;
        border: 1px solid #E2E8F0;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .kpi-container:hover {
        transform: translateY(-4px);
        border-color: #4F46E5;
        box-shadow: 0 10px 20px -5px rgba(79, 70, 229, 0.1);
    }
    .kpi-val {
        font-size: 34px;
        font-weight: 800;
        color: #4F46E5;
        letter-spacing: -0.5px;
    }
    .kpi-lbl {
        font-size: 11px;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: 1px;
        margin-top: 6px;
    }

    /* Input Element Light Mode Override */
    div[data-baseweb="input"], div[data-baseweb="number-input"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 14px !important;
        color: #0F172A !important;
    }
    
    /* Text overrides to ensure pure dark visibility */
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #0F172A !important;
    }

    /* Interactive Button Style */
    button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%) !important;
        color: #FFFFFF !important;
        border-radius: 14px !important;
        padding: 12px 28px !important;
        border: none !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.2) !important;
        width: 100%;
    }
    button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 24px rgba(79, 70, 229, 0.3) !important;
    }
    button p {
        color: #FFFFFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# AUTOMATED BACKEND ML PIPELINE
# ==========================================
@st.cache_data
def calibrate_system():
    try:
        df = pd.read_csv('Dataset for Data Analytics - Sheet1.csv')
        df = df.dropna(subset=['Quantity', 'UnitPrice', 'ItemsInCart', 'OrderStatus'])
        
        X = df[['Quantity', 'UnitPrice', 'ItemsInCart']]
        y = df['OrderStatus']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
        model.fit(X_train, y_train)
        return model, df
    except FileNotFoundError:
        return None, None

model, df = calibrate_system()

if df is None:
    st.error("System Ingestion Failure: 'Dataset for Data Analytics - Sheet1.csv' core asset route cannot be found.")
    st.stop()


# ==========================================
# SIDEBAR NAVIGATION MATRIX (2 TABS ONLY)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='font-weight:800; color:#0F172A; letter-spacing:-1px; margin-bottom:0;'>NEXUS<span style='color:#4F46E5;'>OS</span></h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#64748B; font-size:11px; text-transform:uppercase; font-weight:700; letter-spacing:1.5px; margin-bottom:35px;'>Control Workspace</p>", unsafe_allow_html=True)
    
    workspace_tab = st.radio(
        "WORKSPACE SELECTION",
        ["Telemetry Analytics", "AI Predictive Engine"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown(
        f"""
        <div style="background: #F1F5F9; padding: 16px; border-radius: 14px; border: 1px solid #E2E8F0;">
            <p style="font-size:10px; color:#4F46E5; margin:0; font-weight:700; letter-spacing:0.5px;">SYSTEM STATUS</p>
            <p style="font-size:13px; color:#16A34A; margin:4px 0 0 0; font-weight:700;">Core Network Stable</p>
            <p style="font-size:11px; color:#64748B; margin:8px 0 0 0;">Dataset Batches: <b>{len(df):,} Rows</b></p>
        </div>
        """,
        unsafe_allow_html=True
    )


# ==========================================
# MODULE 1: TELEMETRY ANALYTICS (DASHBOARD)
# ==========================================
if workspace_tab == "Telemetry Analytics":
    st.markdown("<h2 style='color: #0F172A; font-weight: 800; margin-bottom: 2px; letter-spacing:-0.5px;'>Data Analytics Telemetry</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size:14px; margin-bottom:25px;'>Comprehensive database tracking and live order funnel profiles.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="kpi-container"><div class="kpi-val">{len(df):,}</div><div class="kpi-lbl">Total Logged Volume</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="kpi-container"><div class="kpi-val">{int(df["Quantity"].sum()):,}</div><div class="kpi-lbl">Gross Dispatched Units</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="kpi-container"><div class="kpi-val">${df["UnitPrice"].mean():.2f}</div><div class="kpi-lbl">Mean Item Price Point</div></div>', unsafe_allow_html=True)
        
    st.write("###")
    
    graph_col, table_col = st.columns([11, 12])
    
    with graph_col:
        st.markdown("<h4 style='color: #0F172A; font-weight:700; margin-bottom:15px; font-size:15px;'>Workflow Distribution Breakdown</h4>", unsafe_allow_html=True)
        
        status_data = df['OrderStatus'].value_counts().reset_index()
        status_data.columns = ['Workflow Status', 'Record Counts']
        
        fig = px.bar(
            status_data, x='Workflow Status', y='Record Counts',
            color='Record Counts', color_continuous_scale='Purples',
            template='plotly_white'
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_showscale=False,
            font_color="#0F172A"
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    with table_col:
        st.markdown("<h4 style='color: #0F172A; font-weight:700; margin-bottom:15px; font-size:15px;'>Dataset Explorer (Live Overview)</h4>", unsafe_allow_html=True)
        
        display_features = ['OrderID', 'Product', 'Quantity', 'UnitPrice', 'OrderStatus']
        st.dataframe(df[display_features].head(8), use_container_width=True, hide_index=True)


# ==========================================
# MODULE 2: AI PREDICTIVE ENGINE (PREDICTOR)
# ==========================================
elif workspace_tab == "AI Predictive Engine":
    st.markdown("<h2 style='color: #0F172A; font-weight: 800; margin-bottom: 2px; letter-spacing:-0.5px;'>Intelligent Prediction Engine</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: #64748B; font-size:14px; margin-bottom:25px;'>Run pipeline constraints through calibrated random forest structures to parse success probability.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        quantity = st.number_input("Purchase Quantity Volume", min_value=1, max_value=500, value=2)
    with col2:
        unit_price = st.number_input("Item Point Evaluation Value ($)", min_value=0.1, max_value=5000.0, value=45.0)
    with col3:
        items_in_cart = st.number_input("Active Line Items in Cart", min_value=1, max_value=200, value=3)
    
    st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
    
    if st.button("EXECUTE LIVE PIPELINE EVALUATION"):
        input_vector = pd.DataFrame([[quantity, unit_price, items_in_cart]], columns=['Quantity', 'UnitPrice', 'ItemsInCart'])
        res_prediction = model.predict(input_vector)
        
        accent_color = "#16A34A" if res_prediction[0] in ['Shipped', 'Delivered'] else "#DC2626"
        alert_bg = "rgba(22, 163, 74, 0.08)" if res_prediction[0] in ['Shipped', 'Delivered'] else "rgba(220, 38, 38, 0.08)"
        
        st.markdown(
            f"""
            <div style="background: {alert_bg}; padding: 25px; border-radius: 18px; border: 1px solid {accent_color}; border-left: 6px solid {accent_color}; margin-top: 15px;">
                <span style="color: {accent_color}; font-size: 11px; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; display:block; margin-bottom:4px;">Machine Learning Resolution Output</span>
                <h3 style="color: #0F172A; margin: 0; font-size: 22px; font-weight: 800;">Predicted Status Target &rarr; <span style="color:{accent_color}; text-decoration: underline;">{res_prediction[0]}</span></h3>
            </div>
            """, 
            unsafe_allow_html=True
        )