"""
PulseCheck Credit Risk Scoring System - Dual View
Customer-facing loan application with bank-side explainable AI dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import joblib
import shap
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import hashlib
import time

# Page configuration
st.set_page_config(
    page_title="PulseCheck Credit Risk Scoring",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional banking interface
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hide Streamlit branding for professional look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Headers */
    h1 {
        color: #1e3a5f;
        font-family: 'Segoe UI', sans-serif;
        font-weight: 300;
    }
    
    h2 {
        color: #2c5282;
        font-size: 24px;
        margin-top: 2rem;
    }
    
    /* Customer view styling */
    .customer-header {
        background: #ffffff;
        padding: 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .bank-header {
        background: #ffffff;
        padding: 1rem 0;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1e3a5f;
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border-radius: 5px;
        transition: all 0.3s;
        width: 100%;
        font-size: 16px;
    }
    
    .stButton > button:hover {
        background-color: #2c5282;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Form inputs */
    .stSelectbox > div > div {
        background-color: #f7fafc;
    }
    
    .stNumberInput > div > div > input {
        background-color: #f7fafc;
    }
    
    /* Decision boxes */
    .approved-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 2rem;
        border-radius: 5px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .denied-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 2rem;
        border-radius: 5px;
        text-align: center;
        margin: 2rem 0;
    }
    
    .review-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 2rem;
        border-radius: 5px;
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: #1e3a5f;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #edf2f7;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    
    /* Professional tables */
    .dataframe {
        font-size: 14px;
    }
    
    .dataframe thead th {
        background-color: #1e3a5f !important;
        color: white !important;
        font-weight: 500;
    }
    
    .dataframe tbody tr:hover {
        background-color: #f7fafc !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = 'customer'
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Database initialization
@st.cache_resource(ttl=1)  # Short TTL to force refresh during development
def init_database():
    """Initialize DuckDB database with applications table"""
    conn = duckdb.connect('data/applications.duckdb', read_only=False)
    
    # Drop existing table to recreate with correct schema
    conn.execute("DROP TABLE IF EXISTS applications")
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            application_id VARCHAR PRIMARY KEY,
            application_date TIMESTAMP,
            
            -- Personal Information
            person_age INTEGER,
            person_income INTEGER,
            person_home_ownership VARCHAR,
            person_emp_length DOUBLE,
            
            -- Loan Information
            loan_intent VARCHAR,
            loan_grade VARCHAR,
            loan_amnt INTEGER,
            loan_int_rate DOUBLE,
            loan_percent_income DOUBLE,
            
            -- Credit History
            cb_person_default_on_file VARCHAR,
            cb_person_cred_hist_length INTEGER,
            
            -- Model Results
            risk_score DOUBLE,
            risk_category VARCHAR,
            decision VARCHAR,
            decision_date TIMESTAMP,
            
            -- Additional Metadata
            model_version VARCHAR DEFAULT 'XGBoost_v1',
            officer_override VARCHAR DEFAULT NULL,
            officer_notes TEXT DEFAULT NULL
        )
    """)
    
    conn.commit()
    return conn

# Load models
@st.cache_resource
def load_models():
    """Load the trained XGBoost model and scaler"""
    try:
        model = joblib.load('xgboost_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

# Generate application ID
def generate_application_id():
    """Generate unique application ID"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    import random
    random_num = random.randint(1000, 9999)
    return f"APP{timestamp}{random_num}"

# Calculate internal fields
def calculate_internal_fields(income, loan_amount, risk_score):
    """Calculate loan grade and interest rate based on risk"""
    # Loan grade based on risk score
    if risk_score < 0.15:
        loan_grade = 'A'
        base_rate = 5.5
    elif risk_score < 0.25:
        loan_grade = 'B'
        base_rate = 7.0
    elif risk_score < 0.40:
        loan_grade = 'C'
        base_rate = 9.5
    elif risk_score < 0.55:
        loan_grade = 'D'
        base_rate = 12.0
    elif risk_score < 0.70:
        loan_grade = 'E'
        base_rate = 14.5
    elif risk_score < 0.85:
        loan_grade = 'F'
        base_rate = 17.0
    else:
        loan_grade = 'G'
        base_rate = 20.0
    
    # Adjust rate based on loan to income ratio
    loan_to_income = loan_amount / income if income > 0 else 1
    if loan_to_income > 0.5:
        base_rate += 1.5
    elif loan_to_income > 0.3:
        base_rate += 0.5
    
    return loan_grade, round(base_rate, 1)

# Prepare features for model
def prepare_features(data):
    """Prepare input features for model prediction"""
    feature_dict = {}
    
    # Add numerical features
    feature_dict['person_age'] = data['person_age']
    feature_dict['person_income'] = data['person_income']
    feature_dict['person_emp_length'] = data['person_emp_length']
    feature_dict['loan_amnt'] = data['loan_amnt']
    feature_dict['loan_int_rate'] = data['loan_int_rate']
    feature_dict['loan_percent_income'] = data['loan_percent_income']
    feature_dict['cb_person_cred_hist_length'] = data['cb_person_cred_hist_length']
    
    # One-hot encoded features (drop_first=True logic)
    for val in ['OTHER', 'OWN', 'RENT']:
        feature_dict[f'person_home_ownership_{val}'] = 1 if data['person_home_ownership'] == val else 0
    
    for val in ['EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']:
        feature_dict[f'loan_intent_{val}'] = 1 if data['loan_intent'] == val else 0
    
    for val in ['B', 'C', 'D', 'E', 'F', 'G']:
        feature_dict[f'loan_grade_{val}'] = 1 if data['loan_grade'] == val else 0
    
    feature_dict['cb_person_default_on_file_Y'] = 1 if data['cb_person_default_on_file'] == 'Y' else 0
    
    # Create DataFrame with correct column order
    feature_order = [
        'person_age', 'person_income', 'person_emp_length', 'loan_amnt',
        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
        'person_home_ownership_OTHER', 'person_home_ownership_OWN', 'person_home_ownership_RENT',
        'loan_intent_EDUCATION', 'loan_intent_HOMEIMPROVEMENT', 'loan_intent_MEDICAL',
        'loan_intent_PERSONAL', 'loan_intent_VENTURE',
        'loan_grade_B', 'loan_grade_C', 'loan_grade_D', 'loan_grade_E', 'loan_grade_F', 'loan_grade_G',
        'cb_person_default_on_file_Y'
    ]
    
    X = pd.DataFrame([feature_dict])[feature_order]
    return X

# Get risk decision
def get_risk_decision(risk_score):
    """Get decision based on risk score"""
    if risk_score < 0.35:
        return "Approved", "approved"
    elif risk_score < 0.65:
        return "Under Review", "review"
    else:
        return "Denied", "denied"

# Get customer-friendly explanations
def get_customer_explanation(risk_score, age, income, loan_amount, employment_length, credit_history_length, has_default):
    """Generate customer-friendly explanation"""
    explanations = []
    
    # Income to loan ratio
    loan_to_income = loan_amount / income if income > 0 else 1
    if loan_to_income > 0.5:
        explanations.append("• Your loan amount is high relative to your income")
    elif loan_to_income < 0.2:
        explanations.append("✓ Your loan amount is reasonable for your income level")
    
    # Employment stability
    if employment_length < 1:
        explanations.append("• Limited employment history affects your application")
    elif employment_length > 5:
        explanations.append("✓ Your stable employment history strengthens your application")
    
    # Credit history
    if credit_history_length < 3:
        explanations.append("• Building more credit history would help")
    elif credit_history_length > 7:
        explanations.append("✓ Your established credit history is a positive factor")
    
    # Previous defaults
    if has_default == 'Y':
        explanations.append("• Previous credit issues are affecting your score")
    
    # Age factor
    if age < 25:
        explanations.append("• Younger applicants typically have limited credit history")
    elif age > 40:
        explanations.append("✓ Your age group has strong repayment statistics")
    
    return explanations[:3]  # Return top 3 explanations

# Initialize
if not st.session_state.db_initialized:
    st.session_state.conn = init_database()
    st.session_state.db_initialized = True
    
if st.session_state.model is None:
    model, scaler = load_models()
    st.session_state.model = model
    st.session_state.scaler = scaler

# View mode toggle will be placed at the bottom of the page

# Bank login view
if st.session_state.view_mode == 'bank_login' and not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🏦 Bank Officer Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.form_submit_button("Login"):
                # Simple demo authentication
                if username == "admin" and password == "admin":
                    st.session_state.logged_in = True
                    st.session_state.view_mode = 'bank'
                    st.success("Login successful!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid credentials. Use admin/admin for demo.")

# Customer View
elif st.session_state.view_mode == 'customer':
    # Header
    st.title("💳 Loan Application Portal")
    st.markdown("Check your eligibility in under 60 seconds")
    st.markdown("---")
    
    # Application form
    st.markdown("## Loan Application Form")
    
    with st.form("customer_application"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Personal Information")
            age = st.slider("Your Age", 18, 70, 30)
            income = st.slider("Annual Income ($)", 15000, 200000, 50000, step=5000)
            home_ownership = st.selectbox(
                "Living Situation",
                ["RENT", "OWN", "MORTGAGE", "OTHER"],
                format_func=lambda x: {
                    "RENT": "Renting",
                    "OWN": "Own Home (No Mortgage)",
                    "MORTGAGE": "Own Home (With Mortgage)",
                    "OTHER": "Other Arrangement"
                }[x]
            )
            employment_length = st.slider("Years at Current Job", 0.0, 20.0, 3.0, step=0.5)
        
        with col2:
            st.markdown("### Loan Details")
            loan_amount = st.slider("Loan Amount Needed ($)", 1000, 50000, 10000, step=500)
            loan_purpose = st.selectbox(
                "What Will You Use the Loan For?",
                ["PERSONAL", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "EDUCATION", "MEDICAL", "VENTURE"],
                format_func=lambda x: {
                    "PERSONAL": "Personal Use",
                    "DEBTCONSOLIDATION": "Consolidate Debt",
                    "HOMEIMPROVEMENT": "Home Improvement",
                    "EDUCATION": "Education",
                    "MEDICAL": "Medical Expenses",
                    "VENTURE": "Business Investment"
                }[x]
            )
            
            st.markdown("### Credit Information")
            credit_issues = st.radio(
                "Have you had credit issues in the past?",
                ["No", "Yes"],
                horizontal=True
            )
            credit_history = st.select_slider(
                "How long have you had credit?",
                options=["Less than 2 years", "2-5 years", "5-10 years", "More than 10 years"],
                value="5-10 years"
            )
        
        # Convert credit history to numeric
        credit_history_numeric = {
            "Less than 2 years": 1,
            "2-5 years": 3,
            "5-10 years": 7,
            "More than 10 years": 12
        }[credit_history]
        
        # Show estimated monthly payment
        estimated_payment = loan_amount / 36  # Simplified 3-year term
        st.info(f"💡 **Estimated Monthly Payment:** ${estimated_payment:,.0f} (36-month term)")
        
        submitted = st.form_submit_button("🚀 Check My Eligibility", use_container_width=True)
    
    # Process form submission outside the form context
    if submitted:
        with st.spinner("Analyzing your application..."):
            time.sleep(1.5)  # Simulate processing
            
            # Calculate loan percent income
            loan_percent_income = loan_amount / income if income > 0 else 0
                
            # First prediction to get risk score
            temp_data = {
                'person_age': age,
                'person_income': income,
                'person_home_ownership': home_ownership,
                'person_emp_length': employment_length,
                'loan_intent': loan_purpose,
                'loan_grade': 'C',  # Temporary
                'loan_amnt': loan_amount,
                'loan_int_rate': 10.0,  # Temporary
                'loan_percent_income': loan_percent_income,
                'cb_person_default_on_file': 'Y' if credit_issues == "Yes" else 'N',
                'cb_person_cred_hist_length': credit_history_numeric
            }
            
            # Prepare features and get initial risk score
            X = prepare_features(temp_data)
            X_scaled = st.session_state.scaler.transform(X)
            risk_score = st.session_state.model.predict_proba(X_scaled)[0, 1]
            
            # Calculate loan grade and interest rate based on risk
            loan_grade, interest_rate = calculate_internal_fields(income, loan_amount, risk_score)
            
            # Update data with calculated fields
            application_data = temp_data.copy()
            application_data['loan_grade'] = loan_grade
            application_data['loan_int_rate'] = interest_rate
            
            # Get final prediction with correct loan grade and interest rate
            X_final = prepare_features(application_data)
            X_final_scaled = st.session_state.scaler.transform(X_final)
            final_risk_score = st.session_state.model.predict_proba(X_final_scaled)[0, 1]
                
            # Get decision
            decision, decision_type = get_risk_decision(final_risk_score)
                
            # Generate application ID
            app_id = generate_application_id()
            
            # Save to database
            conn = st.session_state.conn
            conn.execute("""
                INSERT INTO applications (
                    application_id, application_date,
                    person_age, person_income, person_home_ownership, person_emp_length,
                    loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income,
                    cb_person_default_on_file, cb_person_cred_hist_length,
                    risk_score, risk_category, decision, decision_date, model_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                app_id, datetime.now(),
                int(age), int(income), home_ownership, float(employment_length),
                loan_purpose, loan_grade, int(loan_amount), float(interest_rate), float(loan_percent_income),
                'Y' if credit_issues == "Yes" else 'N', int(credit_history_numeric),
                float(final_risk_score),  # Convert numpy float32 to Python float
                "Low Risk" if final_risk_score < 0.35 else "Medium Risk" if final_risk_score < 0.65 else "High Risk",
                decision, datetime.now(), 'XGBoost_v1'
            ])
            conn.commit()
        
        # Display results
        
        if decision_type == "approved":
            st.markdown(f"""
                <div class="approved-box">
                    <h1 style="color: #0f5132; margin: 0;">🎉 Congratulations!</h1>
                    <h2 style="color: #0f5132;">You're Pre-Approved!</h2>
                    <p style="font-size: 20px; color: #0f5132;">
                        Loan Amount: ${loan_amount:,}<br>
                        Interest Rate: {interest_rate}% APR<br>
                        Application ID: {app_id}
                    </p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.button("📞 Speak to a Loan Officer", use_container_width=True)
            with col2:
                st.button("📧 Email Me the Details", use_container_width=True)
                
        elif decision_type == "review":
            st.markdown(f"""
                <div class="review-box">
                    <h1 style="color: #664d03; margin: 0;">📋 Additional Review Needed</h1>
                    <h2 style="color: #664d03;">We Need More Information</h2>
                    <p style="font-size: 18px; color: #664d03;">
                        Your application is being reviewed by our team.<br>
                        We'll contact you within 24 hours.<br>
                        Application ID: {app_id}
                    </p>
            </div>
            """, unsafe_allow_html=True)
            
        else:  # denied
            st.markdown(f"""
                <div class="denied-box">
                    <h1 style="color: #842029; margin: 0;">Application Decision</h1>
                    <h2 style="color: #842029;">Unable to Approve at This Time</h2>
                    <p style="font-size: 18px; color: #842029;">
                        Application ID: {app_id}<br>
                        We encourage you to apply again in the future.
                    </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show customer-friendly explanations
        st.markdown("### Understanding Your Decision")
        explanations = get_customer_explanation(
                final_risk_score, age, income, loan_amount, 
                employment_length, credit_history_numeric,
                'Y' if credit_issues == "Yes" else 'N'
        )
        
        for explanation in explanations:
            st.markdown(explanation)
        
        if decision_type != "approved":
            st.markdown("### 💡 Tips to Improve Your Chances")
            st.markdown("""
                - Maintain stable employment for at least 2 years
                - Keep your loan request under 30% of your annual income
                - Build your credit history with regular payments
                - Consider a co-signer if you have limited credit history
            """)

# Bank View
elif st.session_state.view_mode == 'bank' and st.session_state.logged_in:
    st.markdown("### 🏦 Loan Officer Dashboard")
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Applications", "🔍 Application Detail", "📈 Portfolio Analytics", "⚙️ Model Performance"])
    
    with tab1:
        st.markdown("## Recent Applications")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            decision_filter = st.selectbox("Decision", ["All", "Approved", "Denied", "Under Review"])
        with col2:
            risk_filter = st.selectbox("Risk Level", ["All", "Low Risk", "Medium Risk", "High Risk"])
        with col3:
            date_filter = st.date_input("From Date", value=datetime.now().date())
        with col4:
            sort_by = st.selectbox("Sort By", ["Most Recent", "Highest Risk", "Lowest Risk", "Largest Loan"])
        
        # Get applications
        conn = st.session_state.conn
        query = "SELECT * FROM applications WHERE 1=1"
        
        if decision_filter != "All":
            query += f" AND decision = '{decision_filter}'"
        if risk_filter != "All":
            query += f" AND risk_category = '{risk_filter}'"
        
        if sort_by == "Most Recent":
            query += " ORDER BY application_date DESC"
        elif sort_by == "Highest Risk":
            query += " ORDER BY risk_score DESC"
        elif sort_by == "Lowest Risk":
            query += " ORDER BY risk_score ASC"
        else:
            query += " ORDER BY loan_amnt DESC"
        
        applications = conn.execute(query).fetchdf()
        
        if not applications.empty:
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Applications", len(applications))
            with col2:
                approval_rate = (applications['decision'] == 'Approved').mean()
                st.metric("Approval Rate", f"{approval_rate:.1%}")
            with col3:
                avg_risk = applications['risk_score'].mean()
                st.metric("Avg Risk Score", f"{avg_risk:.1%}")
            with col4:
                total_exposure = applications[applications['decision'] == 'Approved']['loan_amnt'].sum()
                st.metric("Total Exposure", f"${total_exposure:,.0f}")
            
            # Display applications table
            display_df = applications[['application_id', 'application_date', 'person_age', 
                                      'person_income', 'loan_amnt', 'risk_score', 
                                      'risk_category', 'decision']].copy()
            
            # Format for display
            display_df['application_date'] = pd.to_datetime(display_df['application_date']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['risk_score'] = (display_df['risk_score'] * 100).round(1).astype(str) + '%'
            display_df['person_income'] = '$' + display_df['person_income'].apply(lambda x: f"{x:,.0f}")
            display_df['loan_amnt'] = '$' + display_df['loan_amnt'].apply(lambda x: f"{x:,.0f}")
            
            # Rename columns for display
            display_df.columns = ['Application ID', 'Date', 'Age', 'Income', 'Loan Amount', 
                                 'Risk Score', 'Risk Category', 'Decision']
            
            st.dataframe(display_df, use_container_width=True, height=400)
        else:
            st.info("No applications found")
    
    with tab2:
        st.markdown("## Application Detail with Explainability")
        
        # Get applications for detail view
        conn = st.session_state.conn
        applications_detail = conn.execute("SELECT * FROM applications ORDER BY application_date DESC").fetchdf()
        
        # Application selector
        if not applications_detail.empty:
            selected_app = st.selectbox(
                "Select Application to Review",
                applications_detail['application_id'].tolist(),
                format_func=lambda x: f"{x} - {applications_detail[applications_detail['application_id']==x]['decision'].values[0]}"
            )
            
            if selected_app:
                app_data = applications_detail[applications_detail['application_id'] == selected_app].iloc[0]
                
                # Display application details
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Applicant Information")
                    st.write(f"**Age:** {app_data['person_age']}")
                    st.write(f"**Income:** ${app_data['person_income']:,.0f}")
                    st.write(f"**Employment Length:** {app_data['person_emp_length']} years")
                    st.write(f"**Home Ownership:** {app_data['person_home_ownership']}")
                    st.write(f"**Credit History:** {app_data['cb_person_cred_hist_length']} years")
                    st.write(f"**Previous Default:** {app_data['cb_person_default_on_file']}")
                
                with col2:
                    st.markdown("### Loan Details")
                    st.write(f"**Amount:** ${app_data['loan_amnt']:,.0f}")
                    st.write(f"**Purpose:** {app_data['loan_intent']}")
                    st.write(f"**Grade:** {app_data['loan_grade']}")
                    st.write(f"**Interest Rate:** {app_data['loan_int_rate']}%")
                    st.write(f"**Loan to Income:** {app_data['loan_percent_income']:.1%}")
                
                # Risk Assessment
                st.markdown("### Risk Assessment")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Risk Score", f"{app_data['risk_score']:.1%}")
                with col2:
                    st.metric("Risk Category", app_data['risk_category'])
                with col3:
                    st.metric("Decision", app_data['decision'])
                
                # SHAP Explanation
                st.markdown("### 🎯 Model Explainability (SHAP Analysis)")
                
                # Prepare the features for this application
                feature_data = {
                    'person_age': app_data['person_age'],
                    'person_income': app_data['person_income'],
                    'person_home_ownership': app_data['person_home_ownership'],
                    'person_emp_length': app_data['person_emp_length'],
                    'loan_intent': app_data['loan_intent'],
                    'loan_grade': app_data['loan_grade'],
                    'loan_amnt': app_data['loan_amnt'],
                    'loan_int_rate': app_data['loan_int_rate'],
                    'loan_percent_income': app_data['loan_percent_income'],
                    'cb_person_default_on_file': app_data['cb_person_default_on_file'],
                    'cb_person_cred_hist_length': app_data['cb_person_cred_hist_length']
                }
                
                X_explain = prepare_features(feature_data)
                X_explain_scaled = st.session_state.scaler.transform(X_explain)
                
                # Create SHAP explainer
                with st.spinner("Generating SHAP explanation..."):
                    explainer = shap.TreeExplainer(st.session_state.model)
                    shap_values = explainer.shap_values(X_explain_scaled)
                    
                    # If binary classification, take values for positive class
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    
                    # Create a more readable feature names mapping
                    feature_names_display = {
                        'person_age': 'Age',
                        'person_income': 'Annual Income',
                        'person_emp_length': 'Employment Length',
                        'loan_amnt': 'Loan Amount',
                        'loan_int_rate': 'Interest Rate',
                        'loan_percent_income': 'Loan to Income Ratio',
                        'cb_person_cred_hist_length': 'Credit History Length',
                        'person_home_ownership_OTHER': 'Home: Other',
                        'person_home_ownership_OWN': 'Home: Own',
                        'person_home_ownership_RENT': 'Home: Rent',
                        'loan_intent_EDUCATION': 'Purpose: Education',
                        'loan_intent_HOMEIMPROVEMENT': 'Purpose: Home Improvement',
                        'loan_intent_MEDICAL': 'Purpose: Medical',
                        'loan_intent_PERSONAL': 'Purpose: Personal',
                        'loan_intent_VENTURE': 'Purpose: Venture',
                        'loan_grade_B': 'Grade B',
                        'loan_grade_C': 'Grade C',
                        'loan_grade_D': 'Grade D',
                        'loan_grade_E': 'Grade E',
                        'loan_grade_F': 'Grade F',
                        'loan_grade_G': 'Grade G',
                        'cb_person_default_on_file_Y': 'Previous Default'
                    }
                    
                    # Get feature importance
                    feature_importance = pd.DataFrame({
                        'feature': [feature_names_display.get(f, f) for f in X_explain.columns],
                        'value': X_explain.values[0],
                        'shap_value': shap_values[0]
                    }).sort_values('shap_value', key=abs, ascending=False)
                    
                    # Display top factors
                    st.markdown("#### Top Factors Affecting This Decision")
                    
                    top_positive = feature_importance[feature_importance['shap_value'] > 0].head(3)
                    top_negative = feature_importance[feature_importance['shap_value'] < 0].head(3)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**🔴 Factors Increasing Risk:**")
                        for _, row in top_negative.iterrows():
                            st.write(f"• {row['feature']}: {row['value']:.2f} (SHAP: {row['shap_value']:.3f})")
                    
                    with col2:
                        st.markdown("**🟢 Factors Decreasing Risk:**")
                        for _, row in top_positive.iterrows():
                            st.write(f"• {row['feature']}: {row['value']:.2f} (SHAP: {row['shap_value']:.3f})")
                    
                    # Waterfall chart
                    st.markdown("#### SHAP Waterfall Chart")
                    
                    # Create waterfall data
                    base_value = explainer.expected_value
                    if isinstance(base_value, np.ndarray):
                        base_value = base_value[1]
                    
                    # Sort features by absolute SHAP value
                    sorted_idx = np.argsort(np.abs(shap_values[0]))[::-1][:10]  # Top 10 features
                    
                    waterfall_features = [feature_names_display.get(X_explain.columns[i], X_explain.columns[i]) 
                                        for i in sorted_idx]
                    waterfall_values = [shap_values[0][i] for i in sorted_idx]
                    
                    # Create waterfall chart
                    fig = go.Figure(go.Waterfall(
                        name="SHAP",
                        orientation="v",
                        measure=["relative"] * len(waterfall_features) + ["total"],
                        x=waterfall_features + ["Final Risk Score"],
                        y=waterfall_values + [sum(shap_values[0]) + base_value],
                        connector={"line": {"color": "rgb(63, 63, 63)"}},
                        decreasing={"marker": {"color": "#84fab0"}},
                        increasing={"marker": {"color": "#fccb90"}},
                        totals={"marker": {"color": "#667eea"}}
                    ))
                    
                    fig.update_layout(
                        title="Feature Contributions to Risk Score",
                        showlegend=False,
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Override section
                st.markdown("### 🔧 Manual Override")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    override_decision = st.selectbox(
                        "Override Decision",
                        ["No Override", "Approve", "Deny", "Request More Information"],
                        index=0
                    )
                    override_notes = st.text_area("Override Notes (Required for override)")
                
                with col2:
                    if st.button("Apply Override", disabled=(override_decision == "No Override")):
                        if override_notes:
                            # Update database with override
                            conn.execute("""
                                UPDATE applications 
                                SET officer_override = ?, officer_notes = ?
                                WHERE application_id = ?
                            """, [override_decision, override_notes, selected_app])
                            conn.commit()
                            st.success("Override applied successfully!")
                        else:
                            st.error("Please provide override notes")
        else:
            st.info("No applications available for review. Submit an application from the Customer Portal first.")
    
    with tab3:
        st.markdown("## Portfolio Analytics")
        
        # Get applications for analytics
        conn = st.session_state.conn
        applications_analytics = conn.execute("SELECT * FROM applications").fetchdf()
        
        if not applications_analytics.empty:
            # Risk distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    applications_analytics, 
                    x='risk_score', 
                    nbins=20,
                    title="Risk Score Distribution",
                    labels={'risk_score': 'Risk Score', 'count': 'Number of Applications'}
                )
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                decision_counts = applications_analytics['decision'].value_counts()
                fig = px.pie(
                    values=decision_counts.values,
                    names=decision_counts.index,
                    title="Decision Distribution",
                    color_discrete_map={
                        'Approved': '#84fab0',
                        'Under Review': '#ffecd2',
                        'Denied': '#fccb90'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Loan purpose analysis
            purpose_risk = applications_analytics.groupby('loan_intent')['risk_score'].mean().sort_values()
            fig = px.bar(
                x=purpose_risk.values,
                y=purpose_risk.index,
                orientation='h',
                title="Average Risk Score by Loan Purpose",
                labels={'x': 'Average Risk Score', 'y': 'Loan Purpose'}
            )
            fig.update_traces(marker_color='#764ba2')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No applications data available for analytics. Submit applications from the Customer Portal to see analytics.")
    
    with tab4:
        st.markdown("## Model Performance Metrics")
        
        # Display model metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC Score", "0.9501")
        with col2:
            st.metric("F1 Score", "0.83")
        with col3:
            st.metric("Precision", "0.89")
        with col4:
            st.metric("Recall", "0.78")
        
        st.markdown("### Model Information")
        from datetime import datetime
        current_time = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        st.info(f"""
        **Model Type:** XGBoost (Gradient Boosting)  
        **Training Dataset:** 32,581 loan applications  
        **Features:** 22 features including personal, loan, and credit history information  
        **Last Updated:** {current_time}  
        **Version:** XGBoost_v1
        """)
        
        st.markdown("### Risk Thresholds")
        st.write("""
        | Risk Score | Category | Default Action |
        |------------|----------|----------------|
        | < 35% | Low Risk | Auto-Approve |
        | 35-65% | Medium Risk | Manual Review |
        | > 65% | High Risk | Auto-Deny |
        """)

# Footer with integrated view toggle
st.markdown("---")

# Footer with view toggle integrated
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        <div style='text-align: center; padding: 20px 0;'>
            <p style='color: #666; font-size: 14px; margin-bottom: 15px;'>
                PulseCheck Credit Risk Scoring System | Powered by XGBoost & SHAP
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )

with col3:
    st.markdown("<div style='padding-top: 10px;'>", unsafe_allow_html=True)
    if st.session_state.view_mode == 'customer':
        if st.button("🏦 Bank Portal", key="bank_toggle", help="Switch to bank officer view"):
            st.session_state.view_mode = 'bank_login'
            st.rerun()
    else:
        if st.button("🏠 Customer Portal", key="customer_toggle", help="Switch to customer view"):
            st.session_state.view_mode = 'customer'
            st.session_state.logged_in = False
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)