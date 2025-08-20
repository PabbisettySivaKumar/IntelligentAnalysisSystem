import streamlit as st
from dotenv import load_dotenv
from data_processor import load_data, preprocess_data, detect_data_quality_issues, get_data_summary, feature_correlation_analysis
from visualisation import create_visualization
from nlp_processing import analyze_text
from ai_integration import get_hf_response, get_googleai_response
from report_generation import generate_report
from advanced_stats import run_bayesian_estimation, run_time_series_forecast, run_causal_inference
import pandas as pd
import os
import numpy as np

# Load environment variables FIRST
load_dotenv()

# Page setup
st.set_page_config(page_title="Smart Analyzer", page_icon="📊", layout="wide")
st.title("📊 Smart Analyzer - AI Data Assistant")

# Sidebar - Data upload
with st.sidebar:
    st.header("Data Configuration")
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv", "xlsx", "json"])
    
    # Debug: Show API key status
    st.divider()
    st.header("API Key Status")
    st.write(f"HF_API_KEY loaded: {os.getenv('HF_API_KEY') is not None}")
    st.write(f"GOOGLE_API_KEY loaded: {os.getenv('GOOGLE_API_KEY') is not None}")

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed' not in st.session_state:
    st.session_state.processed = False

# Data processing
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state.df = df
        st.success("Data loaded successfully!")
        
        if st.button("Preprocess Data", key="preprocess_btn"):
            st.session_state.df = preprocess_data(df)
            st.session_state.processed = True
            st.rerun()

# Main interface tabs
if st.session_state.df is not None:
    tab1, tab2, tab3, tab4,tab5 = st.tabs([
        "📁 Data Explorer", 
        "💬 AI Assistant", 
        "📈 Visualization", 
        "📊 Advanced Statistics",
        "📝 Report Generator"
    ])
    
    # TAB 1: Data Explorer
    with tab1:
        st.subheader("Dataset Preview")
        
        # Show first 10 rows
        st.dataframe(st.session_state.df.head(10))
        
        # Show last 10 rows in expander
        with st.expander("View Last 10 Rows"):
            st.dataframe(st.session_state.df.tail(10))
        
        # Basic dataset info
        st.subheader("Dataset Information")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(st.session_state.df))
        col2.metric("Columns", len(st.session_state.df.columns))
        missing_values = st.session_state.df.isnull().sum().sum()
        col3.metric("Missing Values", missing_values)
        
        # Data Summary
        st.subheader("Data Summary")
        st.dataframe(st.session_state.df.describe())
        
        # Data Types
        st.subheader("Data Types")
        dtype_counts = st.session_state.df.dtypes.value_counts().reset_index()
        dtype_counts.columns = ['Data Type', 'Count']
        st.dataframe(dtype_counts)
        
        # Data Quality Check
        st.subheader("Data Quality Check")
        if st.button("Run Data Quality Analysis", key="data_quality_btn"):
            with st.spinner("Analyzing data quality..."):
                issues = detect_data_quality_issues(st.session_state.df)
                
                if issues:
                    st.warning("Data Quality Issues Detected:")
                    for issue in issues:
                        st.error(f"- {issue}")
                else:
                    st.success("No major data quality issues detected")
                
                # Show detailed data summary
                st.subheader("Comprehensive Data Summary")
                data_summary = get_data_summary(st.session_state.df)
                
                # Display basic summary
                st.write("**Shape:**", data_summary['shape'])
                st.write("**Data Types:**", data_summary['dtypes'])
                
                # Show missing values
                if data_summary['missing_values']:
                    st.write("**Missing Values:**")
                    missing_df = pd.DataFrame.from_dict(data_summary['missing_values'], 
                                                      orient='index', 
                                                      columns=['Missing Count'])
                    st.dataframe(missing_df)
                else:
                    st.info("No missing values found")
                
                # Show unique values
                st.write("**Unique Values per Column:**")
                unique_df = pd.DataFrame.from_dict(data_summary['unique_values'], 
                                                 orient='index', 
                                                 columns=['Unique Values'])
                st.dataframe(unique_df)
                
                # Show skewness and kurtosis
                if data_summary['skewness']:
                    st.subheader("Distribution Analysis")
                    dist_df = pd.DataFrame({
                        'Skewness': data_summary['skewness'],
                        'Kurtosis': data_summary['kurtosis']
                    })
                    st.dataframe(dist_df)
                
                # Show correlation analysis
                st.subheader("Feature Correlation Analysis")
                correlation_results = feature_correlation_analysis(st.session_state.df)
                
                if 'numerical_correlation' in correlation_results:
                    st.write("**Correlation Matrix:**")
                    st.dataframe(correlation_results['numerical_correlation'])
                
                if 'highly_correlated_features' in correlation_results:
                    if correlation_results['highly_correlated_features']:
                        st.write("**Highly Correlated Features (|r| > 0.8):**")
                        for col1, col2, corr in correlation_results['highly_correlated_features']:
                            st.write(f"- {col1} and {col2}: {corr:.2f}")
                    else:
                        st.info("No highly correlated features found")
        
        # Show transformation report if available
        if 'transformation_report' in st.session_state and st.session_state.transformation_report:
            st.subheader("Data Transformation Report")
            st.markdown(st.session_state.transformation_report)
    
    # TAB 2: AI Assistant
    with tab2:
        st.subheader("Intelligent Query Processing")
        query = st.text_area("Ask about your data:", height=150, key="query_input")
        
        col1, col2 = st.columns(2)
        with col1:
            use_hf = st.checkbox("Use Hugging Face", value=True, key="use_hf_check")
        with col2:
            use_google = st.checkbox("Use Google AI", value=True, key="use_google_check")
        
        # Model selection inside the tab
        with st.expander("Model Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                hf_model = st.selectbox(
                    "Hugging Face Model",
                    [
                        "facebook/bart-large-cnn",   # ✅ summarization model
                        "google/flan-t5-base",       # ✅ text-to-text model
                        "deepset/roberta-base-squad2", # ✅ Q&A model
                        "gpt2",                      # ✅ text generation
                        "distilgpt2"
                    ],
                    index=0,
                    key="assistant_hf_model"
                )
            with col2:
                google_model = st.selectbox(
                    "Google AI Model",
                    [
                        "gemini-1.5-pro",       # General purpose, best quality
                        "gemini-1.5-pro-002",   # Explicit version
                        "gemini-1.5-flash",     # Faster & cheaper
                        "gemini-1.5-flash-002", # Latest Flash version
                        "gemini-2.0-flash"      # Newest generation, fast + multimodal
                    ],
                    index=0,
                    key="assistant_google_model"
                )
        
        if st.button("Analyze", key="analyze_btn") and query:
            with st.spinner("Processing..."):
                # NLP analysis
                nlp_results = analyze_text(query)
                
                # AI responses
                responses = []
                
                if use_hf:
                    try:
                        hf_response = get_hf_response(
                            query, 
                            st.session_state.df, 
                            model_name=hf_model
                        )
                        responses.append(("Hugging Face", hf_response))
                    except Exception as e:
                        st.error(f"Hugging Face Error: {str(e)}")
                
                if use_google:
                    try:
                        google_response = get_googleai_response(
                            query,
                            st.session_state.df,
                            model_name=google_model
                        )
                        responses.append(("Google AI", google_response))
                    except Exception as e:
                        st.error(f"Google AI Error: {str(e)}")
                
                # Display results
                for provider, response in responses:
                    with st.expander(f"{provider} Analysis", expanded=True):
                        st.write(response)
    
    # TAB 3: Visualization
    with tab3:
        st.subheader("Interactive Visualization")
        
        chart_type = st.selectbox(
            "Select Chart Type", 
            ["Histogram", "Scatter Plot", "Bar Chart", "Heatmap", "Box Plot", "Line Chart"],
            key="chart_type_select"
        )
        
        # Dynamic controls based on chart type
        if chart_type in ["Histogram", "Box Plot"]:
            column = st.selectbox("Select Column", st.session_state.df.columns, key="hist_column")
        elif chart_type == "Scatter Plot":
            col1 = st.selectbox("X-Axis", st.session_state.df.columns, key="scatter_x")
            col2 = st.selectbox("Y-Axis", st.session_state.df.columns, key="scatter_y")
        elif chart_type == "Bar Chart":
            column = st.selectbox(
                "Select Categorical Column", 
                st.session_state.df.select_dtypes(include=['object']).columns,
                key="bar_column"
            )
        elif chart_type == "Line Chart":
            date_cols = st.session_state.df.select_dtypes(include=['datetime']).columns
            date_col = st.selectbox("Date Column", date_cols, key="line_date") if not date_cols.empty else None
            value_col = st.selectbox("Value Column", st.session_state.df.select_dtypes(include=['int64', 'float64']).columns, key="line_value")
        
        if st.button("Generate Visualization", key="viz_btn"):
            # Prepare arguments for create_visualization
            kwargs = {
                "df": st.session_state.df,
                "chart_type": chart_type
            }
            
            if chart_type in ["Histogram", "Box Plot"]:
                kwargs["column"] = column
            elif chart_type == "Scatter Plot":
                kwargs["col1"] = col1
                kwargs["col2"] = col2
            elif chart_type == "Bar Chart":
                kwargs["column"] = column
            elif chart_type == "Line Chart" and date_col:
                kwargs["date_col"] = date_col
                kwargs["value_col"] = value_col
            
            create_visualization(**kwargs)
    
    
    # TAB 4: Advanced Statistics
    with tab4:
        st.subheader("Advanced Statistical Analysis")

        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Bayesian Estimation", "Time Series Forecasting", "Causal Inference"],
            key="adv_analysis_type"
        )

        if analysis_type == "Bayesian Estimation":
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns
            col = st.selectbox("Select Numeric Column", numeric_cols, key="bayesian_col")
            if st.button("Run Bayesian Estimation", key="run_bayes_btn"):
                result = run_bayesian_estimation(st.session_state.df[col].dropna())
                st.text(result)

        elif analysis_type == "Time Series Forecasting":
            date_cols = st.session_state.df.select_dtypes(include=["datetime", "object"]).columns
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns

            date_col = st.selectbox("Date Column", date_cols, key="ts_date")
            value_col = st.selectbox("Value Column", numeric_cols, key="ts_value")

            if st.button("Run Forecast", key="run_forecast_btn"):
                try:
                    fig, forecast = run_time_series_forecast(
                        st.session_state.df, date_col, value_col
                    )
                    st.pyplot(fig)
                    st.write("Forecasted Values:")
                    st.dataframe(forecast)
                except Exception as e:
                    st.error(f"Forecast failed: {e}")

        elif analysis_type == "Causal Inference":
            numeric_cols = st.session_state.df.select_dtypes(include=np.number).columns
            treatment_col = st.selectbox("Treatment Column", numeric_cols, key="treatment_col")
            outcome_col = st.selectbox("Outcome Column", numeric_cols, key="outcome_col")

            if st.button("Run Causal Inference", key="run_causal_btn"):
                result = run_causal_inference(st.session_state.df, treatment_col, outcome_col)
                st.text(result)

    # TAB 5: Report Generator
    with tab5:
        st.subheader("Automated Report")
        st.info("Select AI models to generate a comprehensive analysis report")
        
        # Model selection
    
        report_google_model = st.selectbox(
                "Google AI Model",
                [
                    "gemini-1.5-pro",
                    "gemini-1.5-pro-002",
                    "gemini-1.5-flash",
                    "gemini-2.0-flash"
                ],
                index=1,
                key="report_google_model"
            )
        
        # Report button inside the tab
        if st.button("Generate Comprehensive Report", key="report_btn"):
            with st.spinner("Generating insights..."):
                try:
                    report = generate_report(
                        st.session_state.df,
                        google_model=report_google_model
                    )
                    st.markdown(report)
                    st.download_button(
                        "Download Report", 
                        report, 
                        file_name="data_analysis_report.md",
                        key="download_report"
                    )
                except Exception as e:
                    st.error(f"Report generation failed: {str(e)}")

else:
    st.info("Please upload a dataset to begin analysis")

# Add debug info at the bottom
st.sidebar.divider()
st.sidebar.caption("Made by Siva Kumar with help from Dotkonnet.")