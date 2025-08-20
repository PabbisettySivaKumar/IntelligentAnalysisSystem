import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st

def load_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.json'):
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format")
            return None
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None
    

def preprocess_data(df):
    """Comprehensive data preprocessing pipeline"""
    # Record original shape
    original_shape = df.shape
    
    # Handle missing values
    df, missing_info = handle_missing_values(df)
    
    # Convert datetime columns
    df = convert_datetime_columns(df)
    
    # Remove duplicates
    df = remove_duplicates(df)
    
    # Handle outliers
    df = handle_outliers(df)
    
    # Encode categorical variables
    df = encode_categorical_variables(df)
    
    # Normalize numerical features
    df = normalize_numerical_features(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Return processed data and transformation summary
    transformation_report = generate_transformation_report(
        original_shape, df.shape, missing_info
    )
    
    return df, transformation_report

def handle_missing_values(df):
    """Handle missing values with multiple strategies"""
    missing_info = {}
    
    for col in df.columns:
        # Calculate missing percentage
        missing_percent = df[col].isnull().mean() * 100
        
        if missing_percent > 0:
            missing_info[col] = {
                'missing_percent': missing_percent,
                'original_dtype': str(df[col].dtype)
            }
            
            # Handle based on data type and missing percentage
            if missing_percent > 30:
                # Drop columns with more than 30% missing values
                df.drop(col, axis=1, inplace=True)
                missing_info[col]['action'] = 'dropped'
            elif df[col].dtype in ['int64', 'float64']:
                # Numerical columns: median imputation
                df[col].fillna(df[col].median(), inplace=True)
                missing_info[col]['action'] = 'median_imputation'
            elif df[col].dtype == 'object':
                # Categorical columns: mode imputation
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                missing_info[col]['action'] = f'mode_imputation ({mode_val})'
            elif 'date' in col.lower() or 'time' in col.lower():
                # Date columns: forward fill
                df[col].fillna(method='ffill', inplace=True)
                missing_info[col]['action'] = 'forward_fill'
            else:
                # Default: drop rows with missing values
                df.dropna(subset=[col], inplace=True)
                missing_info[col]['action'] = 'row_dropped'
    
    return df, missing_info

def convert_datetime_columns(df):
    """Automatically detect and convert datetime columns"""
    date_cols = []
    
    for col in df.select_dtypes(include=['object']).columns:
        try:
            # Attempt conversion to datetime
            df[col] = pd.to_datetime(df[col], errors='raise')
            date_cols.append(col)
            
            # Extract datetime features
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            
        except (ValueError, TypeError):
            continue
    
    return df

def remove_duplicates(df):
    """Remove duplicate rows and report actions"""
    original_rows = len(df)
    df.drop_duplicates(inplace=True)
    removed_rows = original_rows - len(df)
    
    if removed_rows > 0:
        st.info(f"Removed {removed_rows} duplicate rows")
    
    return df

def handle_outliers(df, threshold=3):
    """Detect and handle outliers using Z-score method"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[col]))
        
        # Identify outliers
        outliers = df[z_scores > threshold]
        
        if not outliers.empty:
            # Cap outliers to 95th percentile
            upper_limit = df[col].quantile(0.95)
            lower_limit = df[col].quantile(0.05)
            
            df[col] = np.where(
                df[col] > upper_limit, upper_limit,
                np.where(df[col] < lower_limit, lower_limit, df[col])
            )
    
    return df

def encode_categorical_variables(df):
    """Encode categorical variables using appropriate methods"""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in cat_cols:
        # Use label encoding for binary categorical variables
        if df[col].nunique() == 2:
            df[col] = pd.factorize(df[col])[0]
        
        # Use one-hot encoding for low-cardinality variables
        elif df[col].nunique() <= 10:
            df = pd.concat([df, pd.get_dummies(df[col], prefix=col)], axis=1)
            df.drop(col, axis=1, inplace=True)
        
        # Use frequency encoding for high-cardinality variables
        else:
            freq_map = df[col].value_counts(normalize=True).to_dict()
            df[col] = df[col].map(freq_map)
    
    return df

def normalize_numerical_features(df):
    """Normalize numerical features using min-max scaling"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numeric_cols:
        # Skip if constant value
        if df[col].nunique() == 1:
            continue
            
        # Apply min-max scaling
        min_val = df[col].min()
        max_val = df[col].max()
        
        if min_val != max_val:  # Avoid division by zero
            df[col] = (df[col] - min_val) / (max_val - min_val)
    
    return df

def feature_engineering(df):
    """Create new features through transformations"""
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Create interaction features
    if len(numeric_cols) >= 2:
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                col1 = numeric_cols[i]
                col2 = numeric_cols[j]
                
                # Create multiplicative interaction
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                
                # Create ratio feature
                if df[col2].min() > 0:  # Avoid division by zero
                    df[f'{col1}_div_{col2}'] = df[col1] / df[col2]
    
    # Create polynomial features
    for col in numeric_cols:
        if df[col].nunique() > 10:  # Only for continuous variables
            df[f'{col}_squared'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
    
    return df

def generate_transformation_report(original_shape, new_shape, missing_info):
    """Generate a report of data transformations"""
    report = "## Data Transformation Report\n\n"
    report += f"- **Original dimensions**: {original_shape[0]} rows × {original_shape[1]} columns\n"
    report += f"- **New dimensions**: {new_shape[0]} rows × {new_shape[1]} columns\n"
    
    if missing_info:
        report += "\n### Missing Value Handling\n"
        report += "| Column | Missing % | Action Taken |\n"
        report += "|--------|-----------|--------------|\n"
        for col, info in missing_info.items():
            report += f"| {col} | {info['missing_percent']:.2f}% | {info['action']} |\n"
    
    return report

def get_data_summary(df):
    """Generate comprehensive data summary"""
    summary = {}
    
    # Basic info
    summary['shape'] = df.shape
    summary['dtypes'] = df.dtypes.value_counts().to_dict()
    
    # Missing values
    missing = df.isnull().sum()
    summary['missing_values'] = missing[missing > 0].to_dict()
    
    # Descriptive statistics
    summary['descriptive_stats'] = df.describe(include='all').to_dict()
    
    # Unique values
    summary['unique_values'] = {col: df[col].nunique() for col in df.columns}
    
    # Skewness and kurtosis for numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    summary['skewness'] = {col: df[col].skew() for col in numeric_cols}
    summary['kurtosis'] = {col: df[col].kurtosis() for col in numeric_cols}
    
    return summary

def detect_data_quality_issues(df):
    """Detect common data quality issues"""
    issues = []
    
    # Check for duplicate rows
    if df.duplicated().sum() > 0:
        issues.append(f"Duplicate rows detected: {df.duplicated().sum()}")
    
    # Check for constant columns
    constant_cols = [col for col in df.columns if df[col].nunique() == 1]
    if constant_cols:
        issues.append(f"Constant columns detected: {', '.join(constant_cols)}")
    
    # Check for high cardinality categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    high_card_cols = [col for col in cat_cols if df[col].nunique() > 50]
    if high_card_cols:
        issues.append(f"High-cardinality categorical features: {', '.join(high_card_cols)}")
    
    # Check for skewed distributions
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    skewed_cols = [col for col in numeric_cols if abs(df[col].skew()) > 2]
    if skewed_cols:
        issues.append(f"Highly skewed features: {', '.join(skewed_cols)}")
    
    # Check for potential ID columns
    id_like_cols = [
        col for col in df.columns 
        if 'id' in col.lower() and df[col].nunique() == len(df)
    ]
    if id_like_cols:
        issues.append(f"Potential ID columns: {', '.join(id_like_cols)}")
    
    return issues

def feature_correlation_analysis(df):
    """Analyze feature correlations"""
    results = {}
    
    # Numerical correlations
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    if not numeric_df.empty:
        corr_matrix = numeric_df.corr()
        results['numerical_correlation'] = corr_matrix
        
        # Find highly correlated features
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr = [
            (col1, col2, corr_matrix.loc[col1, col2])
            for col1 in upper_triangle.columns
            for col2 in upper_triangle.index
            if abs(upper_triangle.loc[col1, col2]) > 0.8
        ]
        results['highly_correlated_features'] = high_corr
    
    return results