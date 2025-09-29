import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd

def create_visualization(df, chart_type, **kwargs):
    """
    Create visualization based on chart type and parameters
    """
    plt.figure(figsize=(10, 6))
    
    try:
        if chart_type == "Histogram":
            column = kwargs.get('column')
            if not column:
                st.error("No column selected for histogram")
                return
            plt.hist(df[column].dropna(), bins=20, edgecolor='black')
            plt.title(f"Distribution of {column}")
            plt.xlabel(column)
            plt.ylabel("Frequency")
            
        elif chart_type == "Scatter Plot":
            col1 = kwargs.get('col1')
            col2 = kwargs.get('col2')
            if not col1 or not col2:
                st.error("Please select both X and Y axes for scatter plot")
                return
            if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
                st.error("Both columns must be numeric for scatter plot")
                return
            plt.scatter(df[col1], df[col2], alpha=0.6)
            plt.title(f"{col1} vs {col2}")
            plt.xlabel(col1)
            plt.ylabel(col2)
            # Add regression line
            sns.regplot(x=col1, y=col2, data=df, scatter=False, color='red')
            
        elif chart_type == "Bar Chart":
            column = kwargs.get('column')
            if not column:
                st.error("No column selected for bar chart")
                return
            df[column].value_counts().plot(kind='bar')
            plt.title(f"Count of {column}")
            plt.xticks(rotation=45)
            
        elif chart_type == "Heatmap":
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.empty:
                st.error("No numeric columns found for heatmap")
                return
            sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
            plt.title("Correlation Matrix")
        
        elif chart_type == "Box Plot":
            column = kwargs.get('column')
            if not column:
                st.error("No column selected for box plot")
                return
            sns.boxplot(x=df[column])
            plt.title(f"Box Plot of {column}")
            
        elif chart_type == "Line Chart":
            date_col = kwargs.get('date_col')
            value_col = kwargs.get('value_col')
            if not date_col or not value_col:
                st.error("Date and value columns required for line chart")
                return
            
            # Create a copy to avoid modifying original df
            temp_df = df.copy()
            temp_df[date_col] = pd.to_datetime(temp_df[date_col])
            temp_df.set_index(date_col, inplace=True)
            temp_df[value_col].plot()
            plt.title(f"Trend of {value_col} over time")
            plt.ylabel(value_col)
            
        else:
            st.error(f"Unsupported chart type: {chart_type}")
            return
            
        plt.tight_layout()
        st.pyplot(plt)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")