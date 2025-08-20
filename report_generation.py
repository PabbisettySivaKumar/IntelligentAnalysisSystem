import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
from ai_integration import get_hf_response, get_googleai_response
import base64

def generate_report(df, hf_model="gogle/flan-t5-base", google_model="gemini-pro"):
    # Create markdown report
    report = "# Automated Data Analysis Report\n\n"
    
    # Basic dataset info
    report += f"## Dataset Overview\n"
    report += f"- **Rows**: {len(df)}\n"
    report += f"- **Columns**: {len(df.columns)}\n"
    report += f"- **Missing Values**: {df.isnull().sum().sum()}\n\n"
    
    # Key statistics
    report += "## Key Statistics\n"
    report += df.describe().to_markdown() + "\n\n"
    
    # AI-generated insights
    report += "## AI-Generated Insights\n"
    
    # Hugging Face insights
    # report += "### Hugging Face Analysis\n"
    # hf_insights = get_hf_response("Provide 3 key insights about this dataset", df, hf_model)
    # report += f"{hf_insights}\n\n"
    
    # Google AI insights
    report += "### Google AI Analysis\n"
    google_insights = get_googleai_response("Provide 3 key insights about this dataset", df, google_model)
    report += f"{google_insights}\n\n"
    
    # Recommendations
    report += "## Recommendations\n"
    report += "### Next Steps\n"
    recs = get_googleai_response("Suggest 3 next steps for data analysis based on this dataset",df,google_model)
    report += f"{recs}\n\n"
    
    # Visualizations
    report += "## Key Visualizations\n"
    
    # Create distribution visualizations
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for i, col in enumerate(numeric_cols[:4]):
        plt.subplot(2, 2, i+1)
        df[col].hist()
        plt.title(f"Distribution of {col}")
    
    # Save visualization to buffer
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    report += f"![Distributions](data:image/png;base64,{img_str})\n\n"
    
    return report