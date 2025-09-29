# Intelligent Analysis System

An AI-powered data analysis assistant that transforms CSV files into actionable insights, interactive visualizations, and comprehensive reports through an intuitive Streamlit interface.

## 🚀 Features

- **📊 Auto-Visualization**: Automatically generates relevant charts and graphs from your CSV data
- **🤖 NLP Querying**: Ask questions about your data in natural language and get instant answers
- **📝 Text Analysis**: Extract insights from textual data within your datasets
- **📄 One-Click Reports**: Generate comprehensive analysis reports with a single click
- **🎨 Interactive Interface**: User-friendly Streamlit web interface for seamless data exploration
- **🔧 Modular Architecture**: Clean, maintainable code structure for easy customization and extension

## 🛠️ Technologies Used

- **Frontend Framework**: Streamlit
- **AI/ML**: Natural Language Processing for intelligent querying
- **Data Processing**: Pandas for CSV manipulation
- **Visualization**: Matplotlib/Plotly/Seaborn (auto-generated charts)
- **Language**: Python 3.x

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.8 or higher
- pip (Python package manager)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/PabbisettySivaKumar/IntelligentAnalysisSystem.git
   cd IntelligentAnalysisSystem
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up API keys** (if applicable)
   - Create a `.env` file in the root directory
   - Add your API keys for AI services (e.g., OpenAI, Anthropic)
   ```
   API_KEY=your_api_key_here
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Upload your CSV file**
   - Navigate to the web interface (usually `http://localhost:8501`)
   - Upload your CSV file using the file uploader

3. **Explore your data**
   - View auto-generated visualizations
   - Ask natural language questions about your data
   - Analyze text columns for insights
   - Generate comprehensive reports

## 📊 Example Use Cases

- **Sales Analysis**: Upload sales data and ask "What were the top-performing products last quarter?"
- **Customer Insights**: Analyze customer feedback with text analysis features
- **Financial Reports**: Generate one-click financial summary reports
- **Trend Identification**: Automatically visualize trends and patterns in your data

## 🏗️ Project Structure

```
IntelligentAnalysisSystem/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not tracked)
│
├── modules/               # Modular components
    ├── data_processor.py  # CSV handling and preprocessing
    ├── visualizer.py      # Chart generation logic
    ├── nlp_engine.py      # Natural language query processing
    ├── text_analyzer.py   # Text analysis features
    └── report_generator.py # Report creation module

```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Pabbisetti Siva Kumar**
- GitHub: [@PabbisettySivaKumar](https://github.com/PabbisettySivaKumar)

## 🙏 Acknowledgments

- Streamlit team for the excellent framework
- Open-source community for various libraries and tools
- Contributors who help improve this project

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact me directly at [@itsPSK95](https://x.com/itsPSK95)

---

**Made with ❤️ and AI**
