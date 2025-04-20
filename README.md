# ESG Navigator

![ESG Navigator](generated-icon.png)

## Overview

ESG Navigator is a data-driven dashboard for analyzing Environmental, Social, and Governance (ESG) metrics across companies and industry sectors. The application provides standardized scores with statistical confidence intervals to help stakeholders understand ESG performance in a transparent and rigorous manner.

## Features

- **Sector-specific benchmarking**: Compare companies against relevant industry standards rather than global averages
- **Statistical rigor**: Calculate confidence intervals and perform significance testing to assess the reliability of ESG scores
- **Transparent methodology**: Clearly explain all data processing steps and calculation methods
- **Anomaly detection**: Identify metrics that deviate significantly from expected ranges
- **Comprehensive visualization**: Multiple visualization types for exploring ESG performance across domains

## Technical Architecture

The application is built with a modular architecture consisting of:

1. **Data Layer**: Loads and processes raw ESG metrics with support for multiple data sources
2. **Processing Layer**: Handles data cleaning, imputation, outlier detection, and standardization
3. **Analysis Layer**: Calculates scores with statistical methods including bootstrap resampling
4. **Visualization Layer**: Creates interactive visualizations for ESG performance analysis

### File Hierarchy

```
├── app.py                 # Main Streamlit application
├── data_processor.py      # Data cleaning and standardization
├── score_calculator.py    # ESG score calculation with statistical analysis
├── utils.py               # Utility functions for data operations
├── visualization.py       # Data visualization components
├── .streamlit/            # Streamlit configuration
│   └── config.toml        # Server and theme configuration
├── data/                  # Sample datasets
│   ├── environmental_data.csv    # Environmental metrics
│   ├── social_data.csv           # Social metrics
│   ├── governance_data.csv       # Governance metrics
│   ├── sector_benchmarks.csv     # Industry benchmarks
│   └── metric_descriptions.csv   # Metric explanations and importance
├── CONTRIBUTING.md        # Contribution guidelines
├── DOCUMENTATION.md       # Detailed technical documentation
├── LICENSE                # MIT License
├── README.md              # Project overview (this file)
└── pyproject.toml         # Python project dependencies
```

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, streamlit, plotly, scikit-learn, scipy

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Adityab270900/ESG_Navigator.git
   cd ESG_Navigator
   ```

2. Install dependencies:
   ```
   # Using pip
   pip install numpy pandas plotly scikit-learn scipy streamlit
   
   # Or if you have Poetry installed
   poetry install  # This will use the pyproject.toml file
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Select Industry Sector and Year**: Choose from available sectors and reporting years
2. **Configure Analysis**: Adjust domain weights and analysis settings
3. **Process Data**: Calculate ESG scores and generate visualizations
4. **Explore Results**: Examine scores, benchmarks, and statistical analyses

### Methodology

ESG Navigator uses a transparent, multi-step approach:

1. **Data Processing**:
   - Missing value imputation with KNN
   - Outlier treatment with Winsorization
   - Standardization based on data characteristics and sector benchmarks

2. **Score Calculation**:
   - Metric weighting based on importance (high=3x, medium=2x, low=1x)
   - Domain weighting (customizable)
   - Bootstrap resampling for uncertainty quantification

3. **Statistical Analysis**:
   - Anomaly detection (|z| > 3)
   - Significance testing against sector benchmarks
   - Score classification (Poor to Excellent)

## Data Format

The application expects data in CSV format with the following structure:

### Environmental, Social, and Governance Data
- **company**: Company name
- **sector**: Industry sector
- **year**: Reporting year
- **metrics...**: Various ESG metrics as columns

### Sector Benchmarks
- **sector**: Industry sector
- **metric**: Metric name
- **domain**: ESG domain (environmental, social, governance)
- **low_benchmark**: Lower threshold for typical values
- **average_benchmark**: Industry average
- **high_benchmark**: Upper threshold for high performers
- **unit**: Unit of measurement

### Metric Descriptions
- **domain**: ESG domain (environmental, social, governance)
- **metric**: Metric name
- **unit**: Unit of measurement
- **direction**: Whether higher or lower values are better
- **importance**: Importance level (high, medium, low)
- **description**: Explanation of the metric

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Developed as part of a data-driven approach to ESG analysis
- Inspired by the need for more transparent and standardized ESG scoring methodologies