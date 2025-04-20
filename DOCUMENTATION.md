# ESG Navigator Documentation

## Code Structure

The ESG Navigator codebase is organized into the following main modules:

### 1. `app.py`

The main Streamlit application that handles the user interface and orchestrates the data flow between other components.

**Key Functions:**
- Initialize session state for storing processed data
- Configure UI sidebar for data selection and parameter adjustment
- Process data when requested
- Display visualization and analysis results

### 2. `data_processor.py`

Handles data cleaning, imputation, and standardization for ESG metrics.

**Key Components:**
- `DataProcessor` class: Main class for data processing
  - `process_data()`: Process data for a specific ESG domain
  - Handles missing values with KNN imputation
  - Detects and handles outliers with Winsorization
  - Normalizes metrics based on their characteristics and sector benchmarks

### 3. `score_calculator.py`

Calculates ESG scores with uncertainty quantification and statistical analysis.

**Key Components:**
- `ScoreCalculator` class: Main class for score calculation
  - `calculate_scores()`: Calculate ESG scores from processed domain data
  - `_calculate_domain_score()`: Calculate weighted score for a specific domain
  - `_calculate_overall_score()`: Calculate overall ESG scores with bootstrap confidence intervals
  - `_detect_anomalies()`: Detect anomalies in the ESG metrics
  - `_perform_significance_tests()`: Conduct significance tests against industry benchmarks

### 4. `utils.py`

Utility functions for data loading, manipulation, and helper operations.

**Key Functions:**
- `load_sample_data()`: Load sample data from the data directory
- `load_sector_benchmarks()`: Load sector benchmarks from CSV
- `load_metric_descriptions()`: Load metric descriptions from CSV
- `get_default_weights()`: Generate default weights for metrics within a domain
- `normalize_metric_data()`: Normalize metric data using various methods
- Various data retrieval and manipulation utilities

### 5. `visualization.py`

Creates visualizations for ESG data and analysis results.

**Key Functions:**
- `plot_esg_scores()`: Plot ESG scores with confidence intervals
- `plot_domain_scores()`: Plot domain-level scores as a radar chart or bar chart
- `plot_metric_distribution()`: Plot the distribution of a specific metric
- `plot_anomalies()`: Plot anomalies for a specific company
- `plot_significance_tests()`: Plot significance test results
- `plot_sector_benchmarks()`: Plot sector benchmarks for metrics

## Data Files

ESG Navigator uses the following data files stored in the `data/` directory:

### 1. Domain-specific ESG Data
- `environmental_data.csv`: Environmental metrics by company, sector, and year
- `social_data.csv`: Social metrics by company, sector, and year
- `governance_data.csv`: Governance metrics by company, sector, and year

### 2. Reference Data
- `sector_benchmarks.csv`: Industry-specific benchmarks for ESG metrics
- `metric_descriptions.csv`: Descriptions and metadata for ESG metrics

## Core Concepts

### 1. ESG Domains

The application focuses on three ESG domains:

- **Environmental**: Metrics related to a company's impact on the natural environment
- **Social**: Metrics related to a company's relationships with employees, customers, suppliers, and communities
- **Governance**: Metrics related to a company's leadership, controls, and shareholder rights

### 2. Standardization and Normalization

ESG Navigator standardizes metrics to make them comparable:

- **Z-score standardization**: For normally distributed metrics
- **Min-Max scaling**: For bounded metrics like percentages
- **Robust scaling**: For skewed metrics
- **Benchmark-based scaling**: For metrics with sector-specific benchmarks

### 3. Statistical Methods

The application uses several statistical methods:

- **Bootstrap resampling**: For calculating confidence intervals
- **Anomaly detection**: For identifying unusual values (|z| > 3)
- **Significance testing**: For comparing company performance to benchmarks

### 4. Visualization Types

ESG Navigator provides multiple visualization types:

- **Bar charts with error bars**: For overall ESG scores with confidence intervals
- **Radar charts**: For comparing performance across domains
- **Histograms**: For showing metric distributions
- **Range charts**: For comparing metrics to sector benchmarks
- **Lollipop charts**: For displaying anomalies and significance test results

## Adding New Features

### 1. Adding New Metrics

To add new ESG metrics:

1. Add the metrics to the respective domain data files
2. Update `metric_descriptions.csv` with information about the new metrics
3. Add sector benchmarks for the new metrics in `sector_benchmarks.csv`

### 2. Adding New Visualizations

To add new visualization types:

1. Create a new function in `visualization.py`
2. Add the visualization to the appropriate tab in `app.py`

### 3. Adding New Analysis Methods

To add new analysis methods:

1. Implement the method in `score_calculator.py` or create a new module
2. Add the results to the `calculate_scores()` return dictionary
3. Update the UI in `app.py` to display the new analysis

## Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Ensure data files exist in the `data/` directory
   - Check that data files follow the expected format

2. **Visualization Errors**:
   - Verify that the data contains the expected columns
   - Check for missing values in critical fields

3. **Calculation Errors**:
   - Ensure all required metrics are present in the data
   - Verify that benchmark data exists for the selected sector

## Future Improvements

Potential areas for enhancement:

1. **Data Sources**:
   - Add support for real-time data from ESG data providers
   - Implement connectors for financial databases

2. **Advanced Analytics**:
   - Add predictive modeling for ESG performance trends
   - Implement peer group analysis

3. **User Experience**:
   - Add user authentication and saved analysis profiles
   - Implement report generation and export functionality

4. **Technical Improvements**:
   - Optimize data processing for larger datasets
   - Add caching for improved performance