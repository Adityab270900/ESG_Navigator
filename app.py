import streamlit as st
import pandas as pd
import numpy as np
from data_processor import DataProcessor
from score_calculator import ScoreCalculator
from utils import (
    load_sample_data, 
    load_metric_descriptions,
    load_sector_benchmarks,
    get_default_weights,
    get_available_sectors,
    get_available_years,
    get_companies_for_sector_year,
    get_domain_metrics
)
from visualization import (
    plot_esg_scores,
    plot_domain_scores,
    plot_metric_distribution,
    plot_anomalies,
    plot_significance_tests,
    plot_sector_benchmarks
)

st.set_page_config(
    page_title="ESG Navigator",
    page_icon="🌍",
    layout="wide"
)

st.title("ESG Navigator")
st.subheader("Environmental, Social, and Governance Metrics Analysis")

# Initialize session state for storing processed data
if 'env_data' not in st.session_state:
    st.session_state.env_data = None
if 'social_data' not in st.session_state:
    st.session_state.social_data = None
if 'gov_data' not in st.session_state:
    st.session_state.gov_data = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'companies' not in st.session_state:
    st.session_state.companies = []
if 'scores' not in st.session_state:
    st.session_state.scores = None
if 'selected_sector' not in st.session_state:
    st.session_state.selected_sector = None
if 'selected_year' not in st.session_state:
    st.session_state.selected_year = None
if 'metric_descriptions' not in st.session_state:
    # Load metric descriptions
    try:
        st.session_state.metric_descriptions = {
            'environmental': load_metric_descriptions(domain='environmental'),
            'social': load_metric_descriptions(domain='social'),
            'governance': load_metric_descriptions(domain='governance')
        }
    except:
        st.session_state.metric_descriptions = {}

# Sidebar for data selection and configuration
with st.sidebar:
    st.header("Analysis Selection")
    
    # Sample data selection
    available_sectors = get_available_sectors()
    selected_sector = st.selectbox(
        "Select Industry Sector",
        options=available_sectors,
        index=0
    )
    
    available_years = get_available_years()
    selected_year = st.selectbox(
        "Select Reporting Year",
        options=available_years,
        index=0
    )
    
    # Update session state if selection changed
    if (selected_sector != st.session_state.selected_sector or 
        selected_year != st.session_state.selected_year):
        st.session_state.selected_sector = selected_sector
        st.session_state.selected_year = selected_year
        
        # Clear previous results
        st.session_state.results = None
        st.session_state.scores = None
    
    st.header("Configuration")
    
    # Weight configuration
    st.subheader("Domain Weights")
    env_weight = st.slider("Environmental Weight", 0.0, 1.0, 0.4, 0.05)
    social_weight = st.slider("Social Weight", 0.0, 1.0, 0.35, 0.05)
    gov_weight = st.slider("Governance Weight", 0.0, 1.0, 0.25, 0.05)
    
    # Normalize weights to sum to 1.0
    total = env_weight + social_weight + gov_weight
    if total > 0:
        env_weight = env_weight / total
        social_weight = social_weight / total
        gov_weight = gov_weight / total
    
    domain_weights = {
        'environmental': env_weight,
        'social': social_weight,
        'governance': gov_weight
    }
    
    st.info(f"Normalized weights: Env={env_weight:.2f}, Social={social_weight:.2f}, Gov={gov_weight:.2f}")
    
    # Analysis settings
    st.subheader("Analysis Settings")
    use_sector_benchmarks = st.checkbox("Use Sector-Specific Benchmarks", value=True)
    normalization_method = st.selectbox(
        "Normalization Method (if not using benchmarks)",
        options=["Z-Score", "Min-Max", "Robust"],
        index=0,
        disabled=use_sector_benchmarks
    )
    
    # Bootstrap configuration
    st.subheader("Uncertainty Quantification")
    bootstrap_iterations = st.slider("Bootstrap Iterations", 100, 5000, 1000, 100)
    confidence_level = st.slider("Confidence Level (%)", 80, 99, 95, 1)
    
    # Process data button
    process_button = st.button("Process Data", type="primary")

# Main content area
if process_button or st.session_state.results is None:
    # Process data
    with st.spinner(f"Processing data for {selected_sector} sector, {selected_year}..."):
        try:
            # Load sample data for the selected sector and year
            env_data = load_sample_data('environmental', sector=selected_sector, year=int(selected_year))
            social_data = load_sample_data('social', sector=selected_sector, year=int(selected_year))
            gov_data = load_sample_data('governance', sector=selected_sector, year=int(selected_year))
            
            # Store data in session state
            st.session_state.env_data = env_data
            st.session_state.social_data = social_data
            st.session_state.gov_data = gov_data
            
            # Get metric descriptions for weighting
            metric_descriptions = st.session_state.metric_descriptions
            
            # Get the list of metrics for each domain
            env_metrics = [col for col in env_data.columns if col not in ['company', 'sector', 'year']]
            social_metrics = [col for col in social_data.columns if col not in ['company', 'sector', 'year']]
            gov_metrics = [col for col in gov_data.columns if col not in ['company', 'sector', 'year']]
            
            # Get default metric weights with importance weighting
            env_metric_weights = get_default_weights(
                'environmental', env_metrics, 
                metric_descriptions.get('environmental', None)
            )
            social_metric_weights = get_default_weights(
                'social', social_metrics, 
                metric_descriptions.get('social', None)
            )
            gov_metric_weights = get_default_weights(
                'governance', gov_metrics, 
                metric_descriptions.get('governance', None)
            )
            
            # Create processor and calculator
            processor = DataProcessor()
            calculator = ScoreCalculator(
                domain_weights=domain_weights,
                metric_weights={
                    'environmental': env_metric_weights,
                    'social': social_metric_weights,
                    'governance': gov_metric_weights
                },
                bootstrap_iterations=bootstrap_iterations,
                confidence_level=confidence_level/100
            )
            
            # Process data
            cleaned_env = processor.process_data(
                env_data, domain='environmental', 
                use_benchmarks=use_sector_benchmarks
            )
            cleaned_social = processor.process_data(
                social_data, domain='social', 
                use_benchmarks=use_sector_benchmarks
            )
            cleaned_gov = processor.process_data(
                gov_data, domain='governance', 
                use_benchmarks=use_sector_benchmarks
            )
            
            # Extract company list (assuming all datasets have the same companies)
            companies = cleaned_env['company'].unique().tolist()
            st.session_state.companies = companies
            
            # Calculate scores
            results = calculator.calculate_scores(cleaned_env, cleaned_social, cleaned_gov)
            
            # Store results in session state
            st.session_state.results = results
            st.session_state.scores = results['overall_scores']
            
            st.success("Data processing complete!")
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            st.exception(e)

# Display results if available
if st.session_state.results is not None:
    # Display sector and year information
    st.header(f"ESG Analysis: {st.session_state.selected_sector} Sector ({st.session_state.selected_year})")
    
    # Company selection
    company_filter = st.selectbox(
        "Select Company to View", 
        options=["All Companies"] + st.session_state.companies
    )
    
    # Help button for methodology explanation
    with st.expander("**📊 Methodology & Metrics Explanation**"):
        st.markdown("""
        ### ESG Navigator Methodology
        
        This dashboard provides a transparent analysis of Environmental, Social, and Governance (ESG) performance using the following approaches:
        
        #### Data Processing
        1. **Missing Value Imputation**: K-nearest neighbors (KNN) imputation with weighted distance calculation
        2. **Outlier Treatment**: Winsorization at 1st/99th percentiles to cap extreme values
        3. **Standardization Methods**:
           - *Z-score*: For normally distributed metrics (transforms to mean=0, std=1)
           - *Min-Max scaling*: For bounded ratios like percentages (transforms to range between -1 and 1)
           - *Robust scaling*: For skewed metrics, using median and interquartile range
           - *Benchmark scaling*: For metrics with sector-specific benchmarks, calculated relative to low/average/high industry standards
        
        #### Score Calculation
        1. **Metric Weights**: Weighted by importance (high=3x, medium=2x, low=1x)
        2. **Domain Weights**: Configurable through sliders (default: Environmental=40%, Social=35%, Governance=25%)
        3. **Overall Score**: Weighted sum of domain scores
        4. **Confidence Intervals**: Bootstrap resampling with 1000 iterations at 95% confidence level
        
        #### Statistical Analysis
        1. **Anomaly Detection**: Flags metrics with absolute z-scores > 3 (outside 99.7% of normal distribution)
        2. **Significance Testing**: T-tests for each metric against sector benchmarks
        3. **Rating Scale**: 
           - **-∞ to -0.75**: Poor
           - **-0.75 to -0.25**: Below Average
           - **-0.25 to 0.25**: Average
           - **0.25 to 0.75**: Above Average
           - **0.75 to ∞**: Excellent
        """)
        
        # Display current metrics
        st.subheader("Metrics by Domain")
        metrics_tab1, metrics_tab2, metrics_tab3 = st.tabs(["Environmental", "Social", "Governance"])
        
        with metrics_tab1:
            env_metrics = st.session_state.metric_descriptions.get('environmental', pd.DataFrame())
            if not env_metrics.empty:
                st.dataframe(
                    env_metrics[['metric', 'unit', 'direction', 'importance', 'description']],
                    hide_index=True,
                    column_config={
                        "metric": "Metric",
                        "unit": "Unit",
                        "direction": st.column_config.TextColumn(
                            "Direction",
                            help="Whether higher or lower values are better for this metric"
                        ),
                        "importance": st.column_config.TextColumn(
                            "Importance",
                            help="Weighting applied to this metric (high=3x, medium=2x, low=1x)"
                        ),
                        "description": "Description"
                    }
                )
            
        with metrics_tab2:
            social_metrics = st.session_state.metric_descriptions.get('social', pd.DataFrame())
            if not social_metrics.empty:
                st.dataframe(
                    social_metrics[['metric', 'unit', 'direction', 'importance', 'description']],
                    hide_index=True,
                    column_config={
                        "metric": "Metric",
                        "unit": "Unit",
                        "direction": st.column_config.TextColumn(
                            "Direction",
                            help="Whether higher or lower values are better for this metric"
                        ),
                        "importance": st.column_config.TextColumn(
                            "Importance",
                            help="Weighting applied to this metric (high=3x, medium=2x, low=1x)"
                        ),
                        "description": "Description"
                    }
                )
            
        with metrics_tab3:
            gov_metrics = st.session_state.metric_descriptions.get('governance', pd.DataFrame())
            if not gov_metrics.empty:
                st.dataframe(
                    gov_metrics[['metric', 'unit', 'direction', 'importance', 'description']],
                    hide_index=True,
                    column_config={
                        "metric": "Metric",
                        "unit": "Unit",
                        "direction": st.column_config.TextColumn(
                            "Direction",
                            help="Whether higher or lower values are better for this metric"
                        ),
                        "importance": st.column_config.TextColumn(
                            "Importance",
                            help="Weighting applied to this metric (high=3x, medium=2x, low=1x)"
                        ),
                        "description": "Description"
                    }
                )
        
        # Show benchmarks for current sector
        st.subheader(f"Sector Benchmarks for {st.session_state.selected_sector}")
        benchmarks = load_sector_benchmarks(sector=st.session_state.selected_sector)
        
        if not benchmarks.empty:
            st.dataframe(
                benchmarks[['domain', 'metric', 'low_benchmark', 'average_benchmark', 'high_benchmark', 'unit']],
                hide_index=True,
                column_config={
                    "domain": "Domain",
                    "metric": "Metric",
                    "low_benchmark": st.column_config.NumberColumn(
                        "Low Benchmark",
                        help="Lower threshold for typical values in this sector"
                    ),
                    "average_benchmark": st.column_config.NumberColumn(
                        "Average Benchmark",
                        help="Industry average for this sector"
                    ),
                    "high_benchmark": st.column_config.NumberColumn(
                        "High Benchmark",
                        help="Upper threshold for high performers in this sector"
                    ),
                    "unit": "Unit"
                }
            )
        else:
            st.warning(f"No benchmark data available for {st.session_state.selected_sector} sector.")
    
    # Filter data based on selection
    filtered_scores = st.session_state.scores
    if company_filter != "All Companies":
        filtered_scores = filtered_scores[filtered_scores['company'] == company_filter]
    
    # Display scores
    st.subheader("ESG Scores")
    score_df = filtered_scores.copy()
    
    # Add score interpretation
    score_df['rating'] = pd.cut(
        score_df['esg_score'],
        bins=[-float('inf'), -0.75, -0.25, 0.25, 0.75, float('inf')],
        labels=['Poor', 'Below Average', 'Average', 'Above Average', 'Excellent']
    )
    
    # Format displayed dataframe
    display_cols = ['company', 'esg_score', 'rating', 'lower_bound', 'upper_bound', 
                    'environmental_score', 'social_score', 'governance_score']
    
    st.dataframe(score_df[display_cols].style.format({
        'esg_score': '{:.2f}',
        'lower_bound': '{:.2f}',
        'upper_bound': '{:.2f}',
        'environmental_score': '{:.2f}',
        'social_score': '{:.2f}',
        'governance_score': '{:.2f}'
    }))
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overall ESG Scores", 
        "Domain Scores", 
        "Metric Distribution", 
        "Sector Benchmarks",
        "Anomalies", 
        "Significance Tests"
    ])
    
    with tab1:
        st.subheader("Overall ESG Scores with Confidence Intervals")
        fig = plot_esg_scores(st.session_state.results['overall_scores'], company_filter)
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Understanding ESG Scores
        
        ESG scores are standardized on a scale where:
        - **Positive values** indicate better than industry average performance
        - **Negative values** indicate worse than industry average performance
        - **Zero** represents industry average performance
        
        The confidence intervals show the statistical uncertainty in the score based on bootstrap resampling.
        """)
    
    with tab2:
        st.subheader("Domain Scores")
        fig = plot_domain_scores(
            st.session_state.results['domain_scores'],
            company_filter
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Domain Score Comparison
        
        This visualization shows how a company performs across the three ESG domains:
        - **Environmental**: Measures impact on and stewardship of the natural environment
        - **Social**: Assesses relationships with employees, suppliers, customers, and communities
        - **Governance**: Evaluates leadership, executive pay, audits, internal controls, and shareholder rights
        """)
    
    with tab3:
        st.subheader("Metric Distribution Analysis")
        
        # Domain selection for metrics
        domain = st.radio(
            "Select Domain", 
            ["Environmental", "Social", "Governance"],
            horizontal=True
        )
        
        domain_data = None
        domain_name = domain.lower()
        
        if domain_name == "environmental" and st.session_state.env_data is not None:
            domain_data = st.session_state.env_data
        elif domain_name == "social" and st.session_state.social_data is not None:
            domain_data = st.session_state.social_data
        elif domain_name == "governance" and st.session_state.gov_data is not None:
            domain_data = st.session_state.gov_data
        
        if domain_data is not None:
            # Get metrics from the selected domain
            metrics = [col for col in domain_data.columns 
                      if col not in ['company', 'sector', 'year']]
            
            selected_metric = st.selectbox("Select Metric", metrics)
            
            # Display metric description if available
            if domain_name in st.session_state.metric_descriptions:
                desc_df = st.session_state.metric_descriptions[domain_name]
                metric_info = desc_df[desc_df['metric'] == selected_metric]
                
                if len(metric_info) > 0:
                    unit = metric_info['unit'].values[0]
                    description = metric_info['description'].values[0]
                    direction = metric_info['direction'].values[0]
                    importance = metric_info['importance'].values[0].capitalize()
                    
                    st.markdown(f"""
                    **{selected_metric}** ({unit}) - {importance} importance
                    
                    {description}
                    
                    *{direction.replace('_', ' ').title()}*
                    """)
            
            fig = plot_metric_distribution(domain_data, selected_metric, company_filter)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Sector Benchmarks")
        
        # Domain selection for metrics
        domain = st.radio(
            "Select Domain", 
            ["Environmental", "Social", "Governance"],
            horizontal=True,
            key="benchmark_domain"
        )
        
        domain_name = domain.lower()
        
        # Get metrics for this domain
        metrics = get_domain_metrics(domain_name)
        selected_metric = st.selectbox("Select Metric", metrics, key="benchmark_metric")
        
        # Plot sector benchmarks
        fig = plot_sector_benchmarks(
            selected_sector, 
            selected_metric, 
            domain_name,
            company_data=st.session_state.env_data if domain_name == 'environmental' else
                         st.session_state.social_data if domain_name == 'social' else
                         st.session_state.gov_data,
            company_filter=company_filter
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        ### Understanding Sector Benchmarks
        
        The chart above shows:
        - **Low to Average**: The range from the lowest typical value to the industry average
        - **Average to High**: The range from the industry average to the highest typical value
        - The **vertical line** shows where the selected company falls relative to these benchmarks
        
        These benchmarks are used to calculate normalized scores for more accurate cross-company comparisons.
        """)
    
    with tab5:
        st.subheader("Anomaly Detection")
        
        if company_filter == "All Companies":
            st.info("Please select a specific company to view anomalies.")
        else:
            fig = plot_anomalies(
                st.session_state.results['anomalies'],
                company_filter
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Anomaly Interpretation
            
            The chart shows metrics that are statistically unusual (|z| > 3), meaning they are:
            - More than 3 standard deviations away from the mean
            - Part of only about 0.3% of observations in a normal distribution
            - Worthy of special attention, either as potential risks or as areas of exceptional performance
            
            These anomalies may represent:
            - Errors in data reporting that need verification
            - Genuine outlier performance requiring explanation
            - Areas where immediate action may be needed
            """)
    
    with tab6:
        st.subheader("Significance Tests vs Industry Benchmarks")
        
        if company_filter == "All Companies":
            st.info("Please select a specific company to view significance tests.")
        else:
            fig = plot_significance_tests(
                st.session_state.results['significance_tests'],
                company_filter
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            ### Statistical Significance
            
            The chart shows whether a company's performance on each metric is significantly different from the industry benchmark:
            - **Highly significant (p < 0.01)**: Strong evidence of real difference
            - **Significant (p < 0.05)**: Moderate evidence of real difference
            - **Not significant (p >= 0.05)**: Difference might be due to random variation
            
            Points to the right indicate better performance than the benchmark, while points to the left indicate worse performance.
            """)

else:
    # Instructions when no data is loaded
    st.info("Please select a sector and year, then click 'Process Data' to start the analysis.")
    
    st.header("About ESG Navigator")
    
    st.markdown("""
    ### Understanding ESG Analysis
    
    **ESG Navigator** provides standardized analysis of Environmental, Social, and Governance metrics
    across companies within the same sector. The application:
    
    1. **Processes raw metrics** using standardization and normalization techniques
    2. **Calculates scores** with statistical confidence intervals
    3. **Detects anomalies** that may require attention
    4. **Compares performance** against sector-specific benchmarks
    5. **Tests significance** of deviations from industry averages
    
    ### Key Features
    
    - **Sector-specific benchmarking**: Comparing companies to relevant industry standards
    - **Statistical rigor**: Confidence intervals and significance testing
    - **Transparent methodology**: Clear explanation of how scores are calculated
    - **Anomaly detection**: Identifying metrics that require special attention
    - **Comprehensive visualization**: Multiple views to understand ESG performance
    """)
    
    # Display sample sectors and metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Environmental Metrics")
        env_metrics = get_domain_metrics('environmental')
        for metric in env_metrics:
            st.markdown(f"- {metric}")
    
    with col2:
        st.subheader("Social Metrics")
        social_metrics = get_domain_metrics('social')
        for metric in social_metrics:
            st.markdown(f"- {metric}")
    
    with col3:
        st.subheader("Governance Metrics")
        gov_metrics = get_domain_metrics('governance')
        for metric in gov_metrics:
            st.markdown(f"- {metric}")
