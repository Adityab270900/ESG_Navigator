import pandas as pd
import io
import numpy as np
import os

def load_sample_data(domain, sector=None, year=None):
    """
    Load sample data from the data directory.
    
    Parameters:
    -----------
    domain : str
        ESG domain ('environmental', 'social', 'governance')
    sector : str or None
        Sector to filter data by
    year : str or None
        Year to filter data by
        
    Returns:
    --------
    pandas.DataFrame
        Loaded and filtered data
    """
    file_path = f"data/{domain}_data.csv"
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    
    # Apply filters if provided
    if sector is not None:
        df = df[df['sector'] == sector]
    
    if year is not None:
        df = df[df['year'] == year]
    
    # If no data left after filtering, raise an error
    if len(df) == 0:
        raise ValueError(f"No data available for domain={domain}, sector={sector}, year={year}")
    
    return df

def load_sector_benchmarks(sector=None, metric=None, domain=None):
    """
    Load sector benchmarks from CSV.
    
    Parameters:
    -----------
    sector : str or None
        Sector to filter benchmarks by
    metric : str or None
        Metric to filter benchmarks by
    domain : str or None
        Domain to filter benchmarks by
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with benchmark data
    """
    try:
        file_path = "data/sector_benchmarks.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Benchmark file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Apply filters if provided
        if sector is not None:
            df = df[df['sector'] == sector]
        
        if metric is not None:
            df = df[df['metric'] == metric]
            
        if domain is not None:
            df = df[df['domain'] == domain]
        
        return df
    except Exception as e:
        # Log the error for debugging
        print(f"Error loading sector benchmarks: {str(e)}")
        # Return an empty DataFrame to avoid breaking the app
        return pd.DataFrame(columns=['sector', 'metric', 'domain', 'low_benchmark', 
                                    'average_benchmark', 'high_benchmark', 'unit'])

def load_metric_descriptions(domain=None, metric=None):
    """
    Load metric descriptions from CSV.
    
    Parameters:
    -----------
    domain : str or None
        Domain to filter by
    metric : str or None
        Metric to filter by
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with metric descriptions
    """
    try:
        file_path = "data/metric_descriptions.csv"
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Metric descriptions file not found: {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Apply filters if provided
        if domain is not None:
            df = df[df['domain'] == domain]
        
        if metric is not None:
            df = df[df['metric'] == metric]
        
        return df
    except Exception as e:
        # Log the error for debugging
        print(f"Error loading metric descriptions: {str(e)}")
        # Return an empty DataFrame to avoid breaking the app
        return pd.DataFrame(columns=['domain', 'metric', 'unit', 'direction', 
                                    'importance', 'description'])

def load_data(file):
    """
    Load data from a file upload.
    
    Parameters:
    -----------
    file : UploadedFile
        Uploaded file object from Streamlit
        
    Returns:
    --------
    pandas.DataFrame
        Loaded data
    """
    if file is None:
        return None
    
    try:
        # Reset the file pointer to the beginning
        file.seek(0)
        
        # Read the file content
        content = file.read()
        
        # Determine the file type and load accordingly
        if file.name.endswith('.csv'):
            return pd.read_csv(io.BytesIO(content))
        elif file.name.endswith('.json'):
            return pd.read_json(io.BytesIO(content))
        else:
            raise ValueError(f"Unsupported file format: {file.name}")
    
    except Exception as e:
        raise Exception(f"Error loading file {file.name}: {str(e)}")

def get_default_weights(domain, metrics, metric_descriptions=None):
    """
    Generate default weights for metrics within a domain.
    
    Parameters:
    -----------
    domain : str
        ESG domain ('environmental', 'social', 'governance')
    metrics : list
        List of metrics for the domain
    metric_descriptions : pandas.DataFrame or None
        DataFrame containing metric importance values
        
    Returns:
    --------
    dict
        Dictionary of metric weights that sum to 1.0
    """
    # If we have metric descriptions, use importance to weight
    if metric_descriptions is not None:
        weights = {}
        importance_map = {'high': 3, 'medium': 2, 'low': 1}
        total_importance = 0
        
        for metric in metrics:
            metric_info = metric_descriptions[metric_descriptions['metric'] == metric]
            if len(metric_info) > 0:
                importance = metric_info['importance'].values[0]
                weight = importance_map.get(importance, 2)  # Default to medium if not found
                weights[metric] = weight
                total_importance += weight
            else:
                # Default weight if metric not found in descriptions
                weights[metric] = 2
                total_importance += 2
        
        # Normalize weights to sum to 1.0
        for metric in weights:
            weights[metric] = weights[metric] / total_importance
            
        return weights
    
    # Simple equal weighting if no descriptions available
    return {metric: 1.0 / len(metrics) for metric in metrics}

def get_available_sectors():
    """
    Get a list of available sectors from the data.
    
    Returns:
    --------
    list
        List of sector names
    """
    try:
        # Try to load any domain data to get sectors
        env_data = pd.read_csv("data/environmental_data.csv")
        return sorted(env_data['sector'].unique().tolist())
    except:
        # Fallback if file doesn't exist
        return ["Technology", "Energy", "Construction", "Healthcare", "Materials"]

def get_available_years():
    """
    Get a list of available years from the data.
    
    Returns:
    --------
    list
        List of years
    """
    try:
        # Try to load any domain data to get years
        env_data = pd.read_csv("data/environmental_data.csv")
        return sorted(env_data['year'].unique().tolist())
    except:
        # Fallback if file doesn't exist
        return ["2023"]

def get_companies_for_sector_year(sector, year):
    """
    Get a list of companies for a specific sector and year.
    
    Parameters:
    -----------
    sector : str
        Sector to filter by
    year : int or str
        Year to filter by
        
    Returns:
    --------
    list
        List of company names
    """
    try:
        # Try to load environmental data
        env_data = pd.read_csv("data/environmental_data.csv")
        filtered = env_data[(env_data['sector'] == sector) & (env_data['year'] == year)]
        return sorted(filtered['company'].unique().tolist())
    except:
        # Fallback if file doesn't exist
        return []

def get_domain_metrics(domain):
    """
    Get a list of metrics for a specific domain.
    
    Parameters:
    -----------
    domain : str
        ESG domain ('environmental', 'social', 'governance')
        
    Returns:
    --------
    list
        List of metric names
    """
    try:
        # Load metric descriptions for the domain
        desc = load_metric_descriptions(domain=domain)
        return desc['metric'].tolist()
    except:
        # Fallback if file doesn't exist
        if domain == 'environmental':
            return ['carbon_intensity', 'renewable_energy_pct', 'water_usage_intensity', 
                    'waste_recycling_rate', 'emissions_reduction_target']
        elif domain == 'social':
            return ['gender_diversity_pct', 'employee_satisfaction', 'community_investment_pct', 
                    'health_safety_incidents', 'training_hours_per_employee', 'supply_chain_human_rights_score']
        elif domain == 'governance':
            return ['board_independence_pct', 'ethics_violations', 'executive_compensation_ratio', 
                    'shareholder_rights_score', 'audit_committee_score', 'anti_corruption_policy_score']
        else:
            return []

def align_reporting_periods(env_data, social_data, gov_data):
    """
    Align reporting periods across all ESG tables.
    Assumes data includes a date/period column.
    
    Parameters:
    -----------
    env_data : pandas.DataFrame
        Environmental data
    social_data : pandas.DataFrame
        Social data
    gov_data : pandas.DataFrame
        Governance data
        
    Returns:
    --------
    tuple
        Tuple of aligned DataFrames
    """
    # Filter to common companies, sectors, and years
    common_companies = set(env_data['company']).intersection(
        set(social_data['company']), set(gov_data['company']))
    
    if 'sector' in env_data.columns and 'sector' in social_data.columns and 'sector' in gov_data.columns:
        common_sectors = set(env_data['sector']).intersection(
            set(social_data['sector']), set(gov_data['sector']))
    else:
        common_sectors = None
        
    if 'year' in env_data.columns and 'year' in social_data.columns and 'year' in gov_data.columns:
        common_years = set(env_data['year']).intersection(
            set(social_data['year']), set(gov_data['year']))
    else:
        common_years = None
    
    # Apply filters
    env_filtered = env_data[env_data['company'].isin(common_companies)]
    social_filtered = social_data[social_data['company'].isin(common_companies)]
    gov_filtered = gov_data[gov_data['company'].isin(common_companies)]
    
    if common_sectors:
        env_filtered = env_filtered[env_filtered['sector'].isin(common_sectors)]
        social_filtered = social_filtered[social_filtered['sector'].isin(common_sectors)]
        gov_filtered = gov_filtered[gov_filtered['sector'].isin(common_sectors)]
        
    if common_years:
        env_filtered = env_filtered[env_filtered['year'].isin(common_years)]
        social_filtered = social_filtered[social_filtered['year'].isin(common_years)]
        gov_filtered = gov_filtered[gov_filtered['year'].isin(common_years)]
        
    return env_filtered, social_filtered, gov_filtered

def normalize_metric_data(data, method='zscore', benchmark=None, reverse=False):
    """
    Normalize metric data using various methods.
    
    Parameters:
    -----------
    data : pandas.Series or numpy.ndarray
        Data to normalize
    method : str
        Normalization method: 'zscore', 'minmax', 'robust', or 'benchmark'
    benchmark : dict or None
        Benchmark values with keys: 'low', 'average', 'high'
    reverse : bool
        If True, reverse the normalization (for metrics where lower is better)
        
    Returns:
    --------
    numpy.ndarray
        Normalized data
    """
    data_array = np.array(data)
    
    if method == 'benchmark' and benchmark is not None:
        # Use sector benchmarks for normalization
        low = benchmark['low_benchmark']
        avg = benchmark['average_benchmark']
        high = benchmark['high_benchmark']
        
        # Calculate normalized scores based on position relative to benchmarks
        normalized = np.zeros_like(data_array, dtype=float)
        
        for i, val in enumerate(data_array):
            if val <= low:
                normalized[i] = -1.0  # Below low benchmark
            elif val <= avg:
                # Between low and average (map to -1 to 0)
                normalized[i] = -1.0 + (val - low) / (avg - low)
            elif val <= high:
                # Between average and high (map to 0 to 1)
                normalized[i] = (val - avg) / (high - avg)
            else:
                normalized[i] = 1.0  # Above high benchmark
                
        if reverse:
            normalized = -normalized
            
        return normalized
    
    elif method == 'zscore':
        # Z-score standardization
        mean = np.mean(data_array)
        std = np.std(data_array)
        if std == 0:
            return np.zeros_like(data_array)
        normalized = (data_array - mean) / std
        
    elif method == 'minmax':
        # Min-Max scaling to [0, 1]
        min_val = np.min(data_array)
        max_val = np.max(data_array)
        if max_val == min_val:
            return np.zeros_like(data_array)
        normalized = (data_array - min_val) / (max_val - min_val)
        # Convert from [0, 1] to [-1, 1]
        normalized = normalized * 2 - 1
        
    elif method == 'robust':
        # Robust scaling using median and IQR
        median = np.median(data_array)
        q1 = np.percentile(data_array, 25)
        q3 = np.percentile(data_array, 75)
        iqr = q3 - q1
        if iqr == 0:
            return np.zeros_like(data_array)
        normalized = (data_array - median) / iqr
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Reverse if lower is better
    if reverse:
        normalized = -normalized
        
    return normalized
