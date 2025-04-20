"""
Data Processing Module for ESG Navigator.

This module handles the processing of ESG metrics data, including data cleaning,
missing value imputation, outlier detection and handling, and standardization.
It supports sector-specific benchmarking for more accurate comparisons.

Author: ESG Navigator Team
Date: April 2025
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from utils import load_metric_descriptions, load_sector_benchmarks, normalize_metric_data

class DataProcessor:
    """
    Class for processing ESG data including cleaning, imputation, 
    outlier detection, and standardization.
    
    This class provides methods for preparing ESG metrics data for analysis.
    It handles common data quality issues such as missing values and outliers,
    and standardizes metrics to make them comparable across companies and sectors.
    
    Attributes:
        normalization_info (dict): Stores information about how each metric was normalized,
                                  which is useful for interpretability and transparency.
    """
    
    def __init__(self):
        """
        Initialize the data processor.
        
        Sets up an empty dictionary to store normalization information for each metric,
        which will be populated during the data processing steps.
        """
        self.normalization_info = {}  # Store normalization info for each metric
    
    def process_data(self, data, domain, use_benchmarks=True):
        """
        Process a dataset for a specific ESG domain.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            Raw data with company names and ESG metrics
        domain : str
            The ESG domain ('environmental', 'social', 'governance')
        use_benchmarks : bool
            Whether to use sector benchmarks for normalization
            
        Returns:
        --------
        pandas.DataFrame
            Processed data with cleaned and standardized metrics
        """
        if data is None or data.empty:
            raise ValueError(f"No data provided for {domain} domain")
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure 'company' column exists
        if 'company' not in df.columns:
            raise ValueError(f"{domain} data must contain a 'company' column")
        
        # Extract columns for processing
        id_cols = ['company', 'sector', 'year']
        
        # Keep only the id columns that exist in the dataset
        id_cols = [col for col in id_cols if col in df.columns]
            
        numeric_cols = [col for col in df.columns if col not in id_cols]
        
        if not numeric_cols:
            raise ValueError(f"No metric columns found in {domain} data")
        
        # Load metric descriptions to identify "lower is better" metrics
        try:
            metric_desc = load_metric_descriptions(domain=domain)
            metric_directions = {
                row['metric']: row['direction'] == 'lower_is_better' 
                for _, row in metric_desc.iterrows()
            }
        except:
            # Default to assuming higher is better if descriptions aren't available
            metric_directions = {col: False for col in numeric_cols}
        
        # 1. Handle missing values with KNN imputation
        df_numeric = df[numeric_cols]
        imputer = KNNImputer(n_neighbors=min(5, len(df_numeric) - 1), weights='distance')
        imputed_data = imputer.fit_transform(df_numeric)
        df_imputed = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        
        # 2. Detect and handle outliers (Winsorization at 1st/99th percentiles)
        for col in numeric_cols:
            lower_bound = np.percentile(df_imputed[col], 1)
            upper_bound = np.percentile(df_imputed[col], 99)
            df_imputed[col] = np.clip(df_imputed[col], lower_bound, upper_bound)
        
        # 3. Apply appropriate standardization based on metric characteristics
        # and sector benchmarks if available
        processed_df = df[id_cols].copy()
        
        for col in numeric_cols:
            values = df_imputed[col].values
            reverse = metric_directions.get(col, False)
            
            # Check if we should use sector benchmarks
            if use_benchmarks and 'sector' in df.columns:
                sectors = df['sector'].unique()
                
                # Process each sector with its own benchmarks
                for sector in sectors:
                    # Get benchmarks for this sector and metric
                    try:
                        benchmark_df = load_sector_benchmarks(
                            sector=sector, metric=col, domain=domain)
                        
                        if len(benchmark_df) > 0:
                            benchmark = benchmark_df.iloc[0].to_dict()
                            
                            # Get rows for this sector
                            sector_mask = df['sector'] == sector
                            sector_indices = df.index[sector_mask]
                            
                            # Use benchmark normalization for this sector
                            sector_values = df_imputed.loc[sector_indices, col].values
                            normalized = normalize_metric_data(
                                sector_values, method='benchmark', 
                                benchmark=benchmark, reverse=reverse)
                            
                            # Add normalized values to a new column
                            processed_df.loc[sector_indices, col] = normalized
                            
                            # Store normalization info
                            self.normalization_info[f"{domain}_{col}_{sector}"] = {
                                'method': 'benchmark',
                                'benchmark': benchmark,
                                'reverse': reverse
                            }
                        else:
                            # Fallback to Z-score if benchmark not found
                            sector_mask = df['sector'] == sector
                            sector_indices = df.index[sector_mask]
                            
                            sector_values = df_imputed.loc[sector_indices, col].values
                            # For sector-specific z-scores, we normalize within the sector
                            normalized = normalize_metric_data(
                                sector_values, method='zscore', reverse=reverse)
                            
                            processed_df.loc[sector_indices, col] = normalized
                            
                            # Store normalization info
                            self.normalization_info[f"{domain}_{col}_{sector}"] = {
                                'method': 'zscore',
                                'reverse': reverse
                            }
                    
                    except Exception as e:
                        # Fallback to Z-score if benchmark loading fails
                        sector_mask = df['sector'] == sector
                        sector_indices = df.index[sector_mask]
                        
                        sector_values = df_imputed.loc[sector_indices, col].values
                        normalized = normalize_metric_data(
                            sector_values, method='zscore', reverse=reverse)
                        
                        processed_df.loc[sector_indices, col] = normalized
                        
                        # Store normalization info
                        self.normalization_info[f"{domain}_{col}_{sector}"] = {
                            'method': 'zscore',
                            'reverse': reverse
                        }
            
            else:
                # No sector benchmarks available or not using them
                # Determine the appropriate scaling method based on data characteristics
                
                # Check if it's a ratio/percentage (bounded between 0 and 1 or close to that)
                if 0 <= df_imputed[col].min() and df_imputed[col].max() <= 1.1:
                    method = 'minmax'
                # Check if it's heavily skewed
                elif abs(df_imputed[col].skew()) > 1.0:
                    method = 'robust'
                # Otherwise use Z-score standardization
                else:
                    method = 'zscore'
                
                # Apply normalization
                normalized = normalize_metric_data(
                    values, method=method, reverse=reverse)
                
                processed_df[col] = normalized
                
                # Store normalization info
                self.normalization_info[f"{domain}_{col}"] = {
                    'method': method,
                    'reverse': reverse
                }
        
        return processed_df
