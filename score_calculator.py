import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor

class ScoreCalculator:
    """
    Class for calculating ESG scores with uncertainty quantification,
    anomaly detection, and significance testing.
    """
    
    def __init__(self, domain_weights=None, metric_weights=None, 
                 bootstrap_iterations=1000, confidence_level=0.95):
        """
        Initialize the score calculator with weights and parameters.
        
        Parameters:
        -----------
        domain_weights : dict
            Weights for each ESG domain (environmental, social, governance)
        metric_weights : dict of dicts
            Weights for metrics within each domain
        bootstrap_iterations : int
            Number of bootstrap iterations for uncertainty quantification
        confidence_level : float
            Confidence level for intervals (0.0-1.0)
        """
        # Default domain weights if not provided
        self.domain_weights = domain_weights or {
            'environmental': 0.4,
            'social': 0.35,
            'governance': 0.25
        }
        
        # Metric weights will be set during calculation if not provided
        self.metric_weights = metric_weights or {}
        
        self.bootstrap_iterations = bootstrap_iterations
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_scores(self, env_data, social_data, gov_data):
        """
        Calculate ESG scores from processed domain data.
        
        Parameters:
        -----------
        env_data : pandas.DataFrame
            Processed environmental data
        social_data : pandas.DataFrame
            Processed social data
        gov_data : pandas.DataFrame
            Processed governance data
            
        Returns:
        --------
        dict
            Dictionary containing:
            - domain_scores: DataFrame with domain-level scores
            - overall_scores: DataFrame with overall ESG scores and CIs
            - anomalies: DataFrame with flagged anomalies
            - significance_tests: DataFrame with significance test results
        """
        # Validate input data
        if env_data is None or social_data is None or gov_data is None:
            raise ValueError("All three ESG domains must have data")
        
        # Ensure all datasets have the same companies
        env_companies = set(env_data['company'])
        social_companies = set(social_data['company'])
        gov_companies = set(gov_data['company'])
        
        common_companies = env_companies.intersection(social_companies, gov_companies)
        
        if not common_companies:
            raise ValueError("No common companies found across all three datasets")
        
        # Filter to common companies
        env_data = env_data[env_data['company'].isin(common_companies)]
        social_data = social_data[social_data['company'].isin(common_companies)]
        gov_data = gov_data[gov_data['company'].isin(common_companies)]
        
        # Keep only necessary columns for scoring (remove sector, year if present)
        id_cols = ['company']
        exclude_cols = ['sector', 'year']
        
        # Process each dataset to remove non-metric columns
        env_filtered = env_data.copy()
        social_filtered = social_data.copy()
        gov_filtered = gov_data.copy()
        
        # Only keep id_cols and numeric metric columns
        for df in [env_filtered, social_filtered, gov_filtered]:
            # Remove columns that should be excluded
            for col in exclude_cols:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
        
        # Generate default metric weights if not provided
        domains = {
            'environmental': env_filtered,
            'social': social_filtered,
            'governance': gov_filtered
        }
        
        for domain, data in domains.items():
            metric_cols = [col for col in data.columns if col not in id_cols]
            if domain not in self.metric_weights:
                # Equal weights if not specified
                self.metric_weights[domain] = {col: 1/len(metric_cols) for col in metric_cols}
        
        # Calculate domain scores
        env_scores = self._calculate_domain_score(env_filtered, 'environmental')
        social_scores = self._calculate_domain_score(social_filtered, 'social')
        gov_scores = self._calculate_domain_score(gov_filtered, 'governance')
        
        # Combine domain scores
        domain_scores = pd.merge(env_scores, social_scores, on='company')
        domain_scores = pd.merge(domain_scores, gov_scores, on='company')
        
        # Calculate overall ESG scores with uncertainty quantification
        overall_scores = self._calculate_overall_score(env_filtered, social_filtered, gov_filtered)
        
        # Detect anomalies
        anomalies = self._detect_anomalies(env_filtered, social_filtered, gov_filtered)
        
        # Perform significance tests against industry benchmarks
        significance_tests = self._perform_significance_tests(env_filtered, social_filtered, gov_filtered)
        
        return {
            'domain_scores': domain_scores,
            'overall_scores': overall_scores,
            'anomalies': anomalies,
            'significance_tests': significance_tests
        }
    
    def _calculate_domain_score(self, data, domain):
        """Calculate weighted score for a specific domain."""
        companies = data['company'].unique()
        metric_cols = [col for col in data.columns if col != 'company']
        weights = self.metric_weights[domain]
        
        # Ensure weights are normalized to sum to 1
        weight_sum = sum(weights.values())
        normalized_weights = {k: v/weight_sum for k, v in weights.items()}
        
        scores = []
        for company in companies:
            company_data = data[data['company'] == company]
            
            # Calculate weighted sum of metrics
            domain_score = 0
            for col in metric_cols:
                if col in normalized_weights:
                    domain_score += company_data[col].values[0] * normalized_weights[col]
            
            scores.append({
                'company': company,
                f'{domain}_score': domain_score
            })
        
        return pd.DataFrame(scores)
    
    def _bootstrap_score_calculation(self, env_sample, social_sample, gov_sample, company):
        """Calculate ESG score for a bootstrap sample."""
        # Calculate domain scores for this sample
        env_score = sum(env_sample[col] * self.metric_weights['environmental'][col] 
                        for col in env_sample.index if col != 'company')
        social_score = sum(social_sample[col] * self.metric_weights['social'][col] 
                          for col in social_sample.index if col != 'company')
        gov_score = sum(gov_sample[col] * self.metric_weights['governance'][col] 
                        for col in gov_sample.index if col != 'company')
        
        # Calculate overall ESG score
        esg_score = (
            env_score * self.domain_weights['environmental'] +
            social_score * self.domain_weights['social'] +
            gov_score * self.domain_weights['governance']
        )
        
        return esg_score
    
    def _calculate_overall_score(self, env_data, social_data, gov_data):
        """
        Calculate overall ESG scores with bootstrap confidence intervals.
        """
        companies = env_data['company'].unique()
        results = []
        
        for company in companies:
            # Get company data for each domain
            env_company = env_data[env_data['company'] == company].iloc[0]
            social_company = social_data[social_data['company'] == company].iloc[0]
            gov_company = gov_data[gov_data['company'] == company].iloc[0]
            
            # Calculate point estimate
            env_score = sum(env_company[col] * self.metric_weights['environmental'][col] 
                            for col in env_company.index if col != 'company')
            social_score = sum(social_company[col] * self.metric_weights['social'][col] 
                              for col in social_company.index if col != 'company')
            gov_score = sum(gov_company[col] * self.metric_weights['governance'][col] 
                            for col in gov_company.index if col != 'company')
            
            # Calculate overall ESG score
            esg_score = (
                env_score * self.domain_weights['environmental'] +
                social_score * self.domain_weights['social'] +
                gov_score * self.domain_weights['governance']
            )
            
            # Bootstrap to calculate confidence intervals
            bootstrap_scores = []
            
            for _ in range(self.bootstrap_iterations):
                # Resample with replacement for each domain
                env_sample = env_company.copy()
                social_sample = social_company.copy()
                gov_sample = gov_company.copy()
                
                # Add random noise to simulate resampling (since we only have one observation per company)
                for col in env_sample.index:
                    if col != 'company':
                        env_sample[col] += np.random.normal(0, 0.1)
                
                for col in social_sample.index:
                    if col != 'company':
                        social_sample[col] += np.random.normal(0, 0.1)
                
                for col in gov_sample.index:
                    if col != 'company':
                        gov_sample[col] += np.random.normal(0, 0.1)
                
                # Calculate bootstrap score
                bootstrap_score = self._bootstrap_score_calculation(
                    env_sample, social_sample, gov_sample, company
                )
                bootstrap_scores.append(bootstrap_score)
            
            # Calculate confidence intervals
            lower_bound = np.percentile(bootstrap_scores, (self.alpha/2) * 100)
            upper_bound = np.percentile(bootstrap_scores, (1 - self.alpha/2) * 100)
            
            results.append({
                'company': company,
                'esg_score': esg_score,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'std_error': np.std(bootstrap_scores),
                'environmental_score': env_score,
                'social_score': social_score,
                'governance_score': gov_score
            })
        
        return pd.DataFrame(results)
    
    def _detect_anomalies(self, env_data, social_data, gov_data):
        """
        Detect anomalies in the ESG metrics.
        Flag metrics with |z| > 3 for manual review.
        """
        all_anomalies = []
        
        domains = {
            'environmental': env_data,
            'social': social_data,
            'governance': gov_data
        }
        
        for domain_name, data in domains.items():
            metric_cols = [col for col in data.columns if col != 'company']
            
            for company in data['company'].unique():
                company_data = data[data['company'] == company]
                
                for metric in metric_cols:
                    value = company_data[metric].values[0]
                    
                    # The data is already standardized, so we can check if |z| > 3
                    if abs(value) > 3:
                        all_anomalies.append({
                            'company': company,
                            'domain': domain_name,
                            'metric': metric,
                            'value': value,
                            'flag_reason': f'Extreme value (|z| > 3)'
                        })
        
        return pd.DataFrame(all_anomalies)
    
    def _perform_significance_tests(self, env_data, social_data, gov_data):
        """
        Perform significance tests against industry benchmarks (using dataset averages).
        """
        test_results = []
        
        domains = {
            'environmental': env_data,
            'social': social_data,
            'governance': gov_data
        }
        
        for domain_name, data in domains.items():
            metric_cols = [col for col in data.columns if col != 'company']
            
            for metric in metric_cols:
                # Calculate industry benchmark (mean of all companies)
                benchmark = data[metric].mean()
                benchmark_std = data[metric].std()
                
                for company in data['company'].unique():
                    company_value = data[data['company'] == company][metric].values[0]
                    
                    # Perform simple one-sample t-test
                    t_stat = (company_value - benchmark) / (benchmark_std / np.sqrt(len(data)))
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(data)-1))
                    
                    significance = None
                    if p_value < 0.01:
                        significance = 'Highly significant'
                    elif p_value < 0.05:
                        significance = 'Significant'
                    else:
                        significance = 'Not significant'
                    
                    performance = None
                    if company_value > benchmark:
                        performance = 'Outperforming'
                    else:
                        performance = 'Underperforming'
                    
                    test_results.append({
                        'company': company,
                        'domain': domain_name,
                        'metric': metric,
                        'company_value': company_value,
                        'benchmark': benchmark,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significance': significance,
                        'performance': performance
                    })
        
        return pd.DataFrame(test_results)
