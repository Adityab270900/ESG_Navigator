import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from utils import load_sector_benchmarks, load_metric_descriptions

def plot_esg_scores(scores_df, company_filter="All Companies"):
    """
    Plot ESG scores with confidence intervals.
    
    Parameters:
    -----------
    scores_df : pandas.DataFrame
        DataFrame with ESG scores and confidence intervals
    company_filter : str
        Company to filter by, or "All Companies" for all
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if company_filter != "All Companies":
        df = scores_df[scores_df['company'] == company_filter].copy()
    else:
        # Sort by ESG score for better visualization
        df = scores_df.sort_values('esg_score', ascending=False).copy()
    
    # Cap the number of companies to show for readability
    if len(df) > 20 and company_filter == "All Companies":
        df = df.head(20)
        title = "Top 20 Companies by ESG Score"
    else:
        title = "ESG Scores with 95% Confidence Intervals"
    
    # Create the figure
    fig = go.Figure()
    
    # Add the ESG score points
    fig.add_trace(go.Scatter(
        x=df['company'],
        y=df['esg_score'],
        mode='markers',
        marker=dict(size=10, color='royalblue'),
        name='ESG Score'
    ))
    
    # Add error bars for confidence intervals
    fig.add_trace(go.Scatter(
        x=df['company'],
        y=df['esg_score'],
        error_y=dict(
            type='data',
            symmetric=False,
            array=df['upper_bound'] - df['esg_score'],
            arrayminus=df['esg_score'] - df['lower_bound']
        ),
        mode='markers',
        marker=dict(size=10, color='royalblue'),
        name='95% CI',
        showlegend=False
    ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Company",
        yaxis_title="ESG Score",
        height=500,
        xaxis=dict(tickangle=45),
        margin=dict(l=20, r=20, t=40, b=100)
    )
    
    return fig

def plot_domain_scores(domain_scores, company_filter="All Companies"):
    """
    Plot domain-level scores as a radar chart or bar chart.
    
    Parameters:
    -----------
    domain_scores : pandas.DataFrame
        DataFrame with domain-level scores
    company_filter : str
        Company to filter by, or "All Companies" for all
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    # Make sure we're working with a copy
    df = domain_scores.copy()
    
    if company_filter != "All Companies":
        # For a single company, use a radar chart
        company_data = df[df['company'] == company_filter]
        
        if len(company_data) == 0:
            # Handle case where company isn't found
            fig = go.Figure()
            fig.update_layout(
                title=f"No data found for {company_filter}",
                height=500
            )
            return fig
        
        # Extract domain scores
        domains = ['environmental_score', 'social_score', 'governance_score']
        values = company_data[domains].values[0].tolist()
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=['Environmental', 'Social', 'Governance', 'Environmental'],  # Close the polygon
            fill='toself',
            name=company_filter
        ))
        
        fig.update_layout(
            title=f"ESG Domain Scores for {company_filter}",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-3, 3]  # Assuming standardized scores
                )
            ),
            height=500
        )
    
    else:
        # For multiple companies, use a grouped bar chart
        if len(df) > 10:
            # Limit to top 10 companies by average domain score for readability
            df['avg_domain_score'] = df[['environmental_score', 'social_score', 'governance_score']].mean(axis=1)
            df = df.sort_values('avg_domain_score', ascending=False).head(10)
        
        # Melt the dataframe for easier plotting
        plot_df = df.melt(
            id_vars=['company'],
            value_vars=['environmental_score', 'social_score', 'governance_score'],
            var_name='domain',
            value_name='score'
        )
        
        # Clean up domain names for display
        plot_df['domain'] = plot_df['domain'].str.replace('_score', '').str.capitalize()
        
        # Create grouped bar chart
        fig = px.bar(
            plot_df,
            x='company',
            y='score',
            color='domain',
            barmode='group',
            title="ESG Domain Scores by Company",
            labels={'score': 'Score', 'company': 'Company', 'domain': 'Domain'}
        )
        
        fig.update_layout(
            xaxis=dict(tickangle=45),
            height=500,
            margin=dict(l=20, r=20, t=40, b=100)
        )
    
    return fig

def plot_metric_distribution(domain_data, metric, company_filter="All Companies"):
    """
    Plot the distribution of a specific metric with company highlighting.
    
    Parameters:
    -----------
    domain_data : pandas.DataFrame
        DataFrame with ESG metrics for a domain
    metric : str
        The metric to visualize
    company_filter : str
        Company to highlight, or "All Companies" for none
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if metric not in domain_data.columns:
        # Handle case where metric isn't found
        fig = go.Figure()
        fig.update_layout(
            title=f"Metric '{metric}' not found in the data",
            height=500
        )
        return fig
    
    # Create histogram with individual points
    fig = px.histogram(
        domain_data, 
        x=metric,
        marginal="box",
        title=f"Distribution of {metric}",
        labels={metric: f"{metric} Value"},
        opacity=0.7
    )
    
    # Add rug plot for individual observations
    fig.add_trace(
        go.Scatter(
            x=domain_data[metric],
            y=[0.01] * len(domain_data),
            mode='markers',
            marker=dict(
                symbol='line-ns',
                line=dict(width=1, color='blue'),
                size=10
            ),
            name='Companies'
        )
    )
    
    # If a specific company is selected, highlight it
    if company_filter != "All Companies":
        company_data = domain_data[domain_data['company'] == company_filter]
        if len(company_data) > 0:
            company_value = company_data[metric].values[0]
            
            # Add vertical line for the company's value
            fig.add_shape(
                type="line",
                x0=company_value,
                y0=0,
                x1=company_value,
                y1=1,
                yref="paper",
                line=dict(
                    color="red",
                    width=2,
                    dash="dash",
                ),
            )
            
            # Add annotation
            fig.add_annotation(
                x=company_value,
                y=1,
                yref="paper",
                text=f"{company_filter}: {company_value:.2f}",
                showarrow=True,
                arrowhead=1,
                ax=0,
                ay=-40
            )
    
    fig.update_layout(height=500)
    
    return fig

def plot_anomalies(anomalies_df, company_filter):
    """
    Plot anomalies for a specific company.
    
    Parameters:
    -----------
    anomalies_df : pandas.DataFrame
        DataFrame with anomaly flags
    company_filter : str
        Company to filter by
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if anomalies_df is None or len(anomalies_df) == 0:
        # Handle case where no anomalies are found
        fig = go.Figure()
        fig.update_layout(
            title=f"No anomalies detected for {company_filter}",
            height=500
        )
        return fig
    
    # Filter for the selected company
    company_anomalies = anomalies_df[anomalies_df['company'] == company_filter]
    
    if len(company_anomalies) == 0:
        # Handle case where no anomalies are found for this company
        fig = go.Figure()
        fig.update_layout(
            title=f"No anomalies detected for {company_filter}",
            height=500
        )
        return fig
    
    # Create horizontal bar chart of anomaly values
    fig = px.bar(
        company_anomalies,
        y='metric',
        x='value',
        color='domain',
        orientation='h',
        title=f"Anomalies Detected for {company_filter}",
        labels={'value': 'Standardized Value', 'metric': 'Metric', 'domain': 'Domain'},
        hover_data=['flag_reason']
    )
    
    # Add threshold lines
    fig.add_shape(
        type="line",
        x0=3,
        y0=-0.5,
        x1=3,
        y1=len(company_anomalies) - 0.5,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        ),
    )
    
    fig.add_shape(
        type="line",
        x0=-3,
        y0=-0.5,
        x1=-3,
        y1=len(company_anomalies) - 0.5,
        line=dict(
            color="red",
            width=2,
            dash="dash",
        ),
    )
    
    # Add annotations for thresholds
    fig.add_annotation(
        x=3,
        y=len(company_anomalies) - 0.5,
        text="Upper Threshold (z=3)",
        showarrow=False,
        yshift=10
    )
    
    fig.add_annotation(
        x=-3,
        y=len(company_anomalies) - 0.5,
        text="Lower Threshold (z=-3)",
        showarrow=False,
        yshift=10
    )
    
    fig.update_layout(height=500)
    
    return fig

def plot_significance_tests(tests_df, company_filter):
    """
    Plot significance test results for a specific company.
    
    Parameters:
    -----------
    tests_df : pandas.DataFrame
        DataFrame with significance test results
    company_filter : str
        Company to filter by
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    if tests_df is None or len(tests_df) == 0:
        # Handle case where no significance tests are found
        fig = go.Figure()
        fig.update_layout(
            title=f"No significance test results available for {company_filter}",
            height=500
        )
        return fig
    
    # Filter for the selected company
    company_tests = tests_df[tests_df['company'] == company_filter]
    
    if len(company_tests) == 0:
        # Handle case where no tests are found for this company
        fig = go.Figure()
        fig.update_layout(
            title=f"No significance test results available for {company_filter}",
            height=500
        )
        return fig
    
    # Calculate the difference from benchmark for easier visualization
    company_tests['difference'] = company_tests['company_value'] - company_tests['benchmark']
    
    # Map significance to size
    significance_map = {
        'Not significant': 10,
        'Significant': 15,
        'Highly significant': 20
    }
    company_tests['marker_size'] = company_tests['significance'].map(significance_map)
    
    # Map performance to color
    company_tests['color'] = np.where(company_tests['difference'] > 0, 'green', 'red')
    
    # Limit to top 20 metrics by absolute difference for readability
    if len(company_tests) > 20:
        company_tests['abs_diff'] = abs(company_tests['difference'])
        company_tests = company_tests.sort_values('abs_diff', ascending=False).head(20)
    
    # Create scatter plot
    fig = px.scatter(
        company_tests,
        x='difference',
        y='metric',
        color='domain',
        size='marker_size',
        hover_data=['company_value', 'benchmark', 'p_value', 'significance', 'performance'],
        title=f"Performance vs. Industry Benchmarks for {company_filter}",
        labels={
            'difference': 'Difference from Benchmark',
            'metric': 'Metric',
            'domain': 'Domain'
        }
    )
    
    # Add vertical line at zero (benchmark)
    fig.add_shape(
        type="line",
        x0=0,
        y0=-0.5,
        x1=0,
        y1=len(company_tests) - 0.5,
        line=dict(
            color="black",
            width=1,
            dash="dash",
        ),
    )
    
    # Add annotation for the benchmark line
    fig.add_annotation(
        x=0,
        y=len(company_tests) - 0.5,
        text="Industry Benchmark",
        showarrow=False,
        yshift=10
    )
    
    fig.update_layout(height=600)
    
    return fig

def plot_sector_benchmarks(sector, metric, domain, company_data=None, company_filter="All Companies"):
    """
    Plot sector benchmarks for a specific metric with optional company comparison.
    
    Parameters:
    -----------
    sector : str
        The industry sector to show benchmarks for
    metric : str
        The metric to visualize
    domain : str
        The ESG domain the metric belongs to ('environmental', 'social', 'governance')
    company_data : pandas.DataFrame or None
        Optional company data to overlay on the benchmarks
    company_filter : str
        Company to highlight, or "All Companies" for none
        
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    try:
        # Load benchmark data
        benchmark_df = load_sector_benchmarks(sector=sector, metric=metric, domain=domain)
        
        if len(benchmark_df) == 0:
            # No benchmark data found
            fig = go.Figure()
            fig.update_layout(
                title=f"No benchmark data found for {metric} in {sector} sector",
                height=400
            )
            return fig
        
        # Get the first (and should be only) benchmark row
        benchmark = benchmark_df.iloc[0]
        
        # Get metric description
        try:
            desc_df = load_metric_descriptions(domain=domain, metric=metric)
            if len(desc_df) > 0:
                description = desc_df.iloc[0]['description']
                unit = desc_df.iloc[0]['unit']
                direction = desc_df.iloc[0]['direction']
            else:
                description = "No description available"
                unit = ""
                direction = "higher_is_better"
        except:
            description = "No description available"
            unit = ""
            direction = "higher_is_better"
        
        # Create figure
        fig = go.Figure()
        
        # Add benchmark ranges
        low_val = benchmark['low_benchmark']
        avg_val = benchmark['average_benchmark']
        high_val = benchmark['high_benchmark']
        
        # Add low to average range
        fig.add_trace(go.Bar(
            x=[avg_val - low_val],
            y=['Benchmark'],
            base=low_val,
            marker_color='rgba(220, 220, 220, 0.8)',
            name='Low to Average',
            orientation='h',
            hoverinfo='text',
            hovertext=f"Low: {low_val} {unit}<br>Average: {avg_val} {unit}"
        ))
        
        # Add average to high range
        fig.add_trace(go.Bar(
            x=[high_val - avg_val],
            y=['Benchmark'],
            base=avg_val,
            marker_color='rgba(144, 238, 144, 0.6)',
            name='Average to High',
            orientation='h',
            hoverinfo='text',
            hovertext=f"Average: {avg_val} {unit}<br>High: {high_val} {unit}"
        ))
        
        # If company data is provided and a specific company is selected
        if company_data is not None and company_filter != "All Companies":
            # Filter for selected company
            company_row = company_data[company_data['company'] == company_filter]
            
            if len(company_row) > 0 and metric in company_row.columns:
                # Get company value
                company_val = company_row[metric].values[0]
                
                # Add company marker
                fig.add_trace(go.Scatter(
                    x=[company_val],
                    y=['Company'],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        size=15,
                        color='crimson'
                    ),
                    name=company_filter,
                    hoverinfo='text',
                    hovertext=f"{company_filter}: {company_val} {unit}"
                ))
                
                # Add vertical line for company value
                fig.add_shape(
                    type="line",
                    x0=company_val,
                    y0=-0.5,
                    x1=company_val,
                    y1=1.5,
                    line=dict(
                        color="crimson",
                        width=2,
                        dash="dash",
                    )
                )
        
        # Add vertical line for average benchmark
        fig.add_shape(
            type="line",
            x0=avg_val,
            y0=-0.5,
            x1=avg_val,
            y1=1.5,
            line=dict(
                color="black",
                width=2,
                dash="dot",
            )
        )
        
        # Add annotations
        fig.add_annotation(
            x=avg_val,
            y=1.3,
            text=f"Industry Average: {avg_val} {unit}",
            showarrow=False,
            yshift=10
        )
        
        # Set title and layout
        performance_direction = "Lower is better" if direction == "lower_is_better" else "Higher is better"
        
        fig.update_layout(
            title=f"{metric} Benchmarks for {sector} Sector<br><sup>{performance_direction}</sup>",
            xaxis_title=f"Value ({unit})" if unit else "Value",
            yaxis=dict(
                showticklabels=False,
                showgrid=False
            ),
            height=400,
            hovermode="closest",
            showlegend=True,
            margin=dict(l=20, r=20, t=80, b=20)
        )
        
        # Add description as annotation
        fig.add_annotation(
            text=description,
            align="left",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.15,
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        return fig
        
    except Exception as e:
        # Handle any errors
        fig = go.Figure()
        fig.update_layout(
            title=f"Error plotting benchmarks: {str(e)}",
            height=400
        )
        return fig
