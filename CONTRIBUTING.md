# Contributing to ESG Navigator

Thank you for considering contributing to ESG Navigator! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How Can I Contribute?

### Reporting Bugs

Before creating a bug report:
- Check the issue tracker to see if the bug has already been reported
- Collect as much information as possible about the issue

When creating a bug report, include:
- A clear and descriptive title
- Detailed steps to reproduce the issue
- Expected behavior vs actual behavior
- Screenshots if applicable
- Environment information (OS, browser, Python version, etc.)

### Suggesting Enhancements

Enhancement suggestions are welcome! When suggesting enhancements:
- Use a clear and descriptive title
- Provide a detailed description of the proposed enhancement
- Explain why this enhancement would be useful to ESG Navigator users
- Include examples of how it would work if possible

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Write your code following the coding standards
4. Add or update tests as necessary
5. Update documentation to reflect your changes
6. Submit a pull request with a clear description of the changes

## Development Setup

### Prerequisites

- Python 3.8+
- Required packages listed in the README

### Installation for Development

1. Clone your fork of the repository
2. Install dependencies with `pip install -r requirements.txt`
3. Run the application with `streamlit run app.py`

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use meaningful variable and function names
- Include docstrings for all functions, classes, and modules
- Keep lines under 100 characters when possible

### Testing

- Write unit tests for new functionality
- Ensure all tests pass before submitting a pull request
- Aim for high test coverage for critical components

### Documentation

- Update documentation when changing code functionality
- Document all public functions, classes, and modules
- Keep the README up to date

## Adding New Features

### Data Processing

When adding new data processing capabilities:
- Add the functionality to the `DataProcessor` class in `data_processor.py`
- Ensure it handles edge cases gracefully
- Document the method and its parameters

### Visualization

When adding new visualizations:
- Add the visualization function to `visualization.py`
- Follow the existing pattern for function parameters
- Add the visualization to the appropriate tab in `app.py`

### Metrics

When adding new ESG metrics:
- Update sample data files to include the new metrics
- Add descriptions to `metric_descriptions.csv`
- Add benchmarks to `sector_benchmarks.csv`

## Release Process

The maintainers follow this process for releases:
1. Update version number in relevant files
2. Update the changelog
3. Create a new release on GitHub with release notes
4. Publish the release

## Questions?

If you have any questions about contributing, please open an issue labeled 'question' in the issue tracker.

Thank you for contributing to ESG Navigator!