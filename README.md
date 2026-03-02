# HEINEKEN UK: Single View of Customer (SVOC)

The Single View of Customer (**SVOC**) is a record linkage solution designed to match Beavertown on-trade client records from multiple data sources. Using probabilistic and rule-based matching techniques, SVOC identifies potential matches between datasets, enabling efficient data consolidation and customer identification.

## Overview

SVOC compares records from two datasets (benchmark and input) containing outlet information such as:

- Unique identifier
- Outlet name
- Address
- Postal code
- Geographic coordinates (latitude/longitude)

The algorithm generates N potential matches for each benchmark record by computing similarity scores across multiple dimensions (name, address, location). This allows manual verification and selection of the most appropriate match.

## Key Features

- **Scalable Matching**: K-NN clustering reduces computational complexity by limiting comparisons to geographically proximate records;
- **Multi-Strategy Approach**: Combines rule-based automatic matching with supervised machine learning models;
- **Flexible Configuration**: YAML-based configuration for easy customization;
- **Multiple Similarity Metrics**: Leverages various string similarity algorithms (Levenshtein, Jaro-Winkler, etc.);
- **Trained Models**: Includes pre-trained supervised models for probabilistic matching.

## Repository Structure

├── config/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Configuration files  
├── doc/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Documentation  
├── models/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Trained models  
├── script/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  
│ ├── 00_postcode_dist.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Postcode distance analysis  
│ ├── [01_training.py](/script/01_training.py)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Model training script  
│ ├── 02_code.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Additional utilities  
│ └── [main.py](/script/main.py) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Main pipeline entry point  
├── svoc/&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Core Package  
│ ├── automatic/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Automatic matching module  
│ ├── supervised/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Supervised learning module  
│ ├── constants.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Global constants  
│ ├── datapreparation.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Data preprocessing  
│ ├── orchestrator.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Pipeline orchestration  
│ ├── rl.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Record linkage logic  
│ ├── settings.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Settings management  
│ └── utils.py&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Utility functions  
├── .env &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Environment variables   
├── requirements.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Requirements   
├── requirements_runtime.txt&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Databricks requirements   
└── README.md

## Installation

### Prerequisites

- Python 3.12 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/SDGGroup/HUKsvoc.git
cd HUKsvoc
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Install the package:

```bash
pip install dist/*.whl
```

For development mode:

```bash
pip install -e .
```


### Configuration

Edit the YAML configuration file in the [config](/config/) directory *or* the [.env](.env) file, to specify:

- Data file paths and table names
- Column mappings for input and benchmark datasets
- Number of matches to return (N_MATCHES)
- K-NN parameters (K_NEIGHBOURS)
- Model paths and settings

Example configuration ([dev.yaml](/config/dev.yaml)):

```yaml
DATA_DIR: "./data"

BENCHMARK_DATA_FILENAME: "HUK_sap_data.csv"
BENCHMARK_COLUMNS:
  ID: 'SapCode'
  OUTLET_NAME: 'OutletName'
  POSTCODE: 'OutletPostcode'
  ADDRESS: 'OutletAddress'

INPUT_DATA_FILENAME: "HUK_bowimi_data.csv"
INPUT_COLUMNS:
  ID: 'BowimiId'
  OUTLET_NAME: 'OutletName'
  POSTCODE: 'OutletPostCode'
  ADDRESS: 'OutletAddress'

```
### Usage
Running the Full Pipeline
Execute the main pipeline with:

``` bash
python script/main.py
```
This will:
- Load input and benchmark datasets
- Compute postcode neighborhoods using K-NN
- Perform record linkage (automatic + supervised matching)
- Save results to the output directory

## Methodology

The SVOC pipeline consists of four main stages:

1. **K-NN Clustering.**
- Groups postal codes by geographic proximity using K-Nearest Neighbors;
- Reduces search space by limiting comparisons to nearby locations;
- Uses haversine distance metric on latitude/longitude coordinates;
- Postal code neighborhoods can be cached for reuse.

2. **Feature Calculation.**
For each pair of records within the same cluster, computes several similarity scores.

3. **Automatic Matching.**
- Applies rule-based filters on computed features;
- Identifies high-confidence matches using predefined thresholds.

4. **Probabilistic Matching (Supervised).**
Applies trained machine learning models to unmatched records.
Three model types available:
- Logistic Regression;
- Support Vecto Machine;
- Naive Bayes Classifier.

For detailed methodology, see the [documentation](doc/doc.ipynb) notebook.

![Pipeline](doc/workflow.png)

## Output
The pipeline generates a CSV file containing:

- Top N matches per benchmark record (benchmark ID + input ID);
- Similarity scores for each feature;
- Match type (automatic vs supervised).

## Project Information

- Version: 0.1.0
- Repository: https://github.com/SDGGroup/HUKsvoc
- Python Version: >=3.12










