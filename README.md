# 0XVENOM - Network Intrusion Detection System (NIDS)

🐍 **A powerful command-line Network Intrusion Detection System using Machine Learning**

## Features

- **Multi-Algorithm Detection**: Uses Logistic Regression, K-Nearest Neighbors, and Decision Tree algorithms
- **Automated Feature Selection**: Intelligent feature selection using Random Forest and RFE
- **Cross-Validation**: Robust model validation with 5-fold cross-validation
- **Detailed Analytics**: Comprehensive performance metrics and confusion matrices
- **Colorful CLI Interface**: Beautiful command-line interface with colored output
- **Easy to Use**: Simple command-line arguments for quick analysis

## Installation

1. **Clone or download the project files**

2. **Install dependencies:**
   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

3. **Make the launcher executable:**
   \`\`\`bash
   chmod +x 0xvenom
   \`\`\`

## Usage

### Basic Usage
\`\`\`bash
./0xvenom --train path/to/train.csv --test path/to/test.csv
\`\`\`

### Detailed Analysis
\`\`\`bash
./0xvenom --train path/to/train.csv --test path/to/test.csv --detailed
\`\`\`

### Direct Python Execution
\`\`\`bash
python3 scripts/0xvenom.py --train data/train.csv --test data/test.csv
\`\`\`

## Command Line Options

- `--train`: Path to training dataset (CSV file) - **Required**
- `--test`: Path to testing dataset (CSV file) - **Required**
- `--detailed`: Show detailed evaluation with confusion matrices - **Optional**
- `--version`: Show version information
- `--help`: Show help message

## Output

The system provides:

1. **Data Overview**: Dataset statistics and class distribution
2. **Feature Selection**: Automatically selected important features
3. **Model Performance**: Accuracy scores for all models
4. **Cross-Validation Results**: Precision and recall with confidence intervals
5. **Best Model Recommendation**: Identifies the top-performing model
6. **Detailed Analysis** (optional): Confusion matrices and classification reports

## Example Output

\`\`\`
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██████╗ ██╗  ██╗██╗   ██╗███████╗███╗   ██╗ ██████╗ ███╗   ███╗          ║
║   ██╔═████╗╚██╗██╔╝██║   ██║██╔════╝████╗  ██║██╔═══██╗████╗ ████║          ║
║   ██║██╔██║ ╚███╔╝ ██║   ██║█████╗  ██╔██╗ ██║██║   ██║██╔████╔██║          ║
║   ████╔╝██║ ██╔██╗ ╚██╗ ██╔╝██╔══╝  ██║╚██╗██║██║   ██║██║╚██╔╝██║          ║
║   ╚██████╔╝██╔╝ ██╗ ╚████╔╝ ███████╗██║ ╚████║╚██████╔╝██║ ╚═╝ ██║          ║
║    ╚═════╝ ╚═╝  ╚═╝  ╚═══╝  ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝          ║
║                                                                              ║
║              Network Intrusion Detection System (NIDS)                       ║
║          Detects malicious and unauthorized network access                   ║
║                      Author: Adarsh Kumar Singh                              ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

[INFO] Training data shape: (125973, 42)
[INFO] Testing data shape: (22544, 41)
[SUCCESS] Selected features: src_bytes, dst_bytes, count, srv_count, ...
[SUCCESS] Best Model: Decision Tree (Accuracy: 0.9987)
\`\`\`

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- numpy
- seaborn
- matplotlib
- tabulate
- colorama

## Author

**Adarsh Kumar Singh**

## License

This project is open source and available under the MIT License.
