# Credit Card Fraud Detection using Hybrid Resampling & Neural Networks

## Project Overview
This project implements a Multi-Layered Perceptron (MLP) to detect fraudulent credit card transactions. Given the extreme class imbalance (approx. 0.17% fraud), the project utilizes a **Hybrid Sampling** approach and a **Binary Focal Loss** function to maximize financial recovery and minimize False Positive Rates (FPR).

## Dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

### Key Features
* **Hybrid Resampling:** Combines constrained SMOTE ($k=3$) with Random Undersampling (RUS) to preserve majority class structure while boosting minority signal.
* **Neural Network Architecture:** Multi-layered perceptron with L2 regularization and Adam optimizer.
* **Focal Loss:** Implemented `BinaryFocalCrossentropy` ($\alpha=0.75, \gamma=2.0$) to focus learning on "hard" misclassified examples.
* **Financial Metric:** Includes a custom "Financial Recovery" metric to evaluate the actual dollar amount saved versus total fraud exposure.

## Results
The model was optimized through experiments on batch sizes and layer depth.

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 99.93% |
| **AUPRC** | 0.7683 |
| **Precision** | 0.8485 |
| **FPR** | 0.0003 |
| **Financial Recovery** | **63%** |



## Methodology
1.  **Preprocessing:** Robust scaling of PCA components and stratified train-test splitting (70/30).
2.  **Imbalance Handling:**
    * SMOTE increases fraud instances to a 5% ratio.
    * Random Undersampling adjusts the final ratio to 10% to maintain a realistic distribution.
4.  **Optimization:** Conducted batch size trials (64, 128, 256) and architectural sweeps to find the most efficient gradient descent path.

## Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/Credit-Card-Fraud-Detection.git](https://github.com/your-username/Credit-Card-Fraud-Detection.git)
2. Install dependencies:
   pip install -r requirements.txt
3. Place creditcard.csv in the root directory and execute:
   python src/fraud_detection_model.py

## Author
  - Shriyans Sharma
