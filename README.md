# Hospital Readmission Prediction Project

## Project Title
Hospital Readmission Prediction Project

## Problem Statement
- Hospital readmissions are costly and can indicate poor patient care.
- Predicting which patients are likely to be readmitted helps hospitals improve care, reduce costs, and allocate resources efficiently.
- Traditional methods are manual and prone to errors, so an automated machine learning approach is necessary.

## Objective
- Build a predictive model using patient medical data to identify patients likely to be readmitted.
- Reduce unnecessary hospital readmissions.
- Support hospitals in improving patient care quality.

## Dataset Description
- **Dataset Name:** Diabetes 130-US Hospitals Dataset  
- **Source:** [GitHub Repository](https://github.com/jonneff/Diabetes2)  
- **Features:** Includes patient demographics, lab tests, medications, admission information, and previous medical history.  
- **Target Class:** Readmission (Yes / No)  
- **Instances:** Approximately 100,000 patient records

## Methodology / Approach
1. Load and explore the dataset.
2. Handle missing values and encode categorical variables.
3. Split data into training and testing sets.
4. Train a Random Forest Classifier model.
5. Evaluate the model using accuracy, precision, recall, and F1-score.
6. Identify high-risk patients based on predictions.

## Tools & Technologies Used
- **Programming Language:** Python  
- **Libraries:** pandas, numpy, scikit-learn, matplotlib, seaborn  
- **IDE:** VS Code  
- **Version Control:** GitHub  

## Steps to Run the Project
1. Clone the repository or download the project files.
2. Make sure `main.py` and `data.csv` are in the same folder.
3. Open terminal in VS Code.
4. Install required libraries:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
5. Run the project:
   python main.py
6. The program will display:
   -First 5 rows of the dataset.
   -Detected features and target column.
   -Model accuracy and classification report.

## Results / Output
- **Data preview** of first few rows.
- **Accuracy** of the Random Forest model.
- **Classification report** showing precision, recall, and F1-score.
- **Identified high-risk patients** for readmission.

## Notes
- **Ensure Python 3.x** is installed on your system.
- **No trained model file** is required; the model trains every time you run `main.py`.
- All work is **original and plagiarism-free**.
