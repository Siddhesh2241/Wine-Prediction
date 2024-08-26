# Data Analysis with K-Neighbors Algorithm

This Streamlit app allows users to upload a dataset, perform exploratory data analysis (EDA), and apply the K-Neighbors algorithm to classify the data. The app visualizes distributions of data features and the correlation matrix, providing an intuitive interface for understanding your data and model performance.

## Features

- **Upload CSV File**: Users can upload their dataset in CSV format.
- **Exploratory Data Analysis (EDA)**:
  - Display the first five rows of the dataset.
  - View dataset columns and their data types.
  - Summary statistics of the data.
  - Check for null values in the dataset.
- **Data Visualization**:
  - Histogram plots of each feature.
  - Count plot of the target class.
  - Correlation matrix heatmap.
- **Modeling**:
  - Apply the K-Neighbors algorithm for classification.
  - Display accuracy for different values of `n_neighbors`.
  **Uploading Data**:
   - Click on the "Browse files" button or drag and drop your CSV file.
   - The app will automatically display the data and perform the initial analysis.
  **Exploratory Data Analysis**:
   - The app displays the first few rows, column types, and summary statistics.
   - You can view histograms of all features and the count of each class.
  **Model Training and Evaluation**:
   - The app uses the K-Neighbors algorithm to classify the data.
   - The accuracy for different n_neighbors values is displayed.
