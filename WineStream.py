import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Data Analysis with K-Neighbors Algorithm",
                   page_icon=":bottle_with_popping_cork:",
                   layout="wide")

def WineKN():
    st.title("Wine Data Analysis with K-Neighbors Algorithm")
    st.markdown("This app allows you to upload a CSV file and analyze the data using the K-Neighbors algorithm.")

    # Sidebar for user inputs
    st.sidebar.header("User Input Features")
    n_neighbors = st.sidebar.selectbox("Number of Neighbors (k):", options=[3, 5, 7, 9], index=0)

    # Step 1: Load data
    st.subheader("Upload CSV file :file_folder:")
    upload_file = st.file_uploader("Upload File", type="csv")

    if upload_file is not None:
        data = pd.read_csv(upload_file)
        st.dataframe(data)

        # Step 2: Exploratory Data Analysis (EDA)
        st.subheader("Dataset Overview")
        st.write("First five rows of the dataset:")
        st.write(data.head())

        st.write("Columns of the dataset:")
        st.write(data.columns.tolist())

        st.write("Data types of columns:")
        st.write(data.dtypes)

        st.write("Statistics summary:")
        st.write(data.describe())

        st.write("Check for missing values:")
        st.write(data.isnull().sum())

        # Step 3: Data Analysis
        st.subheader("Data Analysis")
        st.markdown("### Distributions of Features")
        for col in data.columns[1:]:
            fig, ax = plt.subplots(figsize=(3, 2))
            sns.histplot(data[col], kde=True, ax=ax)
            ax.set_title(f'Distribution of {col}', fontsize=10)  
            ax.set_xlabel(ax.get_xlabel(), fontsize=8)  
            ax.set_ylabel(ax.get_ylabel(), fontsize=8)
            ax.tick_params(axis='both', which='major', labelsize=6)
            st.pyplot(fig)
            plt.clf()

        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.countplot(x=data["Class"], ax=ax)
        ax.set_title("Count of Each Class", fontsize=10)
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)  
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=6)
        st.pyplot(fig)
        plt.clf()

        st.markdown("### Correlation Matrix")
        df = data.drop('Class', axis=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix", fontsize=12)
        st.pyplot(fig)

        # Step 4: Prepare Data for Modeling
        st.subheader("Modeling")
        st.write("Preparing data for modeling...")

        X = data.drop("Class", axis=1)
        y = data["Class"]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Step 5: Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        # Step 6: Train the K-Neighbors Classifier
        st.write(f"Training K-Neighbors Classifier with k={n_neighbors}...")
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)

        # Step 7: Prediction and Accuracy
        st.write("Predicting on test data...")
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        st.success(f"Accuracy of the model with k={n_neighbors}: {accuracy:.2f}")

    else:
        st.warning("Please upload a CSV file to proceed.")

def main():
    WineKN()

if __name__ == "__main__":
    main()
