import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def WineKN():
  #Step 1: Load data
  data = pd.read_csv("WinePredictor.csv")
  
  #step 2: EDA ,Prepare and clean

  print("Fist five rows od dataset: \n",data.head())
  print("\nColumns of dataset is: ",data.columns.tolist())
  print("\nTypes of data in columns: ",data.dtypes)
  print("\nInfo of data is: \n",data.info())
  print("\nStatistics data is: \n",data.describe())
  print("\nCheck null values: \n",data.isnull().sum())
  
  
  # step 3: Analyse data
 
  for col in data.columns[1:]:
     plt.figure(figsize=(6,4))
     sns.histplot(data[col],kde=True)
     plt.title(f'Distribution of {col}')
     plt.show()
  
  sns.countplot(x= data["Class"])
  plt.title("Count of each class")
  plt.show()
 
  df = data.drop('Class',axis=1)
  plt.figure(figsize= (10,8))
  sns.heatmap(df[1:].corr(),annot=True,cmap="coolwarm")
  plt.title("Corelation matrix")
  plt.show()
  
  
  # step 4: Prepare data for model
  x = data.drop("Class",axis=1)
  y = data["Class"]
  
  scaler = StandardScaler()
  x_scaled = scaler.fit_transform(x)
  
  
  #Step 5: Split data
  x_train,x_test,y_train,y_test = train_test_split(x_scaled,y,test_size=0.3,random_state=42)
  
  list = [3,5,7,9]
  for i in list:
   # Step 6: Load algorithmic model
   model = KNeighborsClassifier(n_neighbors=i)

   # Step 7: Train dataset
   model.fit(x_train,y_train)

   #Step 8: Predict data
   Prediction = model.predict(x_test)
   print(f"\n when n_neighbors value is {i} Accuracy of model is: ",accuracy_score(y_test,Prediction))
   
   
def main():
   WineKN()

if __name__ =="__main__":
    main()