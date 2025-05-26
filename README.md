import pandas as pd

df = pd.read_csv("Titanic-Dataset.csv")  
df.head()df.info()            
df.describe()          
df.isnull().sum()df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['Fare'])   
df = df[df['Fare'] < 300]  
