'''This project involves loading, cleaning, analyzing, and visualizing data from a Netflix dataset. We'll use Python libraries like Pandas, Matplotlib, and Seaborn to work through the project.
   The goal is to explore the dataset, derive insights, and prepare for potential machine learning tasks'''

#importing libraries and loading dataframe
import pandas as pd

d=pd.read_csv("/content/netflix1.csv")
df=pd.DataFrame(d)
#display first few rows of dataset to know about the contents and attributes
print(df.head(10))

#Cleaning the Dataset
#1.Checking for missing values and dropping the rows with missing values
df.isnull().sum()
#There's no columns with missing values so we don't need to drop any rows

#2.Only check if movie Title and Show ID is duplicated and drop the rows with duplicates
df.duplicated(subset=["show_id"])
df.drop_duplicates(subset=["show_id"],inplace=True)
df.duplicated(subset=["title"])
df.drop_duplicates(subset=["title"],inplace=True)

#Convert the data type of date_added column to date time from object type
df['date_added']=pd.to_datatime(df['data_added'])
df.dtype #to check if data type of date_added has changed



