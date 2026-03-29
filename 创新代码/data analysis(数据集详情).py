# explore_dataset
import pandas as pd
df1 = pd.read_csv(r'E:\创新数据\heart.csv')  
df2 = pd.read_csv(r'E:\创新数据\Heart_Disease_Prediction.csv')  
df3 = pd.read_csv(r'E:\创新数据\framingham.csv') 
print("="*50)
print("1. heart.csv (Kaggle Heart Disease)")
print("="*50)
print(f"Shape: {df1.shape}")
print(f"Columns: {df1.columns.tolist()}")
print(f"Target distribution:\n{df1['HeartDisease'].value_counts()}")
print(f"Missing values: {df1.isnull().sum().sum()}")
print("\n")
print("="*50)
print("2. Heart_Disease_Prediction.csv (Cleveland)")
print("="*50)
print(f"Shape: {df2.shape}")
print(f"Columns: {df2.columns.tolist()}")
target_col = [col for col in df2.columns if 'Heart' in col][0]
print(f"Target column: '{target_col}'")
print(f"Target distribution:\n{df2[target_col].value_counts()}")
print(f"Missing values: {df2.isnull().sum().sum()}")
print("\n")
print("="*50)
print("3. framingham.csv (Framingham)")
print("="*50)
print(f"Shape: {df3.shape}")
print(f"Columns: {df3.columns.tolist()}")
print(f"Target distribution:\n{df3['TenYearCHD'].value_counts()}")
print(f"Missing values:\n{df3.isnull().sum()}")

#仅需清洗framingham
import pandas as pd
import numpy as np

df = pd.read_csv('framingham.csv')
print("清洗前的缺失值统计：")
print(df.isnull().sum())
edu_mode = df['education'].mode()[0]
df['education'].fillna(edu_mode, inplace=True)

cigs_median = df['cigsPerDay'].median()
df['cigsPerDay'].fillna(cigs_median, inplace=True)
df['BPMeds'].fillna(0, inplace=True)
chol_median = df['totChol'].median()
df['totChol'].fillna(chol_median, inplace=True)
bmi_median = df['BMI'].median()
df['BMI'].fillna(bmi_median, inplace=True)
hr_median = df['heartRate'].median()
df['heartRate'].fillna(hr_median, inplace=True)
print("\n清洗后的缺失值统计：")
print(df.isnull().sum())
df.to_csv('framingham_clean.csv', index=False)
print(f"\n清洗完成，保存到 framingham_clean.csv，共 {len(df)} 条")