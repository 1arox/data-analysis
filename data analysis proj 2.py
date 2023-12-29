import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# DATA LOADING
# Loading data from an Excel file into a pandas DataFrame
df = pd.read_excel('C:/Users/aribe/Downloads/Documents/ABUSALOUT2023.xlsx')

# DATA PREPROCESSING 
# Identifying categorical and numeric columns in the DataFrame
categorical_columns = df.select_dtypes(include=['object']).columns
numeric_columns = df.select_dtypes(include=[np.number]).columns

# Filling missing values in numeric columns with the column mean 
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Applying one-hot encoding to categorical columns to convert them into a machine-readable format
df_encoded = pd.get_dummies(df[categorical_columns])
# Combine the one-hot encoded categorical data with the numeric data
df_combined = pd.concat([df[numeric_columns], df_encoded], axis=1)

# FEATURE SCALING 
# Data Transformation on numeric data only
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_combined[numeric_columns])

# DIMENSIONALITY REDUCTION 
# Applying PCA for DIMENIONALITY REDUCTION to reduce the number of random variables under consideration 
pca = PCA(n_components=3)
principal_components = pca.fit_transform(scaled_data)

# MACHINE LEARNING - CLUSTERING
# Applying K-Means clustering with explicit n_init for specifying the number of initializations to run
kmeans = KMeans(n_clusters=3, n_init=10)
clusters = kmeans.fit_predict(scaled_data)

# Adding the clusters back to the DataFrame
df_combined['cluster'] = clusters

# DATA VISUALIZATION  
# Visualizing the distribution of values in one of the numeric columns using a histogram
sns.histplot(df_combined['Automated teller machines (ATMs) (per 100,000 adults)'])
plt.show()

# Displaying the correlation matrix for all numeric columns as a heatmap
correlation_matrix = df_combined[numeric_columns].corr()
sns.heatmap(correlation_matrix, annot=False)
plt.show()

# Calculating and printing the correlation between two specific any columns
correlation_value = df_combined['Adults (ages 15+) and children (ages 0-14) newly infected with HIV'].corr(df_combined['Age dependency ratio (% of working-age population)'])
print("Correlation between Column1 and Column2:", correlation_value)

# OUTPUT
# Printing the df_combined DataFrame to verify the results
print(df_combined.head())
