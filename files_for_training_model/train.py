# -*- coding: utf-8 -*-
"""Predictive_Lab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15izSV3TvIOEJ2thuzGqAHCwtyWynuqG3

## Aditya Gupta, FT-251006, Predictive Lab Assignment 1

## Objective: To predict laptop prices using multiple parameters with the help of ML model.

## Data Source:https://github.com/ShreyaPatil1199/Laptop-Price-Predictor/blob/main/laptop_data.csv

## About Meta Data
1.	Company: This categorical feature represents the laptop manufacturer. It includes renowned brands in the tech industry, influencing pricing based on brand reputation, quality, and market positioning.

2.	TypeName: The laptop's type or category is represented by this categorical feature. It categorizes laptops into various types, such as Notebooks, Ultrabooks, and Gaming laptops, influencing pricing based on the target audience and specific use cases.

3.	Inches: This numerical feature denotes the screen size of the laptop in inches. Larger screens may demand higher prices, often appealing to users seeking enhanced visual experiences.

4.	ScreenResolution: Representing the screen resolution of the laptop, this categorical feature provides insights into display quality. Higher resolutions may contribute to increased prices due to improved visual clarity and user experience.

5.	Cpu: The Central Processing Unit (CPU) of the laptop is captured by this categorical feature. It encompasses various processor types and specifications, with higher-performance CPUs typically resulting in higher laptop prices.

6.	Ram: This categorical feature denotes the Random Access Memory (RAM) capacity of the laptop. RAM size affects the laptop's multitasking capabilities and overall performance, influencing pricing accordingly.

7.	Memory: Representing the Hard Disk or Solid-State Drive (HDD/SSD) memory, this categorical feature contributes to the laptop's storage capacity. Larger storage capacities may lead to higher prices, accommodating user data and software requirements.

8.	GPU: The Graphics Processing Unit (GPU) is captured by this categorical feature. It encompasses different GPU configurations, including integrated and dedicated graphics, impacting the laptop's performance in graphics-intensive tasks and potentially affecting pricing.

9.	OpSys: This categorical feature represents the laptop's Operating System. Different operating systems can influence user preferences and software compatibility, thereby impacting the laptop's perceived value and pricing.

10.	Weight: The weight of the laptop is captured by this numerical feature. Lighter laptops may be more desirable for portability, potentially influencing pricing based on user preferences.

11.	Price_euros: This is the target variable of the prediction model. Representing the laptop's price in Euros, this numerical feature is what the prediction model aims to accurately predict based on the other features.
"""

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
import scipy.stats as stats
from statsmodels.stats import weightstats as stests
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_rows', None)

"""# Printing library versions"""

# Versions:

print("Pandas version :" , pd.__version__)
print("Numpy version: ", np.__version__)
print("Seaborn version: ", sns.__version__)
import matplotlib
print("Matplotlib version:", matplotlib.__version__)
from sklearn import __version__
print("scikitlearn version:",__version__)
from scipy import __version__
print("Scipy version: ",__version__)
import statsmodels
print("Statsmodels version: ",statsmodels.__version__)
from platform import python_version
print("Python version: ",python_version())

#from google.colab import drive
#drive.mount('/content/drive')

#!ls drive/MyDrive/'Colab Notebooks'/'Predictive Lab 2'

"""# Storing data in a data frame"""

laptop=pd.read_csv('laptop_data.csv')
# Loading csv data into a python data frame

"""# Descriptive Statistics"""

laptop.describe().transpose()

laptop.info()

"""## Removing Column 1 as it represents only index and is of no use"""

laptop = laptop.iloc[:, 1:]
laptop.head()

"""## Dropping ScreenResolution as it is usually not the most critical factor in laptop purchase compared to features like screen size and weight."""

laptop.drop(columns=['ScreenResolution'],inplace=True)
laptop.head()

"""## Finding Null values"""

# prompt: find null values in laptop dataframe

# Finding Null values
laptop.isnull().sum()

"""## Replacing null values in OpSys with unknown value"""

# prompt: replace null values with unknown in OpSys column

# Replace null values in 'OpSys' column with 'Unknown'
laptop['OpSys'].fillna('Unknown', inplace=True)

laptop.isnull().sum()

"""## Merging GPUs values to keep the understanding of GPUs easy"""

laptop['Gpu'] = laptop['Gpu'].apply(
    lambda x: 'Nvidia' if x.startswith('Nvidia')
    else 'Intel' if x.startswith('Intel')
    else 'AMD' if x.startswith('AMD')
    else x
)

# Check the updated column
print(laptop['Gpu'].value_counts())

print(laptop['Gpu'].value_counts())

"""## Count of Opsys"""

OpSys_counts = laptop['OpSys'].value_counts()

# Print the counts
OpSys_counts

"""## Merging ScreenResolution values to keep the understanding of Screenresolution Easy"""

# laptop['ScreenResolution'] = laptop['ScreenResolution'].apply(
#     lambda x: 'IPS' if x.startswith('IPS')
#     else 'Touchscreen' if x.startswith('Touchscreen')
#     else '4K' if x.startswith('4K')
#     else 'Quad' if x.startswith('Quad')
#     else 'Full HD' if x.startswith('Full HD')
#     else x
# )

# # Check the updated column
# print(laptop['ScreenResolution'].value_counts())

TypeName_counts = laptop['TypeName'].value_counts()

# Print the counts
TypeName_counts

"""## Merging Memory values to keep the understanding of memory easy"""

def consolidate_storage(value):
    if 'SSD' in value and 'HDD' in value:
        return 'Mixed (SSD + HDD)'
    elif 'SSD' in value:
        return 'SSD'
    elif 'HDD' in value:
        return 'HDD'
    elif 'Flash Storage' in value:
        return 'Flash Storage'
    elif 'Hybrid' in value:
        return 'Hybrid'
    else:
        return 'Other'

# Apply the grouping logic
laptop['Memory'] = laptop['Memory'].apply(consolidate_storage)

# View consolidated counts
storage_counts = laptop['Memory'].value_counts()
print(storage_counts)

"""## Merging CPU values to keep the understanding of CPU easy"""

def consolidate_processor(value):
    value = value.lower()
    if 'core i3' in value:
        return 'Intel Core i3'
    elif 'core i5' in value:
        return 'Intel Core i5'
    elif 'core i7' in value:
        return 'Intel Core i7'
    elif 'celeron' in value:
        return 'Intel Celeron'
    elif 'pentium' in value:
        return 'Intel Pentium'
    elif 'xeon' in value:
        return 'Intel Xeon'
    elif 'ryzen' in value:
        return 'AMD Ryzen'
    elif 'a-series' in value or 'a6' in value or 'a9' in value:
        return 'AMD A-Series'
    elif 'e-series' in value:
        return 'AMD E-Series'
    else:
        return 'Other'

# Apply the grouping logic
laptop['Cpu'] = laptop['Cpu'].apply(consolidate_processor)

# View consolidated counts
processor_counts = laptop['Cpu'].value_counts()
print(processor_counts)

def consolidate_broad_category(value):
    if value in ['Intel Core i7', 'Intel Xeon', 'AMD Ryzen']:
        return 'High-Performance'
    elif value in ['Intel Core i5', 'AMD A-Series']:
        return 'Mid-Range'
    elif value in ['Intel Core i3', 'Intel Celeron', 'Intel Pentium', 'AMD E-Series']:
        return 'Entry-Level'
    else:
        return 'Other'

# Apply the grouping logic
laptop['Cpu'] = laptop['Cpu'].apply(consolidate_broad_category)

# View consolidated counts
broad_category_counts = laptop['Cpu'].value_counts()
print(broad_category_counts)

Company_counts = laptop['Company'].value_counts()

# Print the counts
Company_counts

"""## Merging Companies values to keep the understanding of companies easy"""

Company_counts = laptop['Company'].value_counts()

# Print the counts
Company_counts

top_5_companies = Company_counts.nlargest(5).index.tolist()

def consolidate_companies(company):
  if company not in top_5_companies:
    return 'Other'
  return company

laptop['Company'] = laptop['Company'].apply(consolidate_companies)
print(laptop['Company'].value_counts())

"""## Initiating dummy coding for object data types"""

# Identify categorical columns
categorical_cols = laptop.select_dtypes(include=['object']).columns

# Perform one-hot encoding on categorical columns
laptop = pd.get_dummies(laptop, columns=categorical_cols, drop_first=True)

# Display the first few rows of the encoded DataFrame
print(laptop.head())

"""## Replacing True, False values with 1 and 0"""

# Replace True/False with 1/0
laptop = laptop.replace({True: 1, False: 0})

print(laptop.head())

laptop.info()

"""## Converting datatypes to int"""

# Convert object and float values to int
for col in laptop.columns:
  if laptop[col].dtype == 'object':
    try:
      laptop[col] = laptop[col].astype(int)
    except ValueError:
      print(f"Column {col} cannot be converted to int directly.")

  elif laptop[col].dtype == 'float64':
    laptop[col] = laptop[col].astype(int)
#check the info after conversion
laptop.info()

"""## Standardizing the data"""

# Standardize the data
scaler = StandardScaler()
laptop_standardized = scaler.fit_transform(laptop)

# Create a new DataFrame with standardized values
laptop = pd.DataFrame(laptop_standardized, columns=laptop.columns)

# Display the first few rows of the standardized DataFrame
print(laptop.head())

"""## Generating heatmap"""

f, ax = plt.subplots(figsize=(30, 30))
corr=laptop.corr('pearson')
cp=sns.heatmap(corr,mask=np.zeros_like(corr,dtype=bool),cmap=sns.diverging_palette(200,10,as_cmap=True),square=True,ax=ax,annot=True)
bottom,top = cp.get_ylim()
cp.set_ylim(bottom+0.5,top-0.5) #matplotlib version to 3.1.1 has a bug in the plot and mistakes in the y-axis

"""### We don't see very high correlations in the heatmap so we can keep all the generated features as it is and not exclude any feature.

## Modeling Random Forest model for the processed data.
"""

lap=laptop.copy()

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load your dataset (replace 'lap' with your actual dataset variable)
# Assuming 'lap' is a DataFrame with features and target
X = lap.drop(columns=['Price'])  # Features (remove target column)
y = lap['Price']  # Target variable

# Split the data into training and test sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter distribution for RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]  # Replaced 'auto' with valid values
}

# Initialize the RandomForestRegressor model
rf_model = RandomForestRegressor(random_state=42)

# Initialize RandomizedSearchCV with reduced runtime settings
random_search = RandomizedSearchCV(
    estimator=rf_model,
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter combinations to test
    cv=3,  # 3-fold cross-validation
    n_jobs=-1,  # Use all available processors
    verbose=2,
    random_state=42
)

# Fit the model with RandomizedSearchCV
random_search.fit(X_train, y_train)

# Retrieve the best parameters and the best model
print(f"Best Parameters: {random_search.best_params_}")
best_rf_model = random_search.best_estimator_

# Make predictions on the test set using the optimized model
y_pred = best_rf_model.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)  # Mean Squared Error
r2 = r2_score(y_test, y_pred)  # R-squared score

# Output the results
print(f"Optimized Mean Squared Error: {mse:.2f}")
print(f"Optimized R-squared: {r2:.2f}")

"""The best R - square value is 0.76 and MSE is 0.27."""

import pickle

pickle.dump(best_rf_model, open('model.pkl','wb'))

best_rf_model = pickle.load(open('model.pkl','rb'))