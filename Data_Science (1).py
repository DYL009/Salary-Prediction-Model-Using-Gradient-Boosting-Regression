#!/usr/bin/env python
# coding: utf-8

# In[482]:


import pandas as pd
import numpy as np
# Libraries for visualization
import seaborn as sns
import matplotlib.pyplot as plt


# # Understanding The Dataset

# In[483]:


ds=pd.read_csv('data_science_job.csv')
ds.head(5)
ds


# In[484]:


ds.columns


# In[485]:


ds.isnull().sum()


# In[486]:


ds.shape


# In[487]:


ds.info()


# In[488]:


ds.describe()


# In[489]:


ds.duplicated().sum()


# In[490]:


ds.nunique()


# In[491]:


ds.experience_level.unique()


# In[492]:


ds.work_year.value_counts()


# In[493]:


ds.employment_type.unique()


# In[494]:


ds.employee_residence.unique()


# In[495]:


ds.company_location.unique()


# In[496]:


ds.company_size.unique()


# # Data Cleaning

# In[497]:


missing_percentage=ds.isnull().mean()*100

print(missing_percentage)


# In[498]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(ds.isnull(), cmap='viridis')
plt.title("Missing Values Heatmap")
plt.show()


# In[499]:


ds=ds.dropna()


# In[500]:


ds.shape


# In[501]:


ds.tail()


# In[502]:


ds.sample(5)


# In[503]:


ds.isnull().sum()


# In[504]:


ds['experience_level']=ds['experience_level'].replace({'EN': 'Entry-Level',
    'MI': 'Mid-Level',
    'SE': 'Senior-Level',
    'EX': 'Executive'})


# In[505]:


ds['experience_level']


# In[506]:


ds['job_title']


# In[507]:


# Extract Job Title: Everything before 'in office', 'remote', or '(Remote)'
ds.loc[:, 'Job Title'] = ds['job_title'].str.extract(r'^(.*?)(?:\s+(in office|remote|\(Remote\))|$)', expand=True)[0]

# Extract Work Location: Capture 'in office' or 'remote'
ds.loc[:, 'Work Location'] = ds['job_title'].str.extract(r'(in office|remote|\(Remote\))', expand=True)[0]

# If Work Location is missing, set as 'Unknown'
ds.loc[:, 'Work Location'] = ds['Work Location'].fillna('Unknown')

# Clean up extra spaces
ds.loc[:, 'Job Title'] = ds['Job Title'].str.strip()
ds.loc[:, 'Work Location'] = ds['Work Location'].str.strip()

# # Drop the original column if not needed
# ds = ds.drop(columns=['jobtitle'])

# Display the result
print(ds)



# In[508]:


ds['Job Title'].str.lower()


# In[509]:


ds['Work Location'].str.lower()


# In[510]:


# Check rows with negative salaries
negative_salaries = ds[ds['salary'] < 0]
print(f"Negative salaries found: {len(negative_salaries)}")

# Option 1: Remove rows with negative salaries
ds = ds[ds['salary'] >= 0]

# Option 2: Replace negative salaries with NaN or the mean salary
# ds.loc[ds['salary'] < 0, 'salary'] = ds['salary'].mean()

# Verify the changes
ds.describe()


# In[511]:


# Define a function to detect and remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Apply the function to salary columns
ds = remove_outliers(ds, 'salary')
ds = remove_outliers(ds, 'salary_in_usd')

# Verify the changes
ds.describe()


# In[512]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group data by work_year and calculate mean salary
yearly_salary = ds.groupby('work_year')['salary_in_usd'].mean().reset_index()

# Plot the trends
plt.figure(figsize=(8, 5))
sns.barplot(x='work_year', y='salary_in_usd', data=yearly_salary, palette='viridis')


# In[513]:


# Check correlation between salary and salary_in_usd
correlation = ds[['salary', 'salary_in_usd']].corr()
print(correlation)

# Plot salary vs. salary_in_usd
plt.figure(figsize=(8, 5))
sns.scatterplot(x='salary', y='salary_in_usd', data=ds, alpha=0.7)
plt.title('Salary vs Salary in USD', fontsize=14)
plt.xlabel('Salary')
plt.ylabel('Salary (in USD)')
plt.show()


# In[514]:


ds['conversion_rate'] = ds['salary_in_usd'] / ds['salary']
print(ds['conversion_rate'].describe())


# In[515]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(x='salary', y='salary_in_usd', data=ds, alpha=0.7)
plt.title('Salary vs. Salary in USD', fontsize=14)
plt.xlabel('Salary')
plt.ylabel('Salary (in USD)')
plt.show()


# In[516]:


low_outliers = ds[ds['conversion_rate'] < 0.5]
high_outliers = ds[ds['conversion_rate'] > 3]
print(f"Low Outliers: {len(low_outliers)}, High Outliers: {len(high_outliers)}")


# In[517]:


ds = ds[(ds['conversion_rate'] >= 0.5) & (ds['conversion_rate'] <= 3)]


# In[518]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.histplot(ds['conversion_rate'], bins=30, kde=True)
plt.title('Distribution of Conversion Rates')
plt.xlabel('Conversion Rate')
plt.ylabel('Frequency')
plt.show()


# In[519]:


correlation = ds[['salary', 'salary_in_usd']].corr()
print(correlation)


# In[520]:


# Analyze low outliers
print("Low Outliers Summary:")
print(low_outliers.describe())

# Analyze high outliers
print("High Outliers Summary:")
print(high_outliers.describe())


# In[521]:


# Filter the dataset to keep only valid conversion rates
ds = ds[(ds['conversion_rate'] >= 0.5) & (ds['conversion_rate'] <= 3)]
print(f"Dataset size after removing outliers: {len(ds)}")


# In[522]:


# Replace low outliers with the median of valid conversion rates
median_conversion_rate = ds[(ds['conversion_rate'] >= 0.5) & (ds['conversion_rate'] <= 3)]['conversion_rate'].median()
ds.loc[ds['conversion_rate'] < 0.5, 'conversion_rate'] = median_conversion_rate
ds.loc[ds['conversion_rate'] > 3, 'conversion_rate'] = median_conversion_rate

# Recalculate salary_in_usd after imputation
ds['salary_in_usd'] = ds['salary'] * ds['conversion_rate']


# In[523]:


# Recalculate correlation
correlation = ds[['salary', 'salary_in_usd']].corr()
print("Updated Correlation Matrix:")
print(correlation)


# In[524]:


# Scatter plot after cleaning
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(x='salary', y='salary_in_usd', data=ds, alpha=0.7)
plt.title('Salary vs. Salary in USD (After Cleaning)', fontsize=14)
plt.xlabel('Salary')
plt.ylabel('Salary (in USD)')
plt.show()

# Histogram of conversion rates
plt.figure(figsize=(8, 5))
sns.histplot(ds['conversion_rate'], bins=30, kde=True)
plt.title('Distribution of Conversion Rates (After Cleaning)')
plt.xlabel('Conversion Rate')
plt.ylabel('Frequency')
plt.show()


# In[525]:


print(ds.describe())


# In[526]:


import seaborn as sns
import matplotlib.pyplot as plt

# Boxplot for salary and salary_in_usd
plt.figure(figsize=(10, 6))
sns.boxplot(data=ds[['salary', 'salary_in_usd']], palette="Set3")
plt.title('Salary Distributions')
plt.ylabel('Value')
plt.show()

# Histogram for conversion rates
plt.figure(figsize=(8, 5))
sns.histplot(ds['conversion_rate'], bins=30, kde=True, color='blue')
plt.title('Distribution of Conversion Rates')
plt.xlabel('Conversion Rate')
plt.ylabel('Frequency')
plt.show()


# In[527]:


sns.lineplot(data=ds, x='work_year', y='salary_in_usd', ci=None)
plt.title('Salary in USD Over the Years')
plt.xlabel('Work Year')
plt.ylabel('Average Salary (in USD)')
plt.show()


# In[528]:


ds.head(5)


# In[529]:


ds.columns


# In[530]:


ds=ds.drop(columns=['job_title','Work Location'])


# In[531]:


ds['employment_type'].unique()


# In[532]:


rename_dict={'CT': 'Contract',
    'FL': 'Full-time',
    'FT': 'Freelance',
    'PT': 'Part-time'}

ds['employment_type']=ds['employment_type'].replace(rename_dict)


# In[533]:


ds.head()


# In[534]:


ds['company_location'].unique()


# In[535]:


# Define the mapping dictionary for country codes
country_dict = {
    'DE': 'Germany',
    'IN': 'India',
    'CN': 'China',
    'MX': 'Mexico',
    'US': 'United States',
    'JP': 'Japan',
    'UK': 'United Kingdom'
}

# Replace the country codes in the relevant column (assume the column is 'country_code')
ds['company_location'] = ds['company_location'].replace(country_dict)

# Check the updated dataset
print(ds.head())


# In[536]:


ds.head()


# In[537]:


ds['company_size'].unique()


# In[538]:


# Define the mapping dictionary for size categories
size_dict = {
    'L': 'Large',
    'M': 'Medium',
    'S': 'Small'
}

# Replace the size abbreviations in the relevant column (assume the column is 'size')
ds['company_size'] = ds['company_size'].replace(size_dict)




# In[539]:


ds.head()


# In[540]:


ds.info()


# In[541]:


import seaborn as sns
import matplotlib.pyplot as plt

# Salary distribution plot
plt.figure(figsize=(10, 6))
sns.histplot(ds['salary_in_usd'], bins=30, kde=True, color='blue')
plt.title('Salary Distribution in USD')
plt.xlabel('Salary (USD)')
plt.ylabel('Frequency')
plt.show()


# In[542]:


# Salary by experience level
plt.figure(figsize=(10, 6))
sns.boxplot(data=ds, x='experience_level', y='salary_in_usd', palette='Set3')
plt.title('Salary Distribution by Experience Level')
plt.xlabel('Experience Level')
plt.ylabel('Salary (USD)')
plt.show()


# In[543]:


# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(ds[['salary', 'salary_in_usd', 'conversion_rate']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[544]:


ds.head()


# In[545]:


ds.columns


# In[546]:


# Drop unnecessary columns
columns_to_drop = ['salary_currency', 'employee_residence', 'company_location', 'Job Title']
ds = ds.drop(columns=columns_to_drop)



# In[547]:


ds.head(5)


# In[548]:


ds.columns


# In[549]:


ds.columns = ds.columns.str.lower()


# In[550]:


ds.head()


# In[551]:


ds = ds.drop(columns=['salary'])


# In[552]:


ds.info()


# In[553]:


ds.describe()


# In[554]:


ds.head(5)


# In[555]:


# Check for missing values in each column
missing_values = ds.isnull().sum()
print("Missing Values:\n", missing_values)


# In[556]:


# Check data types of each column
data_types = ds.dtypes
print("Data Types:\n", data_types)


# In[557]:


# One-Hot Encoding for categorical columns
ds_encoded = pd.get_dummies(ds, columns=['job_category', 'employment_type', 'work_setting', 'company_size'], drop_first=True)
print(ds_encoded.head())


# In[558]:


from sklearn.preprocessing import LabelEncoder

# Label Encoding for experience_level (if ordinal)
label_encoder = LabelEncoder()
ds['experience_level'] = label_encoder.fit_transform(ds['experience_level'])
print(ds['experience_level'].head())


# In[559]:


from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Apply scaling to numeric features
ds[['work_year', 'salary_in_usd', 'conversion_rate']] = scaler.fit_transform(ds[['work_year', 'salary_in_usd', 'conversion_rate']])

# Check the scaled values
print(ds[['work_year', 'salary_in_usd', 'conversion_rate']].head())


# In[560]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the scaler
min_max_scaler = MinMaxScaler()

# Apply Min-Max scaling to numeric features
ds[['work_year', 'salary_in_usd', 'conversion_rate']] = min_max_scaler.fit_transform(ds[['work_year', 'salary_in_usd', 'conversion_rate']])

# Check the scaled values
print(ds[['work_year', 'salary_in_usd', 'conversion_rate']].head())


# In[561]:


# Check correlation between numeric columns
correlation_matrix = ds[['work_year', 'salary_in_usd', 'conversion_rate']].corr()
print("Correlation Matrix:\n", correlation_matrix)


# In[562]:


from sklearn.model_selection import train_test_split

# Define features (X) and target variable (y)
X = ds.drop('salary_in_usd', axis=1)  # Features
y = ds['salary_in_usd']  # Target variable

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the training and test sets
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


# In[563]:


import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Check for missing values
missing_values = ds.isnull().sum()
print("Missing Values:\n", missing_values)

# 2. Check data types
data_types = ds.dtypes
print("Data Types:\n", data_types)

# 3. Encode categorical variables
# One-Hot Encoding for categorical columns
ds_encoded = pd.get_dummies(ds, columns=['job_category', 'employment_type', 'work_setting', 'company_size'], drop_first=True)

# Label Encoding for experience_level (if ordinal)
label_encoder = LabelEncoder()
ds_encoded['experience_level'] = label_encoder.fit_transform(ds_encoded['experience_level'])

# 4. Feature Scaling
# Standardization (Z-score scaling)
scaler = StandardScaler()
ds_encoded[['work_year', 'salary_in_usd', 'conversion_rate']] = scaler.fit_transform(ds_encoded[['work_year', 'salary_in_usd', 'conversion_rate']])

# Or, for Min-Max Scaling
# min_max_scaler = MinMaxScaler()
# ds_encoded[['work_year', 'salary_in_usd', 'conversion_rate']] = min_max_scaler.fit_transform(ds_encoded[['work_year', 'salary_in_usd', 'conversion_rate']])

# 5. Check correlation
correlation_matrix = ds_encoded[['work_year', 'salary_in_usd', 'conversion_rate']].corr()
print("Correlation Matrix:\n", correlation_matrix)

# 6. Data splitting
X = ds_encoded.drop('salary_in_usd', axis=1)  # Features
y = ds_encoded['salary_in_usd']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")


# In[564]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")


# In[565]:


import matplotlib.pyplot as plt

# Plotting residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()


# In[566]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - MSE: {mse_rf}")
print(f"Random Forest - R-squared: {r2_rf}")


# In[567]:


import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the XGBoost Regressor
xgb_model = xgb.XGBRegressor(n_estimators=100, random_state=42)

# Train the model
xgb_model.fit(X_train, y_train)

# Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost - MSE: {mse_xgb}")
print(f"XGBoost - R-squared: {r2_xgb}")


# In[568]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Predictions
y_pred_gb = gb_model.predict(X_test)

# Evaluate the model
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

print(f"Gradient Boosting - MSE: {mse_gb}")
print(f"Gradient Boosting - R-squared: {r2_gb}")


# In[569]:


import matplotlib.pyplot as plt

# For Gradient Boosting predictions
residuals_gb = y_test - y_pred_gb
plt.scatter(y_pred_gb, residuals_gb)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values - Gradient Boosting')
plt.show()


# In[570]:


from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(gb_model, X, y, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive for interpretation
cv_scores = -cv_scores

# Calculate the mean and standard deviation of the MSE across folds
mean_mse = np.mean(cv_scores)
std_mse = np.std(cv_scores)

print(f"Mean MSE from cross-validation: {mean_mse}")
print(f"Standard deviation of MSE: {std_mse}")


# In[ ]:




