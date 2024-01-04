#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[4]:


import numpy as np
df=pd.read_csv("news_share_data.csv")


# In[5]:


df.shape


# In[6]:


df.head(1003)


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


df.dtypes


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
# box plot to see outlayer
plt.figure(figsize=(20, 10))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 6, i)
    sns.boxplot(df[feature])
    plt.title(f'Box Plot of {feature}')
    plt.tight_layout()

plt.show()


# In[13]:


#distribution
plt.figure(figsize=(25, 15))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(4, 6, i)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.tight_layout()

plt.show()


# In[14]:



numerical_summary = df.describe()

#iqr method
Q1 = numerical_summary.loc['25%']
Q3 = numerical_summary.loc['75%']
IQR = Q3 - Q1

# outliers using IQR method
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Display outlayes 
outliers = ((df < lower_bound) | (df > upper_bound)).sum()
outliers = outliers[outliers > 0]
print("Features with potential outliers:")
print(outliers)


# In[15]:


import numpy as np

features_with_outliers = ['article_id', 'average_token_length', 'avg_avg_key', 'global_rate_negative_words',
                           'global_rate_positive_words', 'global_sentiment_polarity', 'global_subjectivity',
                           'href_avg_shares', 'max_avg_key', 'num_hrefs', 'num_imgs', 'num_videos', 'shares',
                           'title_sentiment_polarity', 'unique_tokens_rate']

#logarithm transformation
for feature in features_with_outliers:
    df[feature] = np.log1p(df[feature])
# histogram
plt.figure(figsize=(15, 25))

for i, feature in enumerate(features_with_outliers, 1):
    plt.subplot(9,3, i)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[16]:


from scipy.stats.mstats import winsorize
import matplotlib.pyplot as plt
import seaborn as sns

# sentiment features
sentiment_features = ['title_sentiment_polarity', 'global_sentiment_polarity']

#winsorizing technique
for feature in sentiment_features:
    df[feature] = winsorize(df[feature], limits=[0.05, 0.05])


# In[17]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))

for i, feature in enumerate(sentiment_features, 1):
    plt.subplot(1, 2, i)
    sns.histplot(df[feature], bins=30, kde=True)
    plt.title(f'Distribution of {feature}')

plt.tight_layout()
plt.show()


# In[69]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

correlation_matrix = df.corr()

# Plotting the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title("Correlation Matrix")
plt.show()


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer

# one-hot encoding for title column
title_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
title_encoded = title_vectorizer.fit_transform(df['title']).toarray()

# Eone-hot encoding for text column 
text_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
text_encoded = text_vectorizer.fit_transform(df['text']).toarray()


# In[72]:


df.head(1003)


# In[ ]:





# In[ ]:





# In[ ]:





# In[75]:


from sklearn.model_selection import train_test_split

X = df_filtered.drop(target_variable, axis=1)  # Features
y = df_filtered[target_variable]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[76]:


from sklearn.ensemble import RandomForestRegressor  # Example, use appropriate model

model = RandomForestRegressor()  # Initialize the model


# In[77]:


model.fit(X_train, y_train)


# In[78]:


predictions = model.predict(X_test)


# In[79]:


from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


# In[80]:


import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance


y_pred = model.predict(X_test)

# Evaluate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

if hasattr(model, 'feature_importances_'):
    feature_importance = model.feature_importances_
    feature_names = X_test.columns
    importance_dict = dict(zip(feature_names, feature_importance))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nFeature Importance:")
    for feature, importance in sorted_importance:
        print(f"{feature}: {importance}")

# Visualize Predicted vs. Actual
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Number of Shares")
plt.ylabel("Predicted Number of Shares")
plt.title("Actual vs. Predicted Number of Shares")
plt.show()

# Visualize Residuals
residuals = y_test - y_pred
plt.scatter(y_test, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Actual Number of Shares")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.show()

# Permutation Importance
if hasattr(model, 'feature_importances_'):
    perm_importance = permutation_importance(model, X_test, y_test)
    sorted_perm_importance = sorted(zip(X_test.columns, perm_importance.importances_mean), key=lambda x: x[1], reverse=True)

    print("\nPermutation Importance:")
    for feature, importance in sorted_perm_importance:
        print(f"{feature}: {importance}")


# In[ ]:





# In[82]:


df.head(1003)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[87]:


correlations = df[['day_of_week', 'month', 'shares']].corr()
print(correlations)


# In[88]:


# Assuming 'day_of_week', 'month', and other features are part of your X
X = df.drop(['shares'], axis=1)
y = df['shares']


# In[89]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[90]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Model without 'day_of_week' and 'month'
model_no_dates = LinearRegression()
model_no_dates.fit(X_train.drop(['day_of_week', 'month'], axis=1), y_train)
y_pred_no_dates = model_no_dates.predict(X_test.drop(['day_of_week', 'month'], axis=1))
mse_no_dates = mean_squared_error(y_test, y_pred_no_dates)
print("MSE without dates:", mse_no_dates)

# Model with 'day_of_week' and 'month'
model_with_dates = LinearRegression()
model_with_dates.fit(X_train, y_train)
y_pred_with_dates = model_with_dates.predict(X_test)
mse_with_dates = mean_squared_error(y_test, y_pred_with_dates)
print("MSE with dates:", mse_with_dates)


# In[ ]:





# In[ ]:





# In[ ]:





# In[94]:


# Assuming 'day_of_week', 'month', and other features are part of your X
X = df.drop(['shares'], axis=1)
y = df['shares']


# In[95]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[96]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Model without 'day_of_week' and 'month'
model_no_dates = LinearRegression()
model_no_dates.fit(X_train.drop(['day_of_week', 'month'], axis=1), y_train)
y_pred_no_dates = model_no_dates.predict(X_test.drop(['day_of_week', 'month'], axis=1))
mse_no_dates = mean_squared_error(y_test, y_pred_no_dates)
print("MSE without dates:", mse_no_dates)

# Model with 'day_of_week' and 'month'
model_with_dates = LinearRegression()
model_with_dates.fit(X_train, y_train)
y_pred_with_dates = model_with_dates.predict(X_test)
mse_with_dates = mean_squared_error(y_test, y_pred_with_dates)
print("MSE with dates:", mse_with_dates)


# In[99]:



pip install xgboost


# In[100]:


import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

# Create an XGBoost regressor
model_xgb = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)

# Fit the model on the training data
model_xgb.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = model_xgb.predict(X_test)

# Calculate mean squared error on the test set
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
print("Mean Squared Error (XGBoost):", mse_xgb)

# Cross-validation to evaluate the model performance
cross_val_mse_xgb = -cross_val_score(model_xgb, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print("Mean Cross-Validation MSE (XGBoost):", cross_val_mse_xgb.mean())


# In[ ]:





# In[ ]:




