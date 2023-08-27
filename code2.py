#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


# In[35]:


file_path = "Data/po1_data.txt"


# In[36]:


column_names=['subject_identifier',
              'jitter in %',
              'jitter in microsec',
              'jitter r.a.p',
              'jitter p.p.q.5',
             'jitter d.d.p',
             'shimmer in %',
             'shimmer in dB',
             'shimmer a.p.q.3',
             'shimmer a.p.q.5',
             'shimmer a.p.q.11',
             'shimmer d.d.a',
             'autocorrelation between NHR-HNR',
             'NHR ratio',
             'HNR ratio',
             'Median pitch',
             'Mean pitch',
             'SD of pitch',
             'Min pitch',
             'Max pitch',
             'No. of pulses',
             'No. of periods',
             'Mean Period',
             'SD of period',
             'Fraction of unvoiced frames',
             'No. of voice breaks',
             'Degree of voice breaks',
             'UPDRS',
             'PD indicator']


# In[37]:


df = pd.read_csv(file_path, names=column_names)
features = df.drop(["subject_identifier", "PD indicator", "UPDRS"], axis=1)
label = df["PD indicator"]
data=pd.concat([features, label], axis=1)
parkinsons_data=data[data['PD indicator'] == 1 ]
healthy_data=data[data['PD indicator'] == 0]
parkinsons_stats=parkinsons_data.describe()
healthy_stats=healthy_data.describe()


# In[38]:


print("parkinson's Statistics:\n", parkinsons_stats)
print("healthy's Statistics:\n", healthy_stats)


# In[39]:


plt.figure(figsize=(10, 6))
for column in features.columns:
    plt.boxplot([parkinsons_data[column], healthy_data[column]], labels=['Parkinson\'s', 'Healthy'])
    plt.title(column)
    plt.ylabel("Value")
    plt.show()


# In[92]:


num_features = len(features.columns)
num_rows = math.ceil(num_features / 2)
num_cols = 2

plt.figure(figsize=(20, 15))
for idx, column in enumerate(features.columns, 1):
    plt.subplot(num_rows, num_cols, idx)  # Adjust subplot parameters
    plt.hist(parkinsons_data[column], bins=20, alpha=0.5, label="Parkinson's")
    plt.hist(healthy_data[column], bins=20, alpha=0.5, label="Healthy")
    plt.title(column)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
plt.tight_layout()
plt.show()


# In[41]:


from scipy.stats import ttest_ind, f_oneway


# In[42]:


t_test_results = {}
for feature in features.columns:
    t_stat, p_value = ttest_ind(parkinsons_data[feature], healthy_data[feature])
    t_test_results[feature] = {'t_statistic': t_stat, 'p_value': p_value}


# In[43]:


print("T-Test Results:\n", t_test_results)


# In[44]:


anova_results = {}
for feature in features.columns:
    anova_stat, p_value = f_oneway(
        parkinsons_data[feature], healthy_data[feature]
    )
    anova_results[feature] = {'anova_statistic': anova_stat, 'p_value': p_value}


# In[45]:


print("\nANOVA Results:\n", anova_results)


# In[89]:


correlations = data.corr()['PD indicator'].drop('PD indicator')
relevant_features_corr = correlations[abs(correlations) > 0.1]


# In[90]:


print("Correlations with Target:\n", relevant_features_corr)


# In[49]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[53]:


model = LogisticRegression(solver='liblinear', max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(features, label)


# In[54]:


print("Selected Features with RFE:\n", selected_features)


# In[55]:


from sklearn.ensemble import RandomForestClassifier


# In[57]:


model_rf = RandomForestClassifier()
model_rf.fit(features, label)

feature_importance = model_rf.feature_importances_


# In[58]:


print("Feature Importance from Random Forest:\n", feature_importance)


# In[59]:


feature_names = features.columns
plt.figure(figsize=(10, 6))
plt.barh(feature_names, feature_importance)
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Random Forest Feature Importance')
plt.show()


# In[ ]:




