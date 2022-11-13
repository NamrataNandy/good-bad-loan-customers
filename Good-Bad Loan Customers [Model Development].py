#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 500)


# In[2]:


# import data
loan_data = pd.read_csv(r'C://Users/Namrata/Desktop/Corporate Projects/Credit Risk/loan_data_2007_2014 (1).csv')

#gives the count of records across variables
loan_data.shape

# Describe function will only show non missing info.
loan_data.describe()

# dataframe columns
loan_data.columns

# check count of variables
loan_data.count()

# Using isnull() for getting info. on missing values
loan_data.isnull().sum()
# loan_data.isnull().sum().plot(kind="bar")

# plt.show()

loan_data.info()

# loan_data.head(5)

# see what are the unique values in home_ownership column
loan_data.home_ownership.unique()

# see all variables naming with "pymnt"
[column for column in loan_data.columns if "pymnt" in column]


# In[3]:


loan_data.describe()

# loan_data.info()


# In[4]:


#drop columns with more than 80% missing values (keep 20% missing data,axis=1 for columns, inplace true to ammend in same dataset)
loan_data.dropna(thresh=loan_data.shape[0]*0.2, axis=1, how='all', inplace=True)
# loan_data.shape

#drop all redundant and forward-looking columns
loan_data.drop(columns = ['id', 'member_id', 'sub_grade', 'emp_title', 'url', 'desc', 'title',
                          'zip_code', 'next_pymnt_d', 'recoveries', 'collection_recovery_fee',
                          'total_rec_prncp', 'total_rec_late_fee','pub_rec','addr_state',
                          'earliest_cr_line','last_pymnt_d','last_credit_pull_d','issue_d','policy_code'], inplace = True)

# see how the filtered variables are looking
loan_data.isnull().sum()


# In[5]:


# identifying dependent variable
loan_data["loan_status"].value_counts(normalize=True)


# In[6]:


# identifying dependent variable
loan_data["loan_status"].value_counts(normalize=True)

# create a new column based on the loan_status column that will be our target variable
loan_data["good_bad"]=np.where(loan_data.loc[:,"loan_status"].isin(['Charged Off', 'Default',
                                                                       'Late (31-120 days)',
                                                                       'Does not meet the credit policy. Status:Charged Off']),
                                 0, 1)

# Drop the original 'loan_status' column
loan_data.drop(columns=['loan_status'],inplace=True)
loan_data.head(5)


# In[7]:


# data cleaning for term (remove "months" word)
# type(loan_data["term"])
loan_data["term"] = loan_data["term"].str.strip('months')
# loan_data["term"] = loan_data["term"].astype('int')
loan_data["term"].head(5)
loan_data["term"].value_counts(normalize=True)


# data cleaning for emp length using loc
loan_data.loc[loan_data['emp_length'] == "10+ years", 'emp_length'] = 10 
loan_data.loc[loan_data['emp_length'] == '2 years', 'emp_length'] = 2 
loan_data.loc[loan_data['emp_length'] == "3 years", 'emp_length'] = 3 
loan_data.loc[loan_data['emp_length'] == '< 1 year', 'emp_length'] = 0 
loan_data.loc[loan_data['emp_length'] == "5 years", 'emp_length'] = 5 
loan_data.loc[loan_data['emp_length'] == '1 year', 'emp_length'] = 1
loan_data.loc[loan_data['emp_length'] == "4 years", 'emp_length'] = 4 
loan_data.loc[loan_data['emp_length'] == '7 years', 'emp_length'] = 7 
loan_data.loc[loan_data['emp_length'] == '6 years', 'emp_length'] = 6 
loan_data.loc[loan_data['emp_length'] == "8 years", 'emp_length'] = 8 
loan_data.loc[loan_data['emp_length'] == '9 years', 'emp_length'] = 9

# loan_data["emp_length"]=loan_data["emp_length"].astype('category')

# name missing entires in emp_length as "missing"
loan_data["emp_length"] = loan_data.emp_length.fillna("Missing")
# loan_data["emp_length"].value_counts()


# loan_data.head(5)
# loan_data["emp_length"].value_counts().sum()
# loan_data.info()

#Another alternative using function
# def bonus(val):
#     if (val in ["2 years"]):
#         return 2
#     elif (val in ["3 years"]):
#         return 3
#     elif (val in ["5 years"]):
#         return 5
#     elif (val in ["4 years"]):
#         return 4
#     elif (val in ["7 years"]):
#         return 7
#     elif (val in ["6 years"]):
#         return 6
#     elif (val in ["8 years"]):
#         return 8
#     elif (val in ["9 years"]):
#         return 9
#     elif (val in ["<1 year"]):
#         return 0
#     elif (val in ["1 year"]):
#         return 1
#     elif (val in ["10+ years"]):
#         return 10
#     else:
#         return 0
# loan_data["new"] = loan_data["emp_length"].apply(bonus)

# loan_data.head(5)
# loan_data["new"].value_counts()

# loan_data["new"]=loan_data["new"].astype('int')


# In[8]:


# another alternatrive using replace()-> but getting float no.
# loan_data["emp_length1"] = loan_data.emp_length.replace({"1 year": 1, "<1 year": 0, "2 years": 2,"3 years": 3,"4 years":4, 
#                                        "5 years": 5,"6 years":6,"7 years":7,"8 years":8,"9 years":9,"10+ years":10})
# loan_data.emp_length1.value_counts()
# loan_data["emp_length1"]=loan_data["emp_length1"].astype('int')

# loan_data.head()
# type(loan_data["emp_length1"])


# In[9]:


for n in loan_data.select_dtypes('number').columns:
    loan_data[n]=loan_data[n].fillna(value=loan_data[n].median())
    
for i in loan_data.select_dtypes('object').columns:
    loan_data[i]=loan_data[i].fillna(value=loan_data[i].mode())


# In[10]:



loan_data["emp_length"] = loan_data.emp_length.fillna("Missing")
loan_data.loc[loan_data['emp_length']=="Missing","emp_length"]=0
loan_data["emp_length"].value_counts()

# see how the filtered variables are looking
loan_data.isnull().sum()


# In[11]:


# outlier treatment
sns.boxplot (x=loan_data["open_acc"])
loan_data.loc[loan_data["open_acc"]>20,"open_acc"]=20


# In[12]:


# loan_data1.info()

from statsmodels.stats.outliers_influence import variance_inflation_factor
# x=loan_data.drop(columns='good_bad',axis=1)
vif_data=pd.DataFrame()

x=loan_data.drop(columns='good_bad',axis=1).select_dtypes('number')

vif_data["features"]=x.columns


# # calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(x.values, i)
                          for i in range(len(x.columns))]
  
print(vif_data)


# In[13]:


# drop variables with high vif one by one
# loan_data.drop(columns = ['out_prncp_inv'], inplace = True)
# loan_data.drop(columns = ['total_pymnt_inv'], inplace = True)
# loan_data.drop(columns = ['funded_amnt'], inplace = True)
# loan_data.drop(columns = ['funded_amnt_inv'], inplace = True)
loan_data.drop(columns = ['installment'], inplace = True)


# In[14]:


loan_data.isnull().sum()

# create dummy variables for grade,home_ownership,verification_status,pymnt_plan,purpose,application_type
loan_data1=pd.get_dummies(data=loan_data,columns=['initial_list_status','grade','home_ownership','verification_status','pymnt_plan','purpose','application_type'])


# In[15]:


# divide dataset in train test
training_data, testing_data = train_test_split(loan_data1, test_size=0.3, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")


# In[16]:


# Fitting the model

# training_data.head()
x_train=training_data.drop(columns='good_bad',axis=1)
y_train=training_data['good_bad']

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)


# In[17]:


# Predicting on test dataset

x_test=testing_data.drop(columns='good_bad',axis=1)
y_test=testing_data['good_bad']

y_pred = classifier.predict(x_test)


# In[18]:


cm = confusion_matrix(y_test, y_pred)
  
print ("Confusion Matrix : \n", cm)


# In[19]:


print ("Accuracy : ", accuracy_score(y_test, y_pred))
print ("Precision : ", precision_score(y_test, y_pred))
print ("Recall : ", recall_score(y_test, y_pred))


# 

# In[20]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
print(roc_auc)


# Below also gives auc
# from sklearn.metrics import roc_auc_score
# auc = roc_auc_score(y_test, y_pred)
# print(auc)

# matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn')
# plot roc curves
plt.plot(fpr, tpr, linestyle='--',color='orange', label='Logistic Regression')
# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

# Plot positive sloped 1:1 line for reference
plt.plot([0,1],[0,1])

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show();


# In[21]:


print(thresholds)


# In[33]:


print(classification_report(y_test, y_pred))

