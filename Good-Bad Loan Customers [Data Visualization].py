#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
# import the required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
# from sklearn.feature_selection import f_classif
# from sklearn.pipeline import Pipeline
# from sklearn.base import BaseEstimator, TransformerMixin
# from scipy.stats import chi2_contingency

pd.set_option("display.max_columns", 101)
pd.set_option("display.max_rows", 500)



# In[2]:


# import data
loan_data = pd.read_csv(r'C://Users/Namrata/Desktop/Corporate Projects/Credit Risk/loan_data_2007_2014 (1).csv')
# loan_data.shape

# # Describe function will only show non missing info.
# loan_data.describe()

# # dataframe columns
# loan_data.columns

# # check count of variables
# loan_data.count()

# # Using isnull() for getting info. on missing values
# loan_data.isnull().sum()
# # loan_data.isnull().sum().plot(kind="bar")

# # plt.show()

loan_data.info()

# # loan_data.head(5)

# # see what are the unique values in home_ownership column
# loan_data.home_ownership.unique()

# # see all variables naming with "pymnt"
# [column for column in loan_data.columns if "pymnt" in column]


# In[3]:


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


# In[4]:


x = loan_data.tot_cur_bal.dropna()  # Extract all non-missing values of tot_curr_bal into a variable called 'x'
print(x.mean()) # Through Pandas 
print(np.mean(x)) # Through Numpy 

print(x.median())
print(np.percentile(x, 50))  # 50th percentile, same as the median
print(np.percentile(x, 75))  # 75th percentile
print(x.quantile(0.75)) # Pandas method for quantiles, equivalent to 75th percentile

plot1=sns.distplot(loan_data.tot_cur_bal.dropna())

# find obs. for a column with values higher than median
curr_bal_median=pd.Series.median(loan_data["tot_cur_bal"])
condition1=loan_data["tot_cur_bal"] > curr_bal_median
# condition2=loan_data["total_pymnt"] > 10000
loan_data[condition1]
# loan_data[condition1 & condition2,:]


# In[5]:


loan_data.loc[:,['home_ownership','term']]


# In[6]:


# use groupby to see the distribution of term wrt home_ownership
loan_data.groupby(['term','home_ownership']).size()


# In[7]:


# find duplicate rows
duplicates=loan_data.duplicated()
print(duplicates)
loan_data[duplicates]

# sort data
loan_data=loan_data.sort_values(by="loan_amnt", ascending=True)


# In[8]:


# convert object(string) type date column to datetime type
from datetime import datetime
loan_data['earliest_cr_line'] = pd.to_datetime(loan_data['earliest_cr_line'],format='%b-%y')
loan_data['issue_d'] = pd.to_datetime(loan_data['issue_d'],format='%b-%y')
loan_data['last_credit_pull_d'] = pd.to_datetime(loan_data['last_credit_pull_d'],format='%b-%y')
loan_data['last_pymnt_d'] = pd.to_datetime(loan_data['last_pymnt_d'],format='%b-%y')
loan_data.info()
loan_data.head()


# In[9]:


# Visualizing and analysing loan_amnt using matplotlib
# distribution is right skewed (so, mean >median), centered at around 10,000 euros, with most income  being between 1000 and 35,000, with apparent outliers. On avg, the individual loan amount is 8286 euros away from mean. IQR is 12000 (tells where most of our data falls).
plt.hist(loan_data['loan_amnt'],bins=30)
plt.xlabel("loan_amnt")
plt.ylabel("count")
plt.show()

# visualization
plt.hist(loan_data["term"])
loan_data.describe()

# looks like total_acc has outliers in the range of 156. check through a boxplot and do outlier treatment
loan_data.loc[loan_data['total_acc'] > 50, 'total_acc'] = 50 

plt.figure(figsize=(10,5))
sns.boxplot(loan_data["total_acc"]).set_title("Box plot of total_acc")
plt.show()



# outliers can also be treated by identifying the lower bound and upper bound of data
# sns.boxplot(loan_data["annual_inc"]).set_title("Box plot of annual_inc")
# plt.show()

# def outlier_treatment(datacolumn):
#     sorted(datacolumn)
#     Q1,Q3 = np.percentile(datacolumn , [25,75])
#     IQR = Q3 - Q1
#     lower_range = Q1 - (1.5 * IQR)
#     upper_range = Q3 + (1.5 * IQR)
#     return lower_range,upper_range

# l,u = outlier_treatment(loan_data.annual_inc)


# # loan_data[(loan_data.annual_inc < l) | (loan_data.annual_inc > u)]
# loan_data.drop(loan_data[ (loan_data.annual_inc > u) | (loan_data.annual_inc < l) ].index , inplace=True)

# sns.boxplot(loan_data["annual_inc"]).set_title("Box plot of annual_inc")
# plt.show()


# In[10]:


# using sns for plotting histogram of loan amnt
sns.distplot(loan_data["loan_amnt"], kde = False).set_title("Histogram of Loan amount")
plt.show()
# Plot a histogram of both loan amnt and term
sns.distplot(loan_data["loan_amnt"], kde = False)
sns.distplot(loan_data["installment"], kde = False).set_title("Histogram of Both Loan amnt and installment")
plt.show()


# using sns for plotting a boxplot of loan amnt
sns.boxplot(loan_data["loan_amnt"]).set_title("Box plot of Loan_amnt")
plt.show()
# Create a boxplot of installement
sns.boxplot(loan_data["installment"]).set_title("Box plot of installement")
plt.show()
# Create a boxplot of loan amnt and funded amnt- do not do it like this
sns.boxplot(loan_data["loan_amnt"])
sns.boxplot(loan_data["funded_amnt"]).set_title("Box plot of Loan_amnt and funded amnt")
plt.show()
# Create a boxplot of the loan_amnt grouped by purpose 
sns.boxplot(x = loan_data["loan_amnt"], y = loan_data["purpose"])
plt.show()
# creating boxplot
g=sns.boxplot (x=loan_data["mths_since_last_delinq"])
g.set_title("months since last deliquent",y=1.03)


# using sns for creating scatter plot
sns.relplot(x='loan_amnt', y='annual_inc', data=loan_data, kind='scatter',hue='home_ownership')
plt.show()

# seaborn plots creates two types of object-Facetgrid and Axessubplot. 
#Relplot and catplot that can create subplots support facetgrid where as single plots like scatterplot support axessubplot.
h=sns.catplot(x='purpose', y='good_bad', data=loan_data, kind='bar',col='verification_status')
h.fig.suptitle('good_bad distribution across purpose',y=1.03)
h.set_titles('This is {col_name}')
plt.xticks(rotation=90)
plt.show()
type(h)

# Create a boxplot and histogram of the tips grouped by good bad
sns.boxplot(x = loan_data["loan_amnt"], y = loan_data["purpose"]).set_title("Boxplot of loan amnt by purpose")

g = sns.FacetGrid(loan_data, row = "purpose") #group all plots by purpose
g = g.map(plt.hist, "loan_amnt") #for each group, create a histogram
plt.show()


# In[11]:


g = sns.FacetGrid(loan_data, row = "term",col="purpose") 
g.map_dataframe(sns.histplot, x="loan_amnt")
g.set_axis_labels("Loan_amnt", "Count")

h = sns.FacetGrid(loan_data, row = "term")
h.map_dataframe(sns.histplot, x="loan_amnt")
h.set_axis_labels("Loan_amnt", "Count")


# In[12]:


sns.countplot(x='term', data=loan_data)
plt.show()
sns.catplot(x='term', data=loan_data,kind='count',col='purpose')
plt.show()

# extending whiskers to 5 and 95 percentiles
sns.set_palette("Purples")
sns.set_style("whitegrid")
sns.catplot(x='good_bad', y='revol_bal', data=loan_data,kind='box',whis=[5,95])
plt.show()


# In[13]:


loan_data["loanamt_grp"] = pd.cut(loan_data.loan_amnt, [5000, 10000, 15000, 20000, 25000, 30000, 35000]) # Create age strata based on these cut points
plt.figure(figsize=(12, 5))  # Make the figure wider than default (12cm wide by 5cm tall)
sns.boxplot(x="loanamt_grp", y="installment", data=loan_data)  # Make boxplot of BPXSY1 stratified by age group# explore statistics

# loan_data.groupby("loanamt_grp")["good_bad"].value_counts()


# In[14]:


# from scipy import stats
# stats.describe(sorted_loan_data['loan_amnt'])

# example of joint distribution when you are not able to get relation between two variables when compared separately.
# Plot the data
plt.figure(figsize=(10,10))
plt.subplot(2,2,2)
plt.scatter(x = loan_data["installment"], y = loan_data["loan_amnt"])
plt.title("Joint Distribution of installemnt and loan amnt")
# Plot the Marginal X Distribution
plt.subplot(2,2,4)
plt.hist(x = loan_data["installment"], bins = 15)
plt.title("Marginal Distribution of installemnt")
# Plot the Marginal Y Distribution
plt.subplot(2,2,1)
plt.hist(x = loan_data["loan_amnt"], orientation = "horizontal", bins = 15)
plt.title("Marginal Distribution of loan amnt")
# Show the plots
plt.show()

# # split data into 80/20 while keeping the distribution of bad loans in test set same as that in the pre-split dataset
# X = loan_data.drop('good_bad', axis = 1)
# y = loan_data['good_bad']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, 
#                                                     random_state = 42, stratify = y)

# # hard copy the X datasets to avoid Pandas' SetttingWithCopyWarning when we play around with this data later on.
# # this is currently an open issue between Pandas and Scikit-Learn teams
# X_train, X_test = X_train.copy(), X_test.copy()


# In[15]:


sns.regplot(x="loan_amnt", y="funded_amnt", data=loan_data, fit_reg=False, scatter_kws={"alpha": 0.2})

sns.FacetGrid(loan_data, col="home_ownership").map(plt.scatter, "loan_amnt","funded_amnt", alpha=0.4).add_legend()


# In[16]:


#Pearson correlation coefficient ranges from -1 to 1, with values approaching 1 indicating a more perfect positive dependence. In many settings, a correlation of 0.62 would be considered a moderately strong positive dependence.
#ignore deprecation warning message
sns.jointplot(x="installment", y="int_rate", kind='kde', data=loan_data).annotate(stats.pearsonr)


# In[ ]:


# 'corr' method of a dataframe calculates the correlation coefficients for every pair of variables in the dataframe. This method returns a "correlation matrix"
loan_data.dropna().corr()

# cross tabulation
x=pd.crosstab(loan_data.home_ownership,loan_data.good_bad)
x


# The following line does these steps, reading the code from left to right:
# 1 Group the data by every combination of "verification_status","home_ownership"
# 2 Count the number of people in each cell using the 'size' method
# 3 Pivot the home_ownership results into the columns (using unstack)
# 4 Fill any empty cells with 0
# 5 Normalize the data by row
loan_data.groupby(["verification_status","home_ownership"]).size().unstack().fillna(0).apply(lambda x: x/x.sum(), axis=1)

