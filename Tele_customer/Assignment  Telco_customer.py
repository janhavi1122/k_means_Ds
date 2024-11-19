# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 00:34:50 2023

@author: arudr
"""

#3.	Perform clustering analysis on the telecom data set. 
#The data is a mixture of both categorical and numerical data. 
#It consists of the number of customers who churn out. 
#Derive insights and get possible information on factors that may affect the churn decision. 
#Refer to Telco_customer_churn.xlsx dataset.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#IMPORT DATA SET AND CREATE DATAFRAME
t_data = pd.read_excel("E:\datascience\k_means\Telco_customer_churn.xlsx")
t_data.describe()
t_data.columns
t_data.dtypes
t_data.shape
#(3999, 12)


#initially we will perform EDA to analyse the data

#pairplot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(t_data, height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(t_data['Tenure in Months'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#Boxplot and outliers treatment

sns.boxplot(t_data['Count'])
sns.boxplot(t_data['Number of Referrals'])
sns.boxplot(t_data['Tenure in Months'])
sns.boxplot(t_data['Avg Monthly Long Distance Charges'])
sns.boxplot(t_data['Avg Monthly GB Download'])
sns.boxplot(t_data['Total Extra Data Charges'])
sns.boxplot(t_data['Monthly Charge'])
sns.boxplot(t_data['Total Charges'])
sns.boxplot(t_data['Total Refunds'])
sns.boxplot(t_data['Total Long Distance Charges'])
sns.boxplot(t_data['Total Revenue'])

'''
from box plot except count, number of referrals, tenure in months,
avg monthly long dist charges, monthly charge, totla charges
all other colmns have outliers
we need to remove them
'''
#1
iqr = t_data['Number of Referrals'].quantile(0.75)-t_data['Number of Referrals'].quantile(0.25)
iqr
q1=t_data['Number of Referrals'].quantile(0.25)
q3=t_data['Number of Referrals'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Number of Referrals'] =  np.where(t_data['Number of Referrals']>u_limit,u_limit,np.where(t_data['Number of Referrals']<l_limit,l_limit,t_data['Number of Referrals']))
sns.boxplot(t_data['Number of Referrals'])

#2
iqr = t_data['Avg Monthly GB Download'].quantile(0.75)-t_data['Avg Monthly GB Download'].quantile(0.25)
iqr
q1=t_data['Avg Monthly GB Download'].quantile(0.25)
q3=t_data['Avg Monthly GB Download'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Avg Monthly GB Download'] =  np.where(t_data['Avg Monthly GB Download']>u_limit,u_limit,np.where(t_data['Avg Monthly GB Download']<l_limit,l_limit,t_data['Avg Monthly GB Download']))
sns.boxplot(t_data['Avg Monthly GB Download'])

#3
iqr = t_data['Total Extra Data Charges'].quantile(0.75)-t_data['Total Extra Data Charges'].quantile(0.25)
iqr
q1=t_data['Total Extra Data Charges'].quantile(0.25)
q3=t_data['Total Extra Data Charges'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Total Extra Data Charges'] =  np.where(t_data['Total Extra Data Charges']>u_limit,u_limit,np.where(t_data['Total Extra Data Charges']<l_limit,l_limit,t_data['Total Extra Data Charges']))
sns.boxplot(t_data['Total Extra Data Charges'])

#4
iqr = t_data['Total Refunds'].quantile(0.75)-t_data['Total Refunds'].quantile(0.25)
iqr
q1=t_data['Total Refunds'].quantile(0.25)
q3=t_data['Total Refunds'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Total Refunds'] =  np.where(t_data['Total Refunds']>u_limit,u_limit,np.where(t_data['Total Refunds']<l_limit,l_limit,t_data['Total Refunds']))
sns.boxplot(t_data['Total Refunds'])

#5
iqr = t_data['Total Long Distance Charges'].quantile(0.75)-t_data['Total Long Distance Charges'].quantile(0.25)
iqr
q1=t_data['Total Long Distance Charges'].quantile(0.25)
q3=t_data['Total Long Distance Charges'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Total Long Distance Charges'] =  np.where(t_data['Total Long Distance Charges']>u_limit,u_limit,np.where(t_data['Total Long Distance Charges']<l_limit,l_limit,t_data['Total Long Distance Charges']))
sns.boxplot(t_data['Total Long Distance Charges'])

#6
iqr = t_data['Total Revenue'].quantile(0.75)-t_data['Total Revenue'].quantile(0.25)
iqr
q1=t_data['Total Revenue'].quantile(0.25)
q3=t_data['Total Revenue'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
t_data['Total Revenue'] =  np.where(t_data['Total Revenue']>u_limit,u_limit,np.where(t_data['Total Revenue']<l_limit,l_limit,t_data['Total Revenue']))
sns.boxplot(t_data['Total Revenue'])

#now describe dataset
t_data.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset


t_data.drop(['Customer ID','Count','Quarter'],axis=1,inplace=True)
#get dummy variables from data set
df_n = pd.get_dummies(t_data)

df_n.shape
df_n

def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

df_normal = norm_func(df_n)
desc = df_normal.describe()
desc
df_normal.columns
#Total Refunds , Total Extra Data Charges contains NAN

df_normal.drop(['Total Extra Data Charges','Total Refunds'],axis=1,inplace=True)
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('Distance')
#ref of dendrogram

sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()


#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
t_data['cluster'] = cluster_labels
t_data.columns
t_data.shape
t_dataNew = t_data.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
t_dataNew.columns

t_dataNew.iloc[:,2:].groupby(t_dataNew.cluster).mean()
t_dataNew.to_csv("C:\datasets\Telco_customer_churn (1).xlsx",encoding='utf-8')
t_dataNew.cluster.value_counts()
import os
os.getcwd()

#************************************************************
#kmeans Clustering
    
#for this we will used normalized data set df_normal
from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_normal)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
[50926.26965397003,
 44701.402024025585,
 42000.59707770948,
 40149.71809708845,
 38978.40601245236,
 37912.60169484107]
'''

'''
k selected by calculating the difference or decrease in
twss value 
'''
def find_cluster_number(TWSS):
    diff =[]
    for i in range(0,len(TWSS)-1):
        d = TWSS[i]-TWSS[i+1]
        diff.append(d)
    max = 0
    k =0
    for i in range(0,len(diff)):
        if max<diff[i]:
            max = diff[i]
            k = i+3
    return k

k = find_cluster_number(TWSS)
print("Cluster number is = ",k)
plt.plot(k,TWSS,'ro-')
plt.xlabel('No of clusters')
plt.ylabel('Total_within_SS')

model = KMeans(n_clusters=k)
model.fit(df_normal)
model.labels_
mb = pd.Series(model.labels_)
df_normal['clusters'] = mb
df_normal.head()
df_normal.shape
df_normal.columns
df_normal = df_normal.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
df_normal
df_normal.iloc[:,2:11].groupby(df_normal.clusters).mean()
df_normal.to_csv("C:\datasets\KMeans_Telco_customer_churn (1).xlsx")
import os
os.getcwd()


