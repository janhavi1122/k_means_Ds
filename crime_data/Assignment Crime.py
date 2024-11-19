# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 23:07:48 2023

@author: arudr
"""

#2.	Perform clustering for the crime data and 
#identify the number of clusters formed and draw inferences.
# Refer to crime_data.csv dataset.

'''
this dataset is consist of 4 cols murder, assualt, urbanpop and rape
this includes no. of cases filed for each type of cribe in different
states 

'''
#business objective - 
'''
business objective is to perform clustering on states that have 
similar charateristics

'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

c_data = pd.read_csv("E:\datascience\k_means\crime_data.csv")
c_data
c_data.dtypes
'''
Unnamed: 0     object
Murder        float64
Assault         int64
UrbanPop        int64
Rape          float64
dtype: object'''

c_data.shape
# (50, 5)

c_data.columns
#Index(['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')
c_data.describe()
'''
Murder     Assault   UrbanPop       Rape
count  50.00000   50.000000  50.000000  50.000000
mean    7.78800  170.760000  65.540000  21.232000
std     4.35551   83.337661  14.474763   9.366385
min     0.80000   45.000000  32.000000   7.300000
25%     4.07500  109.000000  54.500000  15.075000
50%     7.25000  159.000000  66.000000  20.100000
75%    11.25000  249.000000  77.750000  26.175000
max    17.40000  337.000000  91.000000  46.000000'''


#column unnamed which contains the states does not have much use
#so we can remove it from dataset
c_data.drop(['Unnamed: 0'],inplace=True,axis=1)

c_data.columns
#Index(['Murder', 'Assault', 'UrbanPop', 'Rape'], dtype='object')

#initially we will perform EDA to analyse the data

#pairplot
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(c_data,height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(c_data['Murder'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:], cdf)
plt.show();

#outliers treatment and boxplot

sns.boxplot(c_data['Murder'])
sns.boxplot(c_data['Assault'])
sns.boxplot(c_data['UrbanPop'])
sns.boxplot(c_data['Rape'])

#only last column rape contains outlier
#so we need to remove them

iqr = c_data['Rape'].quantile(0.75)-c_data['Rape'].quantile(0.25)
iqr

q1 = c_data['Rape'].quantile(0.25)
q3=c_data['Rape'].quantile(0.75)

l_limit = q1-(1.5*iqr)
u_limit = q3+(1.5*iqr)

c_data['Rape'] = np.where(c_data.Rape >u_limit,u_limit,np.where(c_data.Rape<l_limit,l_limit,c_data.Rape))
sns.boxplot(c_data['Rape'])

c_data.describe()
#we need to normalise this dataset
#initially normalize the dataset
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#apply this func on airlines dataset
df_norm = norm_fun(c_data)
b = df_norm.describe()
b

#dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
#dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()

#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
c_data['cluster'] = cluster_labels
c_data.columns
c_dataNew = c_data.iloc[:,[-1,0,1,2,3]]
c_dataNew.columns
c_dataNew.iloc[:,2:].groupby(c_dataNew.cluster).mean()
c_dataNew.to_csv("C:\datasets\crime_data (1).csv",encoding='utf-8')
c_dataNew.cluster.value_counts()
import os
os.getcwd()


##************************************************

####################################################
#KMeans Clustering on on crime data
#for this we will used normalized data set df_normal

from sklearn.cluster import KMeans
#total sum of squares
TWSS = []

#initially we will find the ideal cluster number using elbow curve

k = list(range(2,8))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
  
TWSS
'''
[6.755182167622326,
 5.134910987938266,
 3.764333164167156,
 3.253547530244435,
 2.8962730003632755,
 2.650015971697713]'''



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
model.fit(df_norm)
model.labels_
mb = pd.Series(model.labels_)
df_norm['clusters'] = mb
df_norm.head()
df_norm.shape
df_norm.columns
df_norm = df_norm.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10]]
df_norm
df_norm.iloc[:,2:11].groupby(df_norm.clusters).mean()
df_norm.to_csv("C:\datasets\crime_data (1).csv")
import os
os.getcwd()