# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 11:20:39 2023

@author: arudr
"""


#Dataset -  EAST WEST AIRLINES.XLS

'''
The file EastWestAirlines contains information on passengers 
who belong to an airline’s frequent flier program.
For each passenger the data include information on their mileage 
history and on different ways they spent miles in the last year. 
'''
#Business Objective - 
'''
is to find clusters of passengers that are of the similar characteristics
to provide milage offers based on clustering /  groups
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#IMPORT DATA SET AND CREATE DATAFRAME
airlines = pd.read_excel("E:\datascience\k_means\EastWestAirlines.xlsx")

airlines.dtypes
'''ID#                  int64
Balance              int64
Qual_miles           int64
cc1_miles            int64
cc2_miles            int64
cc3_miles            int64
Bonus_miles          int64
Bonus_trans          int64
Flight_miles_12mo    int64
Flight_trans_12      int64
Days_since_enroll    int64
Award?               int64
dtype: object

In this data set all the columns are the integer type 
'''

airlines.columns
airlines.shape
#(3999, 12)

airlines.describe()
'''
airlines.describe()
Out[7]: 
               ID#       Balance  ...  Days_since_enroll       Award?
count  3999.000000  3.999000e+03  ...         3999.00000  3999.000000
mean   2014.819455  7.360133e+04  ...         4118.55939     0.370343
std    1160.764358  1.007757e+05  ...         2065.13454     0.482957
min       1.000000  0.000000e+00  ...            2.00000     0.000000
25%    1010.500000  1.852750e+04  ...         2330.00000     0.000000
50%    2016.000000  4.309700e+04  ...         4096.00000     0.000000
75%    3020.500000  9.240400e+04  ...         5790.50000     1.000000
max    4021.000000  1.704838e+06  ...         8296.00000     1.000000

[8 rows x 12 columns]

from this we can see that min, max, mean values have huge difference
so data need to be normalized
'''

#initially we will perform EDA to analyse the data

#pairplot
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(airlines, hue="Award?", height=3);
plt.show()

#pdf and cdf

counts, bin_edges = np.histogram(airlines['Balance'], bins=10, 
                                 density = True)
pdf = counts/(sum(counts))
print(pdf);
print(bin_edges)

#compute CDF
cdf = np.cumsum(pdf)
plt.plot(bin_edges[1:],pdf)
'''
from pdf we can say that approx 90% of data have balance 20000
'''
plt.plot(bin_edges[1:], cdf)
plt.show();

#Boxplot and outliers treatment

sns.boxplot(airlines['Balance'])
sns.boxplot(airlines['Qual_miles'])
sns.boxplot(airlines['cc1_miles'])
sns.boxplot(airlines['cc2_miles'])
sns.boxplot(airlines['cc3_miles'])
sns.boxplot(airlines['Bonus_miles'])
sns.boxplot(airlines['Bonus_trans'])
sns.boxplot(airlines['Flight_miles_12mo'])
sns.boxplot(airlines['Flight_trans_12'])
sns.boxplot(airlines['Days_since_enroll'])
sns.boxplot(airlines['Award?'])

'''
from box plot except cc2 miles, days since enroll and award? 
all other colmns have outliers
we need to remove them
'''
#1
iqr = airlines['Balance'].quantile(0.75)-airlines['Balance'].quantile(0.25)
iqr
q1=airlines['Balance'].quantile(0.25)
q3=airlines['Balance'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Balance'] =  np.where(airlines['Balance']>u_limit,u_limit,np.where(airlines['Balance']<l_limit,l_limit,airlines['Balance']))
sns.boxplot(airlines['Balance'])

#2
iqr = airlines['Qual_miles'].quantile(0.75)-airlines['Qual_miles'].quantile(0.25)
iqr
q1=airlines['Qual_miles'].quantile(0.25)
q3=airlines['Qual_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Qual_miles'] =  np.where(airlines['Qual_miles']>u_limit,u_limit,np.where(airlines['Qual_miles']<l_limit,l_limit,airlines['Qual_miles']))
sns.boxplot(airlines['Qual_miles'])

#3
iqr = airlines['cc1_miles'].quantile(0.75)-airlines['cc1_miles'].quantile(0.25)
iqr
q1=airlines['cc1_miles'].quantile(0.25)
q3=airlines['cc1_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['cc1_miles'] =  np.where(airlines['cc1_miles']>u_limit,u_limit,np.where(airlines['cc1_miles']<l_limit,l_limit,airlines['cc1_miles']))
sns.boxplot(airlines['cc1_miles'])

#4
iqr = airlines['cc3_miles'].quantile(0.75)-airlines['cc3_miles'].quantile(0.25)
iqr
q1=airlines['cc3_miles'].quantile(0.25)
q3=airlines['cc3_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['cc3_miles'] =  np.where(airlines['cc3_miles']>u_limit,u_limit,np.where(airlines['Bonus_miles']<l_limit,l_limit,airlines['Bonus_miles']))
sns.boxplot(airlines['cc3_miles'])

#5
iqr = airlines['Bonus_miles'].quantile(0.75)-airlines['Bonus_miles'].quantile(0.25)
iqr
q1=airlines['Bonus_miles'].quantile(0.25)
q3=airlines['Bonus_miles'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Bonus_miles'] =  np.where(airlines['Bonus_miles']>u_limit,u_limit,np.where(airlines['Bonus_miles']<l_limit,l_limit,airlines['Bonus_miles']))
sns.boxplot(airlines['Bonus_miles'])

#6
iqr = airlines['Bonus_trans'].quantile(0.75)-airlines['Bonus_trans'].quantile(0.25)
iqr
q1=airlines['Bonus_trans'].quantile(0.25)
q3=airlines['Bonus_trans'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Bonus_trans'] =  np.where(airlines['Bonus_trans']>u_limit,u_limit,np.where(airlines['Bonus_trans']<l_limit,l_limit,airlines['Bonus_trans']))
sns.boxplot(airlines['Bonus_trans'])

#7
iqr = airlines['Flight_miles_12mo'].quantile(0.75)-airlines['Flight_miles_12mo'].quantile(0.25)
iqr
q1=airlines['Flight_miles_12mo'].quantile(0.25)
q3=airlines['Flight_miles_12mo'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Flight_miles_12mo'] =  np.where(airlines['Flight_miles_12mo']>u_limit,u_limit,np.where(airlines['Flight_miles_12mo']<l_limit,l_limit,airlines['Flight_miles_12mo']))
sns.boxplot(airlines['Flight_miles_12mo'])

#8
iqr = airlines['Flight_trans_12'].quantile(0.75)-airlines['Flight_trans_12'].quantile(0.25)
iqr
q1=airlines['Flight_trans_12'].quantile(0.25)
q3=airlines['Flight_trans_12'].quantile(0.75)

l_limit = q1-1.5*(iqr)
u_limit = q3+1.5*iqr
airlines['Flight_trans_12'] =  np.where(airlines['Flight_trans_12']>u_limit,u_limit,np.where(airlines['Flight_trans_12']<l_limit,l_limit,airlines['Flight_trans_12']))
sns.boxplot(airlines['Flight_trans_12'])

#now describe dataset
airlines.describe()
#we can see that there is huge difference between min,max and mean
# values for all the columns so we need to normalize the dataset

#initially normalize the dataset
def norm_fun(i):
    x = (i-i.min())/(i.max()-i.min())
    return x

#apply this func on airlines dataset
df_normal = norm_fun(airlines)
b = df_normal.describe()
b
#as qual miles is containing NAN values so we will drop it
df_normal.drop(['Qual_miles'],axis=1,inplace=True)

#now all the data is normalized
#dendrogram
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_normal,method='complete',metric='euclidean')
plt.figure(figsize=(15,8))
plt.title('Hierarchical clustering dendrogram')
plt.xlabel('index')
plt.ylabel('distance')
#dendrogram
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10)
plt.show()
#now apply clustering 
from sklearn.cluster import AgglomerativeClustering
h_complete = AgglomerativeClustering(n_clusters=3,
                                     linkage='complete',
                                     affinity='euclidean').fit(df_normal)
#apply labels to clusters
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
#assign this series to autoIns dataframe as column
airlines['cluster'] = cluster_labels

airlinesNew = airlines.iloc[:,[-1,0,1,2,3,4,5,6,7,8,9,10,11]]
airlinesNew.iloc[:,2:].groupby(airlinesNew.cluster).mean()

airlinesNew.to_csv("C:\datasets\EastWestAirlines.xlsx",encoding='utf-8')
airlinesNew.cluster.value_counts()
import os
os.getcwd()


#############################################
#KMeans Clustering on east west airlines
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
[2908.260739348439,
 2263.316171028324,
 1932.897637296432,
 1667.4170613324961,
 1484.8283001092923,
 1313.3078600029346]
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
df_normal.to_csv("C:\datasets\KMeans_EastWestAirlines.xlsx")
import os
os.getcwd()

