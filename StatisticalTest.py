#!/usr/bin/env python
# coding: utf-8

# In[1]:


from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import statistics


# In[2]:


repo = "errors\\"
dataset = pd.read_csv(repo+"statistical"+".csv", delimiter=';')

for i in range(len(dataset)):
    for j in range(dataset.shape[1]):
        temp = dataset.iloc[i,j].replace(',','.')
        dataset.iloc[i,j] = float(temp)
print(dataset)


# In[3]:


chrome_dl = pd.read_csv(repo+"errorsMLP_Chrome"+".csv", delimiter=',', header=None)
chrome_stat = dataset["Chrome (ARIMA)"]

explorer_dl = pd.read_csv(repo+"errorsBiLSTM_Explorer"+".csv", delimiter=',', header=None)
explorer_stat = dataset["Explorer (ARIMA)"]

mac_dl = pd.read_csv(repo+"errorsCNN_AppleMac"+".csv", delimiter=',', header=None)
mac_stat = dataset["Mac OS (ARIMA)"]

ubuntu_dl = pd.read_csv(repo+"errorsLSTM_Ubuntu"+".csv", delimiter=',', header=None)
ubuntu_stat = dataset["Ubuntu Linux (TES)"]

office_dl = pd.read_csv(repo+"errorsMLP_Office"+".csv", delimiter=',', header=None)
office_stat = dataset["Office (SES)"]


# In[4]:


print(mac_dl[0].describe())


# In[5]:


print(mac_stat.describe())
print(sum(mac_stat)/len(mac_stat))


# In[6]:


print("Chrome :", wilcoxon(chrome_dl[0].values, chrome_stat.values))

print("Explorer :", wilcoxon(explorer_dl[0].values, explorer_stat.values))

print("Max :", wilcoxon(mac_dl[0].values, mac_stat.values))

print("Ubuntu :", wilcoxon(ubuntu_dl[0].values, ubuntu_stat.values))

print("Office :", wilcoxon(office_dl[0].values, office_stat.values))


# In[7]:


print(chrome_dl[0].values)


# In[8]:


print(chrome_stat.values)


# In[9]:


print(wilcoxon(chrome_dl[0].values, chrome_stat.values))


# In[23]:


group1 = [14.32485104,   5.4607563,   13.45495319,   3.95881462,  14.39033604,
  25.74808311,  21.22142601,   5.46264315,  19.04741669,  65.72915459,
  24.69551659,   1.02880096,  13.80880165,   4.81204939,  13.55166054,
  63.87245178,  20.69388008,  12.05981922,  25.22048569,  39.81968117,
 107.05954504,  12.96543503,   2.28498936,   7.86790085]
group2 = [4.672946, 32.243227, 7.638032, 7.635817, 17.014812, 27.967569, 2.010689,
 30.748665, 13.061463, 11.822944, 3.835486, 13.661701, 0.873164, 20.577333,
 13.70946, 14.155952, 35.253047, 14.704874, 12.392774, 11.625264, 1.878913,
 2.396878, 26.626422, 10.160185]
  
# conduct the Wilcoxon-Signed Rank Test
wilcoxon(group1, group2)


# In[29]:


df = pd.DataFrame()
df["statistical"]=chrome_stat.values
df["dl"]=chrome_dl[0].values
print(df)
df.to_csv(repo+"wlcxn"+"_"+"Chrome"+".csv", index=None)


# In[10]:


sum(chrome_dl[0].values.tolist())/len(chrome_dl[0].values.tolist())

