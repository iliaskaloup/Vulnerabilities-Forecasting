#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:35:19 2022

@author: iliaskaloup
"""

import pandas as pd
import json
import csv
import os


def mergeCategories(directory):
    file_data = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            print(filename)
            with open(directory+"\\"+filename,'r+', encoding="utf-8") as file:
                new_data = json.load(file)
                file_data.append(new_data)
    
    with open("allYears.json", 'w') as json_file:
            json.dump(file_data, json_file, indent=4)
    
    return file_data


inputPath = os.getcwd()
#file_data = mergeCategories(inputPath + '\\data\\')
with open("allYears.json",'r+') as file:
    file_data = json.load(file)
print("end")
print(len(file_data))


# In[2]:


software = "Ubuntu"
keyword = "canonical:ubuntu_linux"

myList = []
# search keywords in cve description
'''for i in range(0, len(file_data)):
    items = file_data[i]['CVE_Items']
    for item in items:
        description = item['cve']['description']['description_data'][0]['value']
        if keyword in description.lower():
            myList.append(item)'''

# search keywords in cve configuration
for i in range(0, len(file_data)):
    items = file_data[i]['CVE_Items']
    for item in items:
        nodes = item['configurations']['nodes']
        if (len(nodes) > 0):
            cpe = nodes[0]["cpe_match"]
            #print(item['configurations']['nodes'][0])
            #for j in range(0,len(cpe)):
            if (len(cpe) > 0):
                uri = cpe[0]["cpe23Uri"]
                if keyword in uri:
                    myList.append(item)
                    #break
print("end")
filename = "allVulns" + software + ".json"
with open(filename, 'w') as json_file:
    json.dump(myList, json_file, indent=4)


# In[3]:


# -*- coding: utf-8 -*-
"""
Created on June 06 10:35:19 2022

@author: iliaskaloup
"""

import pandas as pd
import json
import csv
import os

#software = "Chrome"

filename = "allVulns" + software + ".json"
with open(filename,'r+') as file:
    file_data = json.load(file)

print(len(file_data))


# In[4]:


timestamps = []
for i in range(0, len(file_data)):
    timestamps.append(file_data[i]["publishedDate"])
print(len(timestamps))


# In[5]:


years = []
months = []
for timestamp in timestamps:
    splitted = timestamp.split("-")
    years.append(splitted[0])
    months.append(splitted[1])

# create a dataframe from months and years lists
df = {'Month':months,'Year':years}
df = pd.DataFrame(df)
print(df)


# In[9]:


# aggregate samples of the same months of the same years
# fisrt define unique values of years and months sorted
seq = []
mon = []
y = []
year_values = list(set(df["Year"]))
year_values = sorted(year_values)
#print(year_values)
month_values = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

for year in year_values:
    df_year = df[ df['Year'] == year ]
    #month_values = list(set(df_year["Month"]))
    #month_values = sorted(month_values)
    for month in month_values:
        df_month = df_year[ df_year['Month'] == month ]
        seq.append(len(df_month))
        mon.append(month)
        y.append(year)


# In[10]:


dataset = {'Month':mon,'Year':y,'Vulnerabilities':seq}
dataset = pd.DataFrame(dataset)

# remove months outside the reported period
print(dataset)
for i in range(0, len(dataset['Vulnerabilities'])):
    vuln = dataset['Vulnerabilities'][i]
    if vuln > 0:
        index = i
        break
dataset = dataset.drop(list(range(0, index)))
dataset = dataset.reset_index()
#print(dataset)
for i in range(0, len(dataset['Vulnerabilities'])):
    vuln = dataset['Vulnerabilities'][i]
    if vuln > 0:
        index = i
if index < len(dataset)-1:
    dataset = dataset.drop(list(range(index+1, len(dataset))))
print(dataset)
dates = dataset["Month"] + "-" + dataset["Year"]
dataset["Date"] = dates


# In[49]:


from datetime import datetime
dataset['Datetime'] = pd.to_datetime(dataset['Date'],format ='%m-%Y')

dataset = dataset.set_index('Datetime')

print(dataset)

import matplotlib.pyplot as plt
dataset['Vulnerabilities'].plot(figsize=(16, 6), fontsize=15)
plt.xlabel("Datetime")
plt.ylabel("Vulnerabilities")
plt.title("Time Series Plot")
plt.show()

dataset["Vulnerabilities"].to_csv(software+".csv")
print("Number of months: ",len(dataset))
print("Number of vulnerabilities: ",len(file_data))


# In[50]:


vulns = dataset["Vulnerabilities"].tolist()
allVulns = sum(vulns)
print(allVulns)


# In[51]:


dataset['Vulnerabilities'].describe()


# In[13]:



boxplot = dataset.boxplot(column=['Vulnerabilities'])


# In[15]:


histogram = dataset.hist(column=['Vulnerabilities'],bins=10)


# In[ ]:




