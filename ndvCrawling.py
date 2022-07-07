# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 16:35:19 2022

@author: iliaskaloup
"""

import pandas as pd
import json
import csv
import os
from dataCleansing import groupMonths

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

software = "Chrome"
keyword = "google chrome"

myList = []
for i in range(0, len(file_data)):
    items = file_data[i]['CVE_Items']
    for item in items:
        description = item['cve']['description']['description_data'][0]['value']
        if keyword in description.lower():
            myList.append(item)

filename = "allVulns" + software + ".json"
with open(filename, 'w') as json_file:
    json.dump(myList, json_file, indent=4)

dataset = groupMonths(software)
print("Number of months: ",len(dataset))
print("Number of vulnerabilities: ",len(myList))