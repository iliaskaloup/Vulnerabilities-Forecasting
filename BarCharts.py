#!/usr/bin/env python
# coding: utf-8

# In[4]:


# -*- coding: utf-8 -*-
"""
@author: iliaskaloup
"""

import numpy as np
import matplotlib.pyplot as plt
 
# set width of bar
barWidth = 0.15
 
# set height of bar
mae_stat = [14.027, 7.051, 20.555, 1.293, 1.208]
mae_dl = [13.955, 3.265, 19.810, 1.272, 1.065]
 
# Set position of bar on X axis
br1 = np.arange(len(mae_stat))
br2 = [x + barWidth for x in br1]
 
# Make the plot
with plt.style.context('bmh'):
    fig = plt.subplots(figsize =(12, 8))
    plt.bar(br1, mae_stat, color ='firebrick', width = barWidth,
            edgecolor ='grey', label ='Best Statistical Model')
    plt.bar(br2, mae_dl, width = barWidth,
            edgecolor ='grey', label ='Best Deep Learning Model')


    # Adding Xticks
    plt.title('Vulnerabilities Forecast 24 steps ahead', fontweight ='bold', fontsize = 14)
    plt.xlabel('Software Project', fontweight ='bold', fontsize = 14)
    plt.ylabel('Mean Absolute Error', fontweight ='bold', fontsize = 14)
    plt.xticks([r + 1.5*barWidth for r in range(len(mae_stat))],
            ['google_chrome', 'microsoft_internet_explorer', 'apple_mac_os_x', 'canonical_ubuntu_linux', 'microsoft_office'])

    plt.legend()
    plt.show()









# In[5]:


# set width of bar
barWidth = 0.15
 
# set height of bar
rmse_stat = [17.191, 7.679, 32.042, 1.965, 1.617]
rmse_dl = [15.888, 3.859, 31.689, 1.964, 1.529]
 
# Set position of bar on X axis
br1 = np.arange(len(rmse_stat))
br2 = [x + barWidth for x in br1]
 
# Make the plot
with plt.style.context('bmh'):
    fig = plt.subplots(figsize =(12, 8))
    plt.bar(br1, rmse_stat, color ='firebrick', width = barWidth,
            edgecolor ='grey', label ='Best Statistical Model')
    plt.bar(br2, rmse_dl, width = barWidth,
            edgecolor ='grey', label ='Best Deep Learning Model')


    # Adding Xticks
    plt.title('Vulnerabilities Forecast 24 steps ahead', fontweight ='bold', fontsize = 14)
    plt.xlabel('Software Project', fontweight ='bold', fontsize = 14)
    plt.ylabel('Root Mean Square Error', fontweight ='bold', fontsize = 14)
    plt.xticks([r + 1.5*barWidth for r in range(len(rmse_stat))],
            ['google_chrome', 'microsoft_internet_explorer', 'apple_mac_os_x', 'canonical_ubuntu_linux', 'microsoft_office'])

    plt.legend()
    plt.show()

