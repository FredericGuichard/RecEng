#!/usr/bin/env python
# coding: utf-8

# In[43]:


from numpy import *
from scipy import integrate
from scipy.optimize import fsolve
from scipy.signal import hilbert
import pandas as pd
from scipy.signal import argrelextrema
import pylab as p
import matplotlib.pyplot as plt
import math
import seaborn as sns
from functools import wraps
from statistics import covariance
import numdifftools as nd
import sdeint
from scipy.signal import argrelextrema
from random import Random
import os
os.getcwd()
import sys
import time
import matplotlib
import IPython
matplotlib.pyplot.ion()
from numpy.linalg import multi_dot


# In[44]:


##############################################
##### Query Data File ####################
##############################################

def query(newFilter=pd.DataFrame(),valGrp=['eg_0','eta_0'],prfile='plot_par.csv',varfile='plot_var.csv'):

    Bigdata=pd.read_csv(''.join([path_data,'recengDB.csv']))
    pr=pd.read_csv(''.join([path_data,prfile]))
    vl=genfromtxt(''.join([path_data,varfile]),delimiter=',',dtype=str)
    
    valN=newFilter.columns.values
    for i in range(size(newFilter)): pr[valN[i]][0]=newFilter[valN[i]][0]
    #newFilter should be of the form pd.DataFrame({'var':[val],'var':[val],'var':[val]})
    
    keys=pr.columns.values.tolist()
    for i in range(size(valGrp)): keys.remove(valGrp[i])
    i1 = Bigdata.set_index(keys).index
    i2 = pr.set_index(keys).index
    DataFilter=Bigdata[i1.isin(i2)]
    GroupedD=DataFilter.groupby(valGrp)[vl].mean().reset_index()
    return GroupedD


# In[46]:


##################
###   PLOTS   ####
##################

### Detritivore stocks and stochastic stability ###


subfolder='det_stoch'

pd.options.mode.chained_assignment = None
bigD=1
bigJ=0

### spatial parameters
het=0
#connect=array([[-0.5,0.5],[0.5,-0.5]]) # between patches
connect=array([0])
#connect=array([[0,0],[0,0]]) # between patches
flow=array([0,0,0]) # for each compartment
patchnbr= len(connect)
ecodim= len(flow)#len(pr_sp['r'])
nutrient=0 # the nutrient compartment is the first position
detritus=ecodim-1 # the detritus compartment is at the last position


parfile='recengScenario.csv'
datafile='recengDB.csv'
#folder = ''.join(['./',subfolder,'/'])  
path_data=''.join(['./',subfolder,'/data/'])
path_params=''.join(['./',subfolder,'/params/'])
path_fig=''.join(['./',subfolder,'/fig/'])
isdir = os.path.isdir(path_fig)
if isdir==False: os.mkdir(path_fig)

lstyle=['-','--','-.']
transp=[1,0.2,0.1]
linewidth=[2,3,4]
clr=['r','k','b']

#yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
#order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series
#dfF=pd.DataFrame({'':[],'':[],'':[]})


#yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
fig, axs = plt.subplots(2,2,figsize=(18,15))


#order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series

target=1 # index of plotted species
var_ind=(target+1)*(1+ecodim)-ecodim

# first column

fltr=pd.DataFrame({'scenario_0':[0]})
gd=query(fltr)
yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series


for k in range(order[2]): # ctr vs adj
            for l in range(order[3]): # min,max
                st=0
                for m in unique(order[4]): # with or without engineering
                    axs[0][0].plot(gd[gd.eg_0==m].eta_0, gd[gd.eg_0==m]['_'.join([yVal[k][l],str(target)])],ls=lstyle[st],color=clr[k],alpha=transp[k],linewidth=linewidth[k]) 
                    axs[1][0].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]/sqrt(gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])])),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[1][0].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]/sqrt(gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])])),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    st+=1


for j in range(2):
    for i in range(2):
#         axs[0][j].set_ylim([4,6])
#         axs[1][j].set_ylim([0,0.002])
        axs[i][j].xaxis.label.set_size(32)
        axs[i][j].yaxis.label.set_size(32)
        axs[i][j].xaxis.set_tick_params(labelsize=24)
        axs[i][j].yaxis.set_tick_params(labelsize=24)

fig.savefig(''.join([path_fig,'receng_det_stoch_V_stab.pdf']),bbox_inches='tight')




# In[51]:


##################
###   PLOTS   ####
##################

##### detritivore-detritus correlations ########

pd.options.mode.chained_assignment = None
bigD=1
bigJ=0

### spatial parameters
het=0
#connect=array([[-0.5,0.5],[0.5,-0.5]]) # between patches
connect=array([0])
#connect=array([[0,0],[0,0]]) # between patches
flow=array([0,0,0,0]) # for each compartment
patchnbr= len(connect)
ecodim= len(flow)#len(pr_sp['r'])
nutrient=0 # the nutrient compartment is the first position
detritus=ecodim-1 # the detritus compartment is at the last position


parfile='recengScenario.csv'
datafile='recengDB.csv'
#folder = ''.join(['./',subfolder,'/'])  
path_data='./data/'
path_params='./params/'
path_fig='./fig/'
isdir = os.path.isdir(path_fig)
if isdir==False: os.mkdir(path_fig)

lstyle=['-','--','-.']
transp=[1,0.2,0.1]
linewidth=[2,3,4]
clr=['r','k','b']

fltr=pd.DataFrame({'scenario_0':[0]})
gd=query(fltr)

yVal=[gd.Xcov_5/sqrt(gd.Xcov_4*gd.Xcov_8),gd.XcovAdj_5/sqrt(gd.XcovAdj_4*gd.XcovAdj_8)] # detritus<->detritivore


var_ind=(target+1)*(1+ecodim)-ecodim

fig, axs = plt.subplots(1,1,figsize=(12,10))

axs.plot(gd[gd.eg_0==m].eta_0, yVal[0][gd.eg_0==m],ls='-',color='red',alpha=1,linewidth=3) 
axs.plot(gd[gd.eg_0==m].eta_0, yVal[1][gd.eg_0==m],ls='-',color='grey',alpha=1,linewidth=3) 
axs.xaxis.label.set_size(18)
axs.yaxis.label.set_size(16)
axs.xaxis.set_tick_params(labelsize=36)
axs.yaxis.set_tick_params(labelsize=36)
axs.set_ylim([-1,1])


fig.savefig(''.join([path_fig,'receng_det_stoch_VD_corr.pdf']),bbox_inches='tight')

# In[52]:





##################
###   PLOTS   ####
##################

### Consumer net uptake calculated on long-term stochastic stock and stochastic stability ###


#subfolder='auto_eng'

pd.options.mode.chained_assignment = None
bigD=1
bigJ=0

### spatial parameters
het=0
#connect=array([[-0.5,0.5],[0.5,-0.5]]) # between patches
connect=array([0])
#connect=array([[0,0],[0,0]]) # between patches
flow=array([0,0,0,0]) # for each compartment
patchnbr= len(connect)
ecodim= len(flow)#len(pr_sp['r'])
nutrient=0 # the nutrient compartment is the first position
detritus=ecodim-1 # the detritus compartment is at the last position


parfile='recengScenario.csv'
datafile='recengDB.csv'
#folder = ''.join(['./',subfolder,'/'])  
path_data='./data/'
path_params='./params/'
path_fig='./fig/'
isdir = os.path.isdir(path_fig)
if isdir==False: os.mkdir(path_fig)

lstyle=['-','--','-.']
transp=[1,0.2,0.1]
linewidth=[2,3,4]
clr=['r','k','b']

#yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
#order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series
#dfF=pd.DataFrame({'':[],'':[],'':[]})


#yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
fig, axs = plt.subplots(2,3,figsize=(24,12))


#order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series

target=2 # index of plotted species
var_ind=(target+1)*(1+ecodim)-ecodim

# first column (full recycling)

fltr=pd.DataFrame({'scenario_3':[0],'scenario_0':[0]})
gd=query(fltr)
yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]

order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series


for k in range(order[2]): # ctr vs adj
            for l in range(order[3]): # min,max
                st=0
                for m in unique(order[4]): # with or without engineering
                    Amean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(1)])]
                    AmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(1)])]
                    Dmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(3)])]
                    DmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(3)])]
                    Cmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]
                    CmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    Anot=0.5; lambdaC=3 # half-saturation of primary producer from receng0.csv file, and lambda_C from troph.csv: should be read directly from file!!
                    axs[0][0].plot(gd[gd.eg_0==m].eta_0,lambdaC*Cmean*(Amean/(Anot+Amean))*(1-(Cmean/(gd[gd.eg_0==m].eta_0*Dmean+Cmean))),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[0][0].plot(gd[gd.eg_0==m].eta_0,lambdaC*CmeanAdj*(AmeanAdj/(Anot+AmeanAdj))*(1-(CmeanAdj/(gd[gd.eg_0==m].eta_0*DmeanAdj+CmeanAdj))),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    axs[1][0].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]/gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])]),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[1][0].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]/gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])]),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    st+=1

# second column (decomposition only)

fltr=pd.DataFrame({'scenario_3':[1],'scenario_0':[0]})
gd=query(fltr)
yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series


for k in range(order[2]): # ctr vs adj
            for l in range(order[3]): # min,max
                st=0
                for m in unique(order[4]): # with or without engineering
                    Amean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(1)])]
                    AmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(1)])]
                    Dmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(3)])]
                    DmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(3)])]
                    Cmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]
                    CmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    Anot=0.5; lambdaC=3 # half-saturation of primary producer from receng0.csv file, and lambda_C from troph.csv: should be read directly from file!!
                    axs[0][1].plot(gd[gd.eg_0==m].eta_0,lambdaC*Cmean*(Amean/(Anot+Amean))*(1-(Cmean/(gd[gd.eg_0==m].eta_0*Dmean+Cmean))),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[0][1].plot(gd[gd.eg_0==m].eta_0,lambdaC*CmeanAdj*(AmeanAdj/(Anot+AmeanAdj))*(1-(CmeanAdj/(gd[gd.eg_0==m].eta_0*DmeanAdj+CmeanAdj))),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    axs[1][1].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]/gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])]),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[1][1].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]/gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])]),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    st+=1
#third column (no recycling)

fltr=pd.DataFrame({'scenario_3':[1],'scenario_0':[1]})
gd=query(fltr)
yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series


for k in range(order[2]): # ctr vs adj
            for l in range(order[3]): # min,max
                st=0
                for m in unique(order[4]): # with or without engineering
                    Amean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(1)])]
                    AmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(1)])]
                    Dmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(3)])]
                    DmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(3)])]
                    Cmean=gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]
                    CmeanAdj=gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]
                    Anot=0.5; lambdaC=3 # half-saturation of primary producer from receng0.csv file, and lambda_C from troph.csv: should be read directly from file!!
                    axs[0][2].plot(gd[gd.eg_0==m].eta_0,lambdaC*Cmean*(Amean/(Anot+Amean))*(1-(Cmean/(gd[gd.eg_0==m].eta_0*Dmean+Cmean))),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[0][2].plot(gd[gd.eg_0==m].eta_0,lambdaC*CmeanAdj*(AmeanAdj/(Anot+AmeanAdj))*(1-(CmeanAdj/(gd[gd.eg_0==m].eta_0*DmeanAdj+CmeanAdj))),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    axs[1][2].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['Xstoch',str(target)])]/gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])]),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[1][2].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['XstochAdj',str(target)])]/gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])]),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    st+=1



for j in range(3):
    for i in range(2):
       # axs[0][j].set_ylim([0,2])
        axs[1][j].set_ylim([0,210000000000])
        axs[i][j].xaxis.label.set_size(18)
        axs[i][j].yaxis.label.set_size(16)
        axs[i][j].xaxis.set_tick_params(labelsize=16)
        axs[i][j].yaxis.set_tick_params(labelsize=16)

fig.savefig(''.join([path_fig,'receng_C_stoch_stab.pdf']),bbox_inches='tight')



