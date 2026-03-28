#!/usr/bin/env python
# coding: utf-8

# In[1]:

from numpy import *
import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from scipy.signal import hilbert
from scipy import signal
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
import sys
import time
import matplotlib
import IPython
matplotlib.pyplot.ion()
from numpy.linalg import multi_dot
from dataclasses import dataclass, field
import gc

seterr(divide='ignore', invalid='ignore') # ignore 0 division warnings
#warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

#############################################
############### EDIT LINES BELOW ############

### Set working directory. This is where the receng.py file is located
new_wd="/Users/papa/OneDrive - McGill University/Documents/Projects/RecEng/recengRealclass"
os.chdir(new_wd)
os.getcwd()

### set the 'scenario' subfolder where parameters are stored in 'param' folder. This also
### where results will be stored in 'data' folder'
 
subfolderName='auto_eng_fachab'


### set the batch of simulations adjusting for the effect of differnet level of recycling:
# [0,0,0,1] adjust only detritus input when decomposition is present,[1,0,0,1] adjusts for both nutrient and detritus input 
# when there is no recycling at all, [0,0,0,0] makes no adjustment
k_scenario=array([[0,0,0,1],[1,0,0,1],[0,0,0,0]]) # set Iadj values; each subarray must have length=v.ecodim
#k_scenario=array([0,0,0,0]) # run a single set without adjustment


#### NOTHING NEEDS TO CHANGE BELOW THIS LINE ######
###################################################

from recengFunctions import RecengClass # load the RecengClass (recengFunctions.py should be in new_wd)

### create a RecengClass object and initialize derived variables
v=RecengClass(subfolderName)
v.der_var()

### display parameter values 
with pd.option_context('display.width', 160,'display.max_rows', None,'display.max_columns', None,'display.precision', v.ecodim,): display(v.pr)
with pd.option_context('display.width', 160,'display.max_rows', None,'display.max_columns', None,'display.precision', v.ecodim,): display(v.pr_sp)
print('hoiloss','\n',v.hoiloss,'\n') #non-trophic effect on losses
print('troph','\n',v.troph,'\n') # trophic interactions
print('hoihab','\n',v.hoihab,'\n') # non-trophic habitat dependence
print('hoitroph','\n',v.hoitroph,'\n') #non-trophic effects on trophic interactions

v.open_sp_file('receng')

xaxis=linspace(v.pr['xVal'][0],v.pr['xVal'][1],int(v.pr.resL[0]))
series=linspace(v.pr['serVal'][0],v.pr['serVal'][1],int(v.pr.resS[0]))
liveDB=[]



# loop over recycling adjustment scenarios
for k in range(len(k_scenario)):
    print('scenario: ', k_scenario[k],'\n')
    v.pr.Iadj=k_scenario[k]
  
# loop over parameter space setting the x axis and different series
    for i in series:
        for j in xaxis:
            print('xaxis=',j,'series=',i)

            v.eta[:,2]=j # x axis variable; set column index of the focal species with habitat dependence
            v.hoiloss[:,3,1]=i # series; define compartments invovled in engineering; indices are patch,recipient,effector
            v.pr.Iadj=k_scenario[k] # scenarios for recycling; set which compartments have adjusted inputs

            v.dataReceng=array(v.receng(),dtype=object) # launch main simulation
                    
            v.der_var() # reset derived variables before creating output arrays          
            v.dataArray=append(array([k_scenario[k],v.connect.flatten('F'),v.flow,v.hoihab.flatten('F'),v.hoiloss.flatten('F'),v.hoidrec.flatten('F'),v.hoitroph.flatten('F'),v.troph.flatten('F'),[repeat(j,v.ecodim,axis=0)],[repeat(i,v.ecodim,axis=0)]],dtype=object),array(v.dataReceng,dtype=object))         
            v.dataLbl=['scenario','v.connect','flow','hoihab','hoiloss','hoidrec','hoitroph','troph',v.pr['varName'][0],v.pr['varName'][1],'mXctr','minCtr','maxCtr','Xstoch','Xcov','mXadj','minAdj','maxAdj','XstochAdj','XcovAdj','jacAll','jacAllAdj']          
            v.dataDB=v.out_combine(v.out_par(),v.out_var())           
            liveDB=pd.concat([pd.DataFrame(liveDB),pd.DataFrame(v.dataDB)])
            
            del v # create new instance of v to reinitialize variables
            v=RecengClass(subfolderName) 
            v.der_var()

# write results to file        

if v.bigD==1:
    v.out_write(liveDB,''.join([v.subfolder,'/data/',v.datafile]))  
    v.out_lbl(liveDB,'_'.join([v.pr['varName'][1],'0']),'_'.join(['mXctr','0']),''.join([v.subfolder,'/data/','plot_par.csv']),''.join([v.subfolder,'/data/','plot_var.csv']))

gc.collect() # forces emptying the memory
print('haza') 




