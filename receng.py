#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
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
os.getcwd()
import sys
import time
import matplotlib
import IPython
matplotlib.pyplot.ion()
from numpy.linalg import multi_dot


# In[5]:


#################
### FUNCTIONS ###
#################

#####################
#### DYNAMICS #######
#####################

def constrain(constraints):
    if all(constraint is not None for constraint in constraints):
         assert constraints[0] < constraints[1]
    def wrap(f):
        @wraps(f)
        def wrapper(t, y, *args, **kwargs):
            lower, upper = constraints
            if lower is None:
                lower = -inf
            if upper is None:
                 upper = inf
                    
            too_low = y <= lower
            too_high = y >= upper

            y = maximum(y, ones(shape(y))*lower)
            y = minimum(y, ones(shape(y))*upper)

            result = f(t, y, *args, **kwargs)

            result[too_low] = maximum(result[too_low], ones(too_low.sum())*lower)
            result[too_high] = minimum(result[too_high], ones(too_high.sum())*upper)        
            return result
        return wrapper
    return wrap


### Functional responses ###

def frC(con,X,pn): # return vector with functional responses of each species as a resouece
    return X[con]*(X/(resnot[pn]+X))

def frR(res,X,pn):
    return X*(X[res]/(resnot[pn,res]+X[res]))


def hoiVV(foc,form,X,pn):
    match form:
        case 'hab': # returns an vector with total habitat effect from each habitat on focal species
            #return 1-((hoihab[pn,foc,:]*sum(hoihab[pn]*X))/(sum(hoihab[pn]*X)+eta[foc]*X))
            #return 1-((hoihab[pn,foc,:]*X[foc])/(X[foc]+eta[pn,foc]*hoihab[pn,foc,:]*X))
            hab_vector=1-((hoihab[pn,foc,:]*X[foc])/(X[foc]+eta[pn,foc]*hoihab[pn,foc,:]*X))
            return prod(hab_vector[hab_vector>0])
        case 'troph':
            return (hoitroph[pn,foc,:]*X+epsi[pn])/(X+epsi[pn])
        case 'loss':
            return (hoiloss[pn,foc,:]*X+epsi[pn])/(X+epsi[pn])
        case 'drec':
            return (hoidrec[pn,foc,:]*X+epsi[pn])/(X+epsi[pn])

### Additive Terms ###        

def trophicVV(foc,X,pn): # return difference between 2 scalars (growth-consumption) of the focal species
    #growth=troph[pn,foc,:]*frC(foc,X,pn)*product(hoiVV(foc,'hab',X,pn)[hoiVV(foc,'hab',X,pn)>0])*product(hoiVV(foc,'troph',X,pn)[hoiVV(foc,'troph',X,pn)>0])
    growth=troph[pn,foc,:]*frC(foc,X,pn)*hoiVV(foc,'hab',X,pn)*prod(hoiVV(foc,'troph',X,pn)[hoiVV(foc,'troph',X,pn)>0])
    consumption=troph[pn,:,foc]*frR(foc,X,pn)
    for eng in range(ecodim):
        #consumption[eng]=consumption[eng]*product(hoiVV(eng,'hab',X,pn)[hoiVV(eng,'hab',X,pn)>0])*product(hoiVV(eng,'troph',X,pn)[hoiVV(eng,'troph',X,pn)>0])
        consumption[eng]=consumption[eng]*hoiVV(eng,'hab',X,pn)*prod(hoiVV(eng,'troph',X,pn)[hoiVV(eng,'troph',X,pn)>0])
    return sum(growth)-sum(consumption)

def lossV(foc,X,pn): # return scalar of modified loss of focal species
    return X[foc]*m[pn,foc]*prod(hoiVV(foc,'loss',X,pn)[hoiVV(foc,'loss',X,pn)>0])

def recyclingV(foc,X,pn):
    recTot=0
    if foc==detritus: #recTot=sum(delta[pn]*lossV(foc,X,pn))
        for sp in range(ecodim):
            rec=delta[pn,sp]*lossV(sp,X,pn)
            recTot+=rec
    return recTot

def direct_recyclingV(foc,X,pn):
    dRecTot=0
    if foc==nutrient:      
        for sp in range(ecodim):
            dRec=r[pn,sp]*X[sp]*prod(hoiVV(sp,'drec',X,pn))
            dRecTot+=dRec
    else:
        dRec= -r[pn,foc]*X[foc]*prod(hoiVV(foc,'drec',X,pn))
        dRecTot+=dRec        
    return dRecTot


### Spatialization ###

def X_byPatch(X): # take vector F and return 2D array of local ecosystem vectors for each patch
    localdyn=zeros((patchnbr,ecodim))
    comp=concatenate([[i*patchnbr] for i in range(ecodim)])
    compfull= concatenate([comp+i for i in range(patchnbr)])
    Xl=X[compfull]
    return Xl.reshape((patchnbr,ecodim))


def localDyn(X): # vector F with homogeneous ecosystem types  
    localdyn=zeros((patchnbr,ecodim))
    Xl=X_byPatch(X)
    for p in range(patchnbr):
        for c in range(ecodim):
            localdyn[p,c]=trophicVV(c,Xl[p,:],p)-lossV(c,Xl[p,:],p)+recyclingV(c,Xl[p,:],p)+direct_recyclingV(c,Xl[p,:],p)+I[p,c]
    return localdyn.flatten('F')


def connectivity(): # matrix C
    mshape=[patchnbr*ecodim,patchnbr*ecodim]
    comp_connect=transpose(connect)
    conn = zeros(mshape)
    for i in range(ecodim): conn[i*patchnbr:i*patchnbr+patchnbr,i*patchnbr:i*patchnbr+patchnbr]=comp_connect
 #   print('C matrix:',shape(conn))
    return conn

def flows(): # matrix Q
    comp_flow=flow*diag(zeros(ecodim)+1)
    land=diag(zeros(patchnbr)+1)
 #   print('Q matrix:',shape(kron(comp_flow,land)))
    return kron(comp_flow,land)

def dMdf_dt(t,X):
    return localDyn(X)+multi_dot([flows(),connectivity(),X])


### STOCHASTIC DYNAMICS ####

def G(x,t):
    global B
    #return stddev[0]*sqrt(B) # demographic stchasticity
    return stddev[0]*B  # Environmenal stochastiity

def fdf(x,t):  
    return localDyn(x)+multi_dot([flows(),connectivity(),x])

##############################
##### Stability Analysis #####
##############################


def highpass(data: ndarray, cutoff: float, sample_rate: float, poles: int = 5):
    sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered_data = scipy.signal.sosfiltfilt(sos, data)
    return filtered_data

def localjac(X,pnbr):
    localdyn=zeros(ecodim)
    for c in range(ecodim):
        localdyn[c]=(trophicVV(c,X,pnbr)-lossV(c,X,pnbr))+recyclingV(c,X,pnbr)+direct_recyclingV(c,X,pnbr)+I[pnbr,c]
    return localdyn

def jacEq(Xeq,pnbr):
    Xl=X_byPatch(Xeq)
    fun= lambda X: localjac(X,pnbr) 
    jac = nd.Jacobian(fun)(Xl[pnbr,:])
    return jac.flatten('F')

def minmax(xseries): 
    df = pd.DataFrame(xseries, columns=['data'])
    n = 10  # number of points to be checked before and after
    # Find local peaks
    df['min'] = df.iloc[argrelextrema(df.data.values, less_equal,
                        order=n)[0]]['data']
    df['max'] = df.iloc[argrelextrema(df.data.values, greater_equal,
                        order=n)[0]]['data']   
    return min(df['min'].dropna()),max(df['max'].dropna())

########################
### Output Functions ###
########################

def out_par(parfile,dim):
    prf=pd.read_csv(''.join([path_params,parfile]))
    vname=[]
    for col in prf.columns:
        for row in range(dim):
            name=''.join([col,str(row)])
            vname.append(name)
    parval=prf.to_numpy().flatten('F')
    valdf=pd.DataFrame(parval).T
    valdf.columns=vname
    return valdf

def out_var(data,lbl):
    dataSize=[]
    for i in range(len(data)): dataSize.append(size(data[i]))
    vlbl=[];k=0    
    for col in lbl:
        for row in range(dataSize[k]):
            name='_'.join([str(col),str(row)])
            vlbl.append(name)
        k+=1
    datafr=[]
    for g in range(len(data)):
        datafr=concatenate([datafr,array(data[g]).flatten('F')],axis=0)
        #datafr=pd.concat([pd.DataFrame(datafr),pd.DataFrame(data[g])],axis=1)
    datafr=pd.DataFrame(datafr).T
    datafr.columns=vlbl
    for i in range(1,ecodim):
        datafr=datafr.drop('_'.join(['eta',str(i)]),axis=1)
        datafr=datafr.drop('_'.join(['eg',str(i)]),axis=1)
    return pd.DataFrame(datafr)

def out_combine(par,data):
    return pd.concat([par,data],axis=1)

def out_write_OFF(data,file):
    fileExist=os.path.isfile(file)
    if fileExist:
        current_db=pd.read_csv(file)
        merged_db=pd.concat([current_db,data],join='outer',axis=0)
    else: merged_db=data
    with open(file, 'w') as out: #  out.write('\n')
        merged_db.to_csv(file,mode='w',index=False,header=True)

def out_write(data,file): # never append new data to any exisitng file
    merged_db=data
    with open(file, 'w') as out: #  out.write('\n')
        merged_db.to_csv(file,mode='w',index=False,header=True)

def out_lbl(data,last_param,first_var,prfile,varfile):
    param_col=pd.DataFrame(data.iloc[0,0:data.columns.get_loc(last_param)+1]).T
    with open(prfile, 'w') as out:
        param_col.to_csv(prfile,mode='w',index=False,header=True)
    var_col=pd.DataFrame(data.iloc[:,data.columns.get_loc(first_var):].columns.values).T
    with open(varfile, 'w') as out:
        var_col.to_csv(varfile,mode='w',index=False,header=False)
    
            
#####################
####PARAMETERS#######
#####################
def params(save=0,read=1,fname='receng_params.csv'):
    directory = ''.join([folder,'params/',])
    return pd.read_csv(os.path.join(directory,fname))

def setParams(fromFile=1,fnbr=0):
    if fromFile==1: prSim=params(read=1,fname=pfiles['name'][fnbr])
    else: prSim=params()
    return prSim

def genParams(val=[],pspace=[],fromfile=1,filen='paramfile',append=0):
    directory = ''.join([folder,'params/',])
    if append==1: pfiles=pd.read_csv(os.path.join(directory,'receng_parspace.csv'))
    else: pfiles=pd.DataFrame()
    par=params(save=0,read=fromfile,fname=filen)
    print(len(val),len(pspace))
    for i in range(len(val)):
        for j in range(len(pspace)) :
            if len(pspace)==1: par[pspace[j]]=val[i]
            else: par[pspace[j]]=val[i][j]
        timestr = time.strftime("%Y%m%d-%H%M%S")
        filename=''.join(['receng_params',timestr,str(i),'.csv'])        
        file_path = os.path.join(directory,filename)
        par.to_csv(file_path)
        fdf=pd.DataFrame({'name':pd.Series(filename)})
        pfiles=pd.concat([pfiles,fdf], ignore_index=True)
    if append==0: open('receng_parspace.csv','w')
    pfiles.to_csv('receng_parspace.csv')

def open_matrix_file(matrice):
    match het:
        case 1:           
            matlist=empty((patchnbr,ecodim,ecodim))
            for i in range(patchnbr):   
                filename=''.join([folder,'params/',matrice,str(i),'.csv'])
                matlist[i]=genfromtxt(filename,delimiter=',')
        case 0:           
            matlist=empty((patchnbr,ecodim,ecodim))
            for i in range(patchnbr):   
                filename=''.join([folder,'params/',matrice,'0','.csv'])
                matlist[i]=genfromtxt(filename,delimiter=',')               
    return matlist

def save_matrix_file(matrice):
    match het:
        case 1:           
            for i in range(patchnbr):   
                filename=''.join([folder,'params/',matrice,str(i),'.csv'])
                savetxt(filename,matrice,delimiter=',')
        case 0:           
            for i in range(patchnbr):   
                filename=''.join([folder,'params/',matrice,'0','.csv'])
                savetxt(filename,matrice,delimiter=',')               
    return

def open_sp_file(matrice):

    prlengthFile=pd.read_csv(''.join([folder,'params/',matrice,'0','.csv']))
    prlength=len(prlengthFile.columns)
    varNames=prlengthFile.columns.values
    match het:
        case 1:           
            for j in range(prlength):
                myStr =varNames[j]
                globals()[varNames[j]]=[]
                for i in range(patchnbr):
                    filename=''.join([folder,'params/',matrice,str(i),'.csv'])
                    df=pd.read_csv(filename) 
                    globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                globals()[varNames[j]]=reshape(globals()[varNames[j]],(patchnbr,ecodim))       
        case 0:           
            for j in range(prlength):
                myStr =varNames[j]
                globals()[varNames[j]]=[]
                for i in range(patchnbr):
                    filename=''.join([folder,'params/',matrice,'0','.csv'])
                    df=pd.read_csv(filename) 
                    globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                globals()[varNames[j]]=reshape(globals()[varNames[j]],(patchnbr,ecodim))

    return

def save_sp_file(matrice):

    prlengthFile=pd.read_csv(''.join([folder,'params/',matrice,'0','.csv']))
    prlength=len(prlengthFile.columns)
    varNames=prlengthFile.columns.values
    match het:
        case 1:           
            for j in range(prlength):
                myStr =varNames[j]
                globals()[varNames[j]]=[]
                for i in range(patchnbr):
                    filename=''.join([folder,'params/',matrice,str(i),'.csv'])
                    df=pd.read_csv(filename) 
                    globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                globals()[varNames[j]]=reshape(globals()[varNames[j]],(patchnbr,ecodim))       
        case 0:           
            for j in range(prlength):
                myStr =varNames[j]
                globals()[varNames[j]]=[]
                for i in range(patchnbr):
                    filename=''.join([folder,'params/',matrice,'0','.csv'])
                    df=pd.read_csv(filename) 
                    globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                globals()[varNames[j]]=reshape(globals()[varNames[j]],(patchnbr,ecodim))

    return


##############################################
##### Query Data File ####################
##############################################

def query(newFilter=pd.DataFrame(),valGrp=['eg_0','eta_0'],prfile='plot_par.csv',varfile='plot_var.csv'):

    Bigdata=pd.read_csv(''.join([folder,'data/','recengDB.csv']))
    pr=pd.read_csv(''.join([folder,'data/',prfile]))
    vl=genfromtxt(''.join([folder,'data/',varfile]),delimiter=',',dtype=str)
    
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

############################################
####### Colonisation and extinction ########
############################################

def array_add(some_array, array_index):
    return some_array

def array_drop(somme_array_index):
    return some_array

def ext_test(stock_array):
    ext_thresh=0.0000001
    ext_index=where(stock_array<ext_thresh)
    return ext_index

def extinction(ext_index):
    hoihab=open_matrix_file('hoihab')
    hoiloss=open_matrix_file('hoiloss')
    hoidrec=open_matrix_file('hoidrec')
    hoitroph=open_matrix_file('hoitroph')
    troph=open_matrix_file('troph')
    open_sp_file('receng')
    
    hoihab=delete(hoihab,ext_index)
    hoiloss=delete(hoiloss,ext_index)
    hoidrec=delete(hoidrec,ext_index)
    hoitroph=delete(hoitroph,ext_index)
    troph=delete(troph,ext_index)
    
    savetxt('hoihab.csv',hoihab,delimiter=',')
    savetxt('hoiloss.csv',hoiloss,delimiter=',')
    savetxt('hoidrec.csv',hoidrec,delimiter=',')
    savetxt('hoitroph.csv',hoitroph,delimiter=',')
    savetxt('troph.csv',torph,delimiter=',')
    
    
    

#############################################
####### Main Simulation function ############
#############################################

def receng():

    global eta,delta,r,m,I,epsi,resnot,stddev,frtype,B,pr,hoihab,hoiloss,hoidrec,hoitroph,troph   
    liveDB=[]
   
    X0=zeros(ecodim*patchnbr)
    for g in range(ecodim*patchnbr):
        X0[g]=random.uniform(0.1,2)
    X = integrate.solve_ivp(dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
    tr=X.t
    trans=int(len(X.y[0])/4)
    mX=mean(X.y[:,trans:],axis=1)

    if sum(pr.Iadj)>0: 
        if pr.Iadj[detritus]==1: # turn off recycling and adjust the detritus input
            deltaOld=delta
            delta=zeros((patchnbr,ecodim))
            I[:,detritus]+=sum(deltaOld*m*mX) 

        if pr.Iadj[nutrient]==1: # turn off direct recycling and adjust the nutrient input in the isolated green web
            rOld=r
            r=zeros((patchnbr,ecodim))
            I[:,nutrient]+=sum(rOld*mX)

        X0=zeros(ecodim*patchnbr)
        for g in range(ecodim*patchnbr):
            X0[g]=random.uniform(0.1,2)
        X = integrate.solve_ivp(dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
        tr=X.t
        trans=int(len(X.y[0])/4)
        mX=mean(X.y[:,trans:],axis=1)

    minTemp=[];maxTemp=[]
    for g in range(ecodim*patchnbr):
        minX,maxX=minmax(X.y[g][int(-len(t)/2):-1])
        minTemp.append(minX),maxTemp.append(maxX)
    minCtr.append(minTemp);maxCtr.append(maxTemp)
    mXctr.append(mX)  # storing either mean stock values ( with Iadj= 0 or 1) 


    if pr.jac[0]==1:
        jacAll=jacEq(mX,0)

    if sum(stddev)>0:
        stoch=zeros(ecodim*patchnbr)+1
        B = stoch*diag(mX)
        sde = sdeint.itoint(fdf, G, mX, t)
        trans=int(len(t)/4)
        sdestn=sde[trans:,0:]         
        #sdestn = signal.sosfilt(sos, sdestn, axis=-1) # apply sos high-pass filter to stochastic time series
        sdecov=cov(transpose(sdestn))
        sdemean=mean(sdestn,axis=0)         
        Xcov.append(sdecov.flatten())
        Xstoch.append(transpose(sdemean))

    if sum(pr.lmdaAdj)>0:
        mXl=X_byPatch(mX)
        for pnumb in range(patchnbr):
            for sp in range(ecodim):
                troph[pnumb,sp,:]=troph[pnumb,sp,:]*hoiVV(sp,'hab',mXl[pnumb],pnumb)*hoiVV(sp,'troph',mXl[pnumb],pnumb)
                m[pnumb,sp]=m[pnumb,sp]*prod(hoiVV(sp,'loss',mXl[pnumb],pnumb)[hoiVV(sp,'loss',mXl[pnumb],pnumb)>0])
                r[pnumb,sp]=r[pnumb,sp]*prod(hoiVV(sp,'drec',mXl[pnumb],pnumb)[hoiVV(sp,'drec',mXl[pnumb],pnumb)>0])          

        hoitroph=zeros((patchnbr,ecodim,ecodim))+1
        hoihab=zeros((patchnbr,ecodim,ecodim))
        hoiloss=zeros((patchnbr,ecodim,ecodim))+1
        hoidrec=zeros((patchnbr,ecodim,ecodim))+1

        X0=zeros(ecodim*patchnbr)
        for g in range(ecodim*patchnbr):
            X0[g]=random.uniform(0.1,2)
        X = integrate.solve_ivp(dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
        tr=X.t
        trans=int(len(X.y[0])/4)
        mX=mean(X.y[:,trans:],axis=1)

        minTemp=[];maxTemp=[]
        for g in range(ecodim*patchnbr):
            minX,maxX=minmax(X.y[g][int(-len(t)/2):-1])
            minTemp.append(minX),maxTemp.append(maxX)
        minAdj.append(minTemp);maxAdj.append(maxTemp) 
        mXadj.append(mX)    

        if pr.jac[0]==1:
            jacAllAdj=jacEq(mX,0)  

        if sum(stddev)>0:
            stoch=zeros(ecodim*patchnbr)+1
            B = stoch*diag(mX)
            sde = sdeint.itoint(fdf, G, mX, t)
            trans=int(len(t)/4)
            sdestn=sde[trans:,0:]
            #savetxt('time_series.txt', array(sdestn)) # save adjusted stocahstic time series to file
            #sdestn = signal.sosfilt(sos, sdestn, axis=-1) # apply sos high-pass filter to stochastic time series
            sdecov=cov(transpose(sdestn))
            sdemean=mean(sdestn,axis=0)         
            XcovAdj.append(sdecov.flatten())
            XstochAdj.append(transpose(sdemean))
            
    dataArray=[mXctr,minCtr,maxCtr,Xstoch,Xcov,mXadj,minAdj,maxAdj,XstochAdj,XcovAdj,jacAll,jacAllAdj]        
    return dataArray


# In[241]:


#######################
##### MAIN SCRIPT #####
#######################


# Call a specific scenario (folder name within working directory)
#subfolder='det_stoch'
subfolder='auto_eng_stoch'


bigD=1

### spatial parameters
het=0
#connect=array([[-0.5,0.5],[0.5,-0.5]]) # between patches
connect=array([0])
#connect=array([[0,0],[0,0]]) # between patches
flow=array([0,0,0,0]) # for each compartment -> size needs to correspond to number of compartments (ecodim)
patchnbr= len(connect)
ecodim= len(flow)#len(pr_sp['r'])
nutrient=0 # the nutrient compartment is the first position
detritus=ecodim-1 # the detritus compartment is at the last position
sos = signal.butter(4,0.25, 'highpass', output='sos') # define highpass filter applied to stochastic time series


parfile='recengScenario.csv'
datafile='recengDB.csv'
folder = ''.join(['./',subfolder,'/'])  
path_data=''.join([folder,'data/'])
isdir = os.path.isdir(path_data)
#if isdir==False: os.mkdir(path_data) 
path_params=''.join([folder,'params/'])
path_fig=''.join([folder,'fig'])
isdir = os.path.isdir(path_fig)
#if isdir==False: os.mkdir(path_fig)

pr=pd.read_csv(''.join([path_params,parfile]))
pr_sp=pd.read_csv(''.join([path_params,'receng0.csv']))
with pd.option_context('display.width', 160,'display.max_rows', None,'display.max_columns', None,'display.precision', ecodim,): display(pr)
with pd.option_context('display.width', 160,'display.max_rows', None,'display.max_columns', None,'display.precision', ecodim,): display(pr_sp)

open_sp_file('receng')

hoihab=open_matrix_file('hoihab')
hoiloss=open_matrix_file('hoiloss')
hoidrec=open_matrix_file('hoidrec')
hoitroph=open_matrix_file('hoitroph')
troph=open_matrix_file('troph')

print('hoiloss','\n',hoiloss,'\n')
print('troph','\n',troph,'\n')


seterr(divide='ignore', invalid='ignore') # ignore 0 division warnings
#warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

minX=[];maxX=[];minAdj=[];maxAdj=[];minCtr=[];maxCtr=[];mXctr=[];mXadj=[];Xstoch=[];Xcov=[];XstochAdj=[];XcovAdj=[];liveDB=[]

jacAll=[];jacAllAdj=[] # that won't work with the current code

t = linspace(0,5000,10000) # changed from 10000, 50000

xaxis=linspace(pr['xVal'][0],pr['xVal'][1],int(pr.resL[0]))
series=linspace(pr['serVal'][0],pr['serVal'][1],int(pr.resS[0]))
liveDB=[]


k_scenario=array([[0,0,0,1],[1,0,0,1],[0,0,0,0]]) # set Iadj values; each subarray must have length=ecodim
#k_scenario=array([0,0,0,0])

for k in range(len(k_scenario)):
    pr.Iadj=k_scenario[k]
    
    for i in series:
        for j in xaxis:
            print('xaxis=',j,'series=',i)
            lmdaAdj=[];rAdj=[];mAdj=[];minX=[];maxX=[];minAdj=[];maxAdj=[];minCtr=[];maxCtr=[];mXctr=[];mXadj=[];Xstoch=[];Xcov=[];XstochAdj=[];XcovAdj=[]

            eta[:,2]=j  # column index of the focal species with habitat dependence
            hoiloss[:,3,1]=i # patch,recipient,effector
            pr.Iadj=k_scenario[k]
            
            dataReceng=array(receng(),dtype=object)
            
            pr=pd.read_csv(''.join([path_params,parfile]))
            open_sp_file('receng')
            hoihab=open_matrix_file('hoihab')
            hoiloss=open_matrix_file('hoiloss')
            hoidrec=open_matrix_file('hoidrec')
            hoitroph=open_matrix_file('hoitroph')
            troph=open_matrix_file('troph')
          
            dataArray=append(array([k_scenario[k],connect.flatten('F'),flow,hoihab.flatten('F'),hoiloss.flatten('F'),hoidrec.flatten('F'),hoitroph.flatten('F'),troph.flatten('F'),[repeat(j,ecodim,axis=0)],[repeat(i,ecodim,axis=0)]],dtype=object),array(dataReceng,dtype=object))
            dataLbl=['scenario','connect','flow','hoihab','hoiloss','hoidrec','hoitroph','troph',pr['varName'][0],pr['varName'][1],'mXctr','minCtr','maxCtr','Xstoch','Xcov','mXadj','minAdj','maxAdj','XstochAdj','XcovAdj','jacAll','jacAllAdj']
            dataDB=out_combine(out_par(parfile,ecodim),out_var(dataArray,dataLbl))
            
            liveDB=pd.concat([pd.DataFrame(liveDB),pd.DataFrame(dataDB)])

        
if bigD==1:
    out_write(liveDB,''.join([subfolder,'/data/',datafile]))  
    out_lbl(liveDB,'_'.join([pr['varName'][1],'0']),'_'.join(['mXctr','0']),''.join([subfolder,'/data/','plot_par.csv']),''.join([subfolder,'/data/','plot_var.csv']))
print('haza') 


# In[236]:


##################
###   PLOTS   ####
##################


subfolder='det_stoch'


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
folder = ''.join(['./',subfolder,'/'])  
path_data=''.join([folder,'data'])
path_params=''.join([folder,'params'])
path_fig=''.join([folder,'fig'])
isdir = os.path.isdir(path_fig)
if isdir==False: os.mkdir(path_fig)



fltr=pd.DataFrame({'scenario_3':[0],'scenario_0':[0]})
gd=query(fltr)

yVal=[['minCtr','maxCtr'],['minAdj','maxAdj']]
fig, axs = plt.subplots(ecodim,max(2,patchnbr),figsize=(14,28))

lstyle=['-','--','-.']
transp=[1,0.2,0.1]
linewidth=[2,3,4]
clr=['r','k','b']


order=[ecodim,patchnbr,len(yVal),len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2) sets of variables, (3)  variables, (4) series
#dfF=pd.DataFrame({'':[],'':[],'':[]})


ind=0
for i in range(order[0]): # row
    var_ind=(i+1)*(1+ecodim)-ecodim
    for j in range(order[1]):    # column 
        for k in range(order[2]): # ctr vs adj
            for l in range(order[3]): # min,max
                st=0
                for m in unique(order[4]): # with or without engineering
                    axs[i][j].plot(gd[gd.eg_0==m].eta_0, gd[gd.eg_0==m]['_'.join([yVal[k][l],str(ind)])],ls=lstyle[st],color=clr[k],alpha=transp[k],linewidth=linewidth[k]) 
                    axs[i][j+1].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['Xstoch',str(i)])]/gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])]),ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    axs[i][j+1].plot(gd[gd.eg_0==m].eta_0,(gd[gd.eg_0==m]['_'.join(['XstochAdj',str(i)])]/gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])]),ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    #axs[i][j+1].set_ylim([0,0.000015])
                    #axs[i][j+1].set_xlim([0.2,2.6])
                    #axs[i][j+1].plot(gd[gd.eg_0==m].eta_0,gd[gd.eg_0==m]['_'.join(['Xcov',str(var_ind-1)])],ls=lstyle[st],color=clr[0],alpha=transp[0],linewidth=linewidth[0])
                    #axs[i][j+1].plot(gd[gd.eg_0==m].eta_0,gd[gd.eg_0==m]['_'.join(['XcovAdj',str(var_ind-1)])],ls=lstyle[st],color=clr[1],alpha=transp[2],linewidth=linewidth[1])
                    st+=1
        #print(ind)
        ind+=1

for j in range(max(2,patchnbr)):
    for i in range(ecodim):
        axs[i][j].xaxis.label.set_size(18)
        axs[i][j].yaxis.label.set_size(16)
        axs[i][j].xaxis.set_tick_params(labelsize=16)
        axs[i][j].yaxis.set_tick_params(labelsize=16)

fig.savefig(''.join([path_fig,'receng_stoch_stab.pdf']),bbox_inches='tight')


# In[19]:


############################################
###   PLOTS Jacobians and correlations  ####
############################################

fltr=pd.DataFrame({'scenario_3':[0],'scenario_0':[0]})
gd=query(fltr)

#yVal=[[gd.jacAll_3*gd.jacAll_1,gd.jacAllAdj_3*gd.jacAllAdj_1], # from detritus to autotroph
#    [gd.jacAll_3*gd.jacAll_4*gd.jacAll_9,gd.jacAllAdj_3*gd.jacAllAdj_4*gd.jacAllAdj_9], # from detritus to consumer
#    [gd.jacAll_11,gd.jacAllAdj_11], # HOI regulation on consumer
#    [gd.jacAll_7,gd.jacAllAdj_7], # HOI regulation on autotroph
#    [gd.jacAll_6,gd.jacAllAdj_6]] # top-down regulation


yVal=[[gd.jacAll_12*gd.jacAll_1,gd.jacAllAdj_12*gd.jacAllAdj_1], # from detritus to autotroph
    [gd.jacAll_12*gd.jacAll_1*gd.jacAll_6,gd.jacAllAdj_12*gd.jacAllAdj_1*gd.jacAllAdj_6], # from detritus to consumer
    [gd.jacAll_14,gd.jacAllAdj_14], # HOI regulation on consumer
    [gd.jacAll_13,gd.jacAllAdj_13], # HOI regulation on autotroph
    [gd.jacAll_9,gd.jacAllAdj_9], # top-down consumer->producer regulation
    [gd.jacAll_9*gd.jacAll_4,gd.jacAllAdj_9*gd.jacAllAdj_4], # top-down consumer->nutrient regulation
    [gd.jacAll_6*gd.jacAll_1,gd.jacAllAdj_6*gd.jacAllAdj_1]] # Bottom-up from nutrient to consumer


fig, axs = plt.subplots(len(yVal),max(2,patchnbr),figsize=(14,42))

lstyle=['-','--','-.']
transp=[1,0.2,0.1]
linewidth=[2,3,4]
clr=['r','k','b']



order=[len(yVal),patchnbr,len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2)  variables, (3) series


for i in range(order[0]): # row
    for j in range(order[1]):    # column 
        for k in range(order[2]): # ctr vs adj
            st=0
            for m in unique(order[3]): # with or without engineering
                axs[i][j].plot(gd[gd.eg_0==m].eta_0, yVal[i][k][gd.eg_0==m],ls=lstyle[st],color=clr[k],alpha=transp[k],linewidth=linewidth[k]) 
                st+=1

yVal=[[gd.Xcov_14/sqrt(gd.Xcov_10*gd.Xcov_15),gd.XcovAdj_14/sqrt(gd.XcovAdj_10*gd.XcovAdj_15)], # detritus<->consumer
    [gd.Xcov_9/sqrt(gd.Xcov_10*gd.Xcov_5),gd.XcovAdj_9/sqrt(gd.XcovAdj_10*gd.XcovAdj_5)], # consumer<->producer
    [gd.Xcov_13/sqrt(gd.Xcov_5*gd.Xcov_15),gd.XcovAdj_13/sqrt(gd.XcovAdj_5*gd.XcovAdj_15)], # detritus<->producer
    [gd.Xcov_12/sqrt(gd.Xcov_0*gd.Xcov_15),gd.XcovAdj_12/sqrt(gd.XcovAdj_0*gd.XcovAdj_15)], # detritus<->nutrient
    [gd.Xcov_4/sqrt(gd.Xcov_0*gd.Xcov_5),gd.XcovAdj_4/sqrt(gd.XcovAdj_0*gd.XcovAdj_5)], # producer<->nutrient
    [gd.Xcov_8/sqrt(gd.Xcov_10*gd.Xcov_0),gd.XcovAdj_8/sqrt(gd.XcovAdj_10*gd.XcovAdj_0)], # nutrient<->consumer
    [gd.Xcov_14/sqrt(gd.Xcov_10*gd.Xcov_15),gd.XcovAdj_14/sqrt(gd.XcovAdj_10*gd.XcovAdj_15)]] # 

order=[len(yVal),patchnbr,len(yVal[0]),gd['eg_0']] # order: (0) row, (1) column, (2)  variables, (3) series


for i in range(order[0]): # row
    for j in range(order[1]):    # column 
        for k in range(order[2]): # ctr vs adj
            st=0
            for m in unique(order[3]): # with or without engineering
                axs[i][j+1].plot(gd[gd.eg_0==m].eta_0, yVal[i][k][gd.eg_0==m],ls=lstyle[st],color=clr[k],alpha=transp[k],linewidth=linewidth[k]) 
                st+=1

for j in range(max(2,patchnbr)):
    for i in range(len(yVal)):
        axs[i][j].xaxis.label.set_size(18)
        axs[i][j].yaxis.label.set_size(16)
        axs[i][j].xaxis.set_tick_params(labelsize=16)
        axs[i][j].yaxis.set_tick_params(labelsize=16)

fig.savefig(''.join([path_fig,'receng_jac_corr.pdf']),bbox_inches='tight')




