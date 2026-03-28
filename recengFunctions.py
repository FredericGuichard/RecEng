#!/usr/bin/env python
# coding: utf-8

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
os.getcwd()
import sys
import time
import matplotlib
import IPython
matplotlib.pyplot.ion()
from numpy.linalg import multi_dot
from dataclasses import dataclass, field



class RecengClass:
    
    
    # def__init__(self,hoihab,hoiloss,hoidrec,hoitroph,troph,pr,pr_sp,eta,epsi,resnot,
    #           r,delta,m,patchnbr,ecodim,detritus,stddev,I,B,het,connect,flow,
#           nutrient: int = 0)
    def __init__(self,subfold):
        self.subfolder=subfold
        self.parfile='recengScenario.csv'
        self.datafile='recengDB.csv'
        #self.isdir = os.path.isdir(path_data)
        #if isdir==False: os.mkdir(path_data) 
        
        #self.isdir = os.path.isdir(path_fig)
        self.het=0
        self.bigD=1
        #connect=array([[-0.5,0.5],[0.5,-0.5]]) # between patches
        self.connect=array([0])
    #   connect=array([[0,0],[0,0]]) # between patches
        self.flow=array([0,0,0,0]) # for each compartment -> size needs to correspond to number of compartments (ecodim)
        self.nutrient = 0
        self.minX=[]
        self.maxX=[]
        self.minAdj=[]
        self.maxAdj=[]
        self.minCtr=[]
        self.maxCtr=[]
        self.mXctr=[]
        self.mXadj=[]
        self.Xstoch=[]
        self.Xcov=[]
        self.XstochAdj=[]
        self.XcovAdj=[]   
        self.jacAll=[]
        self.jacAllAdj=[]
        self.sdestn=[]
        #self.liveDB=[]
        self.dataArray=[]
        self.dataLbl=[]
        self.dataDB=[]

        
    def der_var(self):
        self.folder = ''.join(['./',self.subfolder,'/'])  
        self.path_data=''.join([self.folder,'data/'])
        self.path_params=''.join([self.folder,'params/'])
        self.path_fig=''.join([self.folder,'fig'])
        self.pr = pd.read_csv(''.join([self.path_params,self.parfile]))
        self.pr_sp = pd.read_csv(''.join([self.path_params,'receng0.csv']))
        self.eta=asarray([self.pr_sp.eta],dtype='float64')
        self.epsi=asarray([self.pr_sp.epsi],dtype='float64')
        self.resnot=asarray([self.pr_sp.resnot],dtype='float64')
        self.r=asarray([self.pr_sp.r],dtype='float64')
        self.delta=asarray([self.pr_sp.delta],dtype='float64')
        self.m=asarray([self.pr_sp.m],dtype='float64')
        self.stddev=asarray([self.pr_sp.stddev],dtype='float64')
        self.I=asarray([self.pr_sp.I],dtype='float64')
        self.patchnbr: int = len(self.connect)
        self.ecodim: int = len(self.flow)
        self.detritus: int = self.ecodim-1
        self.B=zeros(self.ecodim*self.patchnbr)+1 
        self.hoihab=self.open_matrix_file('hoihab')
        self.hoiloss=self.open_matrix_file('hoiloss')
        self.hoidrec=self.open_matrix_file('hoidrec')
        self.hoitroph=self.open_matrix_file('hoitroph')
        self.troph=self.open_matrix_file('troph')
        

        
    
    
    #####################
    #### DYNAMICS #######
    #####################
    
    def constrain(self,constraints):
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
    
    def frC(self,con,X,pn): # return vector with functional responses of each species as a resouece
        return X[con]*(X/(self.resnot[pn]+X))
    
    def frR(self,res,X,pn):
        return X*(X[res]/(self.resnot[pn,res]+X[res]))
    
    
    def hoiVV(self,foc,form,X,pn):
        match form:
            case 'hab': # returns a vector with total habitat effect from each habitat on focal species
                #return 1-((hoihab[pn,foc,:]*sum(hoihab[pn]*X))/(sum(hoihab[pn]*X)+eta[foc]*X))
                #return 1-((hoihab[pn,foc,:]*X[foc])/(X[foc]+eta[pn,foc]*hoihab[pn,foc,:]*X))
                hab_vector=1-((self.hoihab[pn,foc,:]*X[foc])/(X[foc]+self.eta[pn,foc]*self.hoihab[pn,foc,:]*X))
                return prod(hab_vector[hab_vector>0])
            case 'troph':
                return (self.hoitroph[pn,foc,:]*X+self.epsi[pn])/(X+self.epsi[pn])
            case 'loss':
                return (self.hoiloss[pn,foc,:]*X+self.epsi[pn])/(X+self.epsi[pn])
            case 'drec':
                return (self.hoidrec[pn,foc,:]*X+self.epsi[pn])/(X+self.epsi[pn])
    
    ### Additive Terms ###        
    
    def trophicVV(self,foc,X,pn): # return difference between 2 scalars (growth-consumption) of the focal species
        #growth=self.troph[pn,foc,:]*frC(foc,X,pn)*product(hoiVV(foc,'hab',X,pn)[hoiVV(foc,'hab',X,pn)>0])*product(hoiVV(foc,'troph',X,pn)[hoiVV(foc,'troph',X,pn)>0])
        growth=self.troph[pn,foc,:]*self.frC(foc,X,pn)*self.hoiVV(foc,'hab',X,pn)*prod(self.hoiVV(foc,'troph',X,pn)[self.hoiVV(foc,'troph',X,pn)>0])
        consumption=self.troph[pn,:,foc]*self.frR(foc,X,pn)
        for eng in range(self.ecodim):
            #consumption[eng]=consumption[eng]*product(hoiVV(eng,'hab',X,pn)[hoiVV(eng,'hab',X,pn)>0])*product(hoiVV(eng,'troph',X,pn)[hoiVV(eng,'troph',X,pn)>0])
            consumption[eng]=consumption[eng]*self.hoiVV(eng,'hab',X,pn)*prod(self.hoiVV(eng,'troph',X,pn)[self.hoiVV(eng,'troph',X,pn)>0])
        return sum(growth)-sum(consumption)
    
    def lossV(self,foc,X,pn): # return scalar of modified loss of focal species
        return X[foc]*self.m[pn,foc]*prod(self.hoiVV(foc,'loss',X,pn)[self.hoiVV(foc,'loss',X,pn)>0])
    
    def recyclingV(self,foc,X,pn):
        recTot=0
        if foc==self.detritus: #recTot=sum(delta[pn]*lossV(foc,X,pn))
            for sp in range(self.ecodim):
                rec=self.delta[pn,sp]*self.lossV(sp,X,pn)
                recTot+=rec
        return recTot
    
    def direct_recyclingV(self,foc,X,pn):
        dRecTot=0
        if foc==self.nutrient:      
            for sp in range(self.ecodim):
                dRec=self.r[pn,sp]*X[sp]*prod(self.hoiVV(sp,'drec',X,pn))
                dRecTot+=dRec
        else:
            dRec= -self.r[pn,foc]*X[foc]*prod(self.hoiVV(foc,'drec',X,pn))
            dRecTot+=dRec        
        return dRecTot
    
    
    ### Spatialization ###
    
    def X_byPatch(self,X): # take vector F and return 2D array of local ecosystem vectors for each patch
        localdyn=zeros((self.patchnbr,self.ecodim))
        comp=concatenate([[i*self.patchnbr] for i in range(self.ecodim)])
        compfull= concatenate([comp+i for i in range(self.patchnbr)])
        Xl=X[compfull]
        return Xl.reshape((self.patchnbr,self.ecodim))
    
    
    def localDyn(self,X): # vector F with homogeneous ecosystem types  
            localdyn=zeros((self.patchnbr,self.ecodim))
            Xl=self.X_byPatch(X)
            for p in range(self.patchnbr):
                for c in range(self.ecodim):
                    localdyn[p,c]=self.trophicVV(c,Xl[p,:],p)-self.lossV(c,Xl[p,:],p)+self.recyclingV(c,Xl[p,:],p)+self.direct_recyclingV(c,Xl[p,:],p)+self.I[p,c]
            return localdyn.flatten('F')
        
    
    def connectivity(self): # matrix C
        mshape=[self.patchnbr*self.ecodim,self.patchnbr*self.ecodim]
        comp_connect=transpose(self.connect)
        conn = zeros(mshape)
        for i in range(self.ecodim): conn[i*self.patchnbr:i*self.patchnbr+self.patchnbr,i*self.patchnbr:i*self.patchnbr+self.patchnbr]=comp_connect
     #   print('C matrix:',shape(conn))
        return conn
    
    def flows(self): # matrix Q
        comp_flow=self.flow*diag(zeros(self.ecodim)+1)
        land=diag(zeros(self.patchnbr)+1)
     #   print('Q matrix:',shape(kron(comp_flow,land)))
        return kron(comp_flow,land)
    
    def dMdf_dt(self,t,X):
        return self.localDyn(X)+multi_dot([self.flows(),self.connectivity(),X])
    
    
    ### STOCHASTIC DYNAMICS ####
    
    def G(self,x,t):
        #global B
        #return stddev[0]*sqrt(B) # demographic stchasticity
        return self.stddev[0]*self.B  # Environmenal stochastiity
    
    def fdf(self,x,t):  
        return self.localDyn(x)+multi_dot([self.flows(),self.connectivity(),x])
    
    ##############################
    ##### Stability Analysis #####
    ##############################
    
    
    def highpass(self,data: ndarray, cutoff: float, sample_rate: float, poles: int = 5):
        sos = scipy.signal.butter(poles, cutoff, 'highpass', fs=sample_rate, output='sos')
        filtered_data = scipy.signal.sosfiltfilt(sos, data)
        return filtered_data
    
    def localjac(self,X,pnbr):
        localdyn=zeros(self.ecodim)
        for c in range(self.ecodim):
            localdyn[c]=(self.trophicVV(c,X,pnbr)-self.lossV(c,X,pnbr))+self.recyclingV(c,X,pnbr)+self.direct_recyclingV(c,X,pnbr)+self.I[pnbr,c]
        return localdyn
    
    def jacEq(self,Xeq,pnbr):
        Xl=self.X_byPatch(Xeq)
        fun= lambda X: self.localjac(X,pnbr) 
        jac = nd.Jacobian(fun)(Xl[pnbr,:])
        return jac.flatten('F')
    
    def minmax(self,xseries): 
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
    
    def out_par(self):
        prf=pd.read_csv(''.join([self.path_params,self.parfile]))
        vname=[]
        for col in prf.columns:
            for row in range(self.ecodim):
                name=''.join([col,str(row)])
                vname.append(name)
        parval=prf.to_numpy().flatten('F')
        valdf=pd.DataFrame(parval).T
        valdf.columns=vname
        return valdf
    
    def out_var(self):
        dataSize=[]
        for iv in range(len(self.dataArray)): dataSize.append(size(self.dataArray[iv]))
        #print(dataSize,'\n')
        vlbl=[]
        vk=0    
        for col in self.dataLbl:
            for row in range(dataSize[vk]):
                name='_'.join([str(col),str(row)])
                vlbl.append(name)
            vk+=1
        vk=0
        datafr=[]
        for g in range(len(self.dataArray)):
            datafr=concatenate([datafr,array(self.dataArray[g]).flatten('F')],axis=0)
            #datafr=pd.concat([pd.DataFrame(datafr),pd.DataFrame(data[g])],axis=1)
        datafr=pd.DataFrame(datafr).T
        datafr.columns=vlbl
        for i in range(1,self.ecodim):
            datafr=datafr.drop('_'.join(['eta',str(i)]),axis=1)
            datafr=datafr.drop('_'.join(['eg',str(i)]),axis=1)
        return pd.DataFrame(datafr)
    
    def out_combine(self,par,data):
        return pd.concat([par,data],axis=1)
    
    def out_write_OFF(self,data,file):
        fileExist=os.path.isfile(file)
        if fileExist:
            current_db=pd.read_csv(file)
            merged_db=pd.concat([current_db,data],join='outer',axis=0)
        else: merged_db=data
        with open(file, 'w') as out: #  out.write('\n')
            merged_db.to_csv(file,mode='w',index=False,header=True)
    
    def out_write(self,data,file): # never append new data to any exisitng file
        merged_db=data
        with open(file, 'w') as out: #  out.write('\n')
            merged_db.to_csv(file,mode='w',index=False,header=True)
    
    def out_lbl(self,data,last_param,first_var,prfile,varfile):
        param_col=pd.DataFrame(data.iloc[0,0:data.columns.get_loc(last_param)+1]).T
        with open(prfile, 'w') as out:
            param_col.to_csv(prfile,mode='w',index=False,header=True)
        var_col=pd.DataFrame(data.iloc[:,data.columns.get_loc(first_var):].columns.values).T
        with open(varfile, 'w') as out:
            var_col.to_csv(varfile,mode='w',index=False,header=False)
        
                
    #####################
    ####PARAMETERS#######
    #####################
    def params(self,save=0,read=1,fname='receng_params.csv'):
        directory = ''.join([folder,'params/',])
        return pd.read_csv(os.path.join(directory,fname))
    
    def setParams(self,fromFile=1,fnbr=0):
        if fromFile==1: prSim=params(read=1,fname=pfiles['name'][fnbr])
        else: prSim=params()
        return prSim
    
    def genParams(self,val=[],pspace=[],fromfile=1,filen='paramfile',append=0):
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
    
    def open_matrix_file(self,matrice):
        match self.het:
            case 1:           
                matlist=empty((self.patchnbr,self.ecodim,self.ecodim))
                for i in range(self.patchnbr):   
                    filename=''.join([self.folder,'params/',matrice,str(i),'.csv'])
                    matlist[i]=genfromtxt(filename,delimiter=',')
            case 0:           
                matlist=empty((self.patchnbr,self.ecodim,self.ecodim))
                for i in range(self.patchnbr):   
                    filename=''.join([self.folder,'params/',matrice,'0','.csv'])
                    matlist[i]=genfromtxt(filename,delimiter=',')               
        return matlist
    
    def save_matrix_file(self,matrice):
        match self.het:
            case 1:           
                for i in range(self.patchnbr):   
                    filename=''.join([self.folder,'params/',matrice,str(i),'.csv'])
                    savetxt(filename,matrice,delimiter=',')
            case 0:           
                for i in range(self.patchnbr):   
                    filename=''.join([self.folder,'params/',matrice,'0','.csv'])
                    savetxt(filename,matrice,delimiter=',')               
        return
    
    def open_sp_file(self, matrice):
    
        prlengthFile=pd.read_csv(''.join([self.folder,'params/',matrice,'0','.csv']))
        prlength=len(prlengthFile.columns)
        varNames=prlengthFile.columns.values
        match self.het:
            case 1:           
                for j in range(prlength):
                    myStr =varNames[j]
                    globals()[varNames[j]]=[]
                    for i in range(self.patchnbr):
                        filename=''.join([self.folder,'params/',matrice,str(i),'.csv'])
                        df=pd.read_csv(filename) 
                        globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                    globals()[varNames[j]]=reshape(globals()[varNames[j]],(self.patchnbr,self.ecodim))       
            case 0:           
                for j in range(prlength):
                    myStr =varNames[j]
                    globals()[varNames[j]]=[]
                    for i in range(self.patchnbr):
                        filename=''.join([self.folder,'params/',matrice,'0','.csv'])
                        df=pd.read_csv(filename) 
                        globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                    globals()[varNames[j]]=reshape(globals()[varNames[j]],(self.patchnbr,self.ecodim))
    
        return
    
    def save_sp_file(self,folder, matrice):
    
        prlengthFile=pd.read_csv(''.join([folder,'params/',matrice,'0','.csv']))
        prlength=len(prlengthFile.columns)
        varNames=prlengthFile.columns.values
        match het:
            case 1:           
                for j in range(prlength):
                    myStr =varNames[j]
                    globals()[varNames[j]]=[]
                    for i in range(self.patchnbr):
                        filename=''.join([folder,'params/',matrice,str(i),'.csv'])
                        df=pd.read_csv(filename) 
                        globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                    globals()[varNames[j]]=reshape(globals()[varNames[j]],(self.patchnbr,self.ecodim))       
            case 0:           
                for j in range(prlength):
                    myStr =varNames[j]
                    globals()[varNames[j]]=[]
                    for i in range(self.patchnbr):
                        filename=''.join([folder,'params/',matrice,'0','.csv'])
                        df=pd.read_csv(filename) 
                        globals()[varNames[j]]=concatenate([globals()[varNames[j]],df[varNames[j]]])
                    globals()[varNames[j]]=reshape(globals()[varNames[j]],(self.patchnbr,self.ecodim))
    
        return
    
    
    ##############################################
    ##### Query Data File ####################
    ##############################################
    
    def query(self,newFilter=pd.DataFrame(),valGrp=['eg_0','eta_0'],prfile='plot_par.csv',varfile='plot_var.csv'):
    
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
    
    def extinction(self,ext_index):
        hoihab=open_matrix_file('hoihab')
        hoiloss=open_matrix_file('hoiloss')
        hoidrec=open_matrix_file('hoidrec')
        hoitroph=open_matrix_file('hoitroph')
        self.troph=open_matrix_file('troph')
        open_sp_file('receng')
        
        hoihab=delete(hoihab,ext_index)
        hoiloss=delete(hoiloss,ext_index)
        hoidrec=delete(hoidrec,ext_index)
        hoitroph=delete(hoitroph,ext_index)
        self.troph=delete(self.troph,ext_index)
        
        savetxt('hoihab.csv',hoihab,delimiter=',')
        savetxt('hoiloss.csv',hoiloss,delimiter=',')
        savetxt('hoidrec.csv',hoidrec,delimiter=',')
        savetxt('hoitroph.csv',hoitroph,delimiter=',')
        savetxt('troph.csv',torph,delimiter=',')
        
        
        
    
    #############################################
    ####### Main Simulation function ############
    #############################################
    
    def receng(self):
   
        sos = signal.butter(4,0.25, 'highpass', output='sos') # define highpass filter applied to stochastic time series
        t = linspace(0.0,5000.0,10000) # changed from 10000, 50000
        X0=zeros(self.ecodim*self.patchnbr)
        for g in range(self.ecodim*self.patchnbr):
            X0[g]=random.uniform(0.1,2)
  
        ## uncomment both fachab tests if h<1 and lamda adjustment is needed
        fachab=1    
        if self.hoihab[0,2,3]<1:
            fachab=self.hoihab[0,2,3]
            self.hoihab[0,2,3]=1
        #################################    
            
        X = integrate.solve_ivp(self.dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
        tr=X.t
        trans=int(len(X.y[0])/4)
        mX=mean(X.y[:,trans:],axis=1)
        
        ## uncomment if h<1 and lamda adjustment is needed
        if fachab<1:
            self.troph[0,2,1]=self.troph[0,2,1]*(1-(mX[2]/(self.eta[:,2]*mX[3]+mX[2])))/(1-(fachab*mX[2]/(self.eta[:,2]*mX[3]+mX[2])))            
            print(self.troph[0,2,1])
            self.hoihab[0,2,3]=fachab
            X = integrate.solve_ivp(self.dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
            tr=X.t
            trans=int(len(X.y[0])/4)
            mX=mean(X.y[:,trans:],axis=1)
        ###################################    
        
    
        if sum(self.pr.Iadj)>0: 
            if self.pr.Iadj[self.detritus]==1: # turn off recycling and adjust the detritus input
                deltaOld=self.delta
                self.delta=zeros((self.patchnbr,self.ecodim))
                self.I[:,self.detritus]+=sum(deltaOld*self.m*mX) 
    
            if self.pr.Iadj[self.nutrient]==1: # turn off decomposition and adjust the nutrient input in the isolated green web
                rOld=self.r
                self.r=zeros((self.patchnbr,self.ecodim))
                self.I[:,self.nutrient]+=sum(rOld*mX)
    
            X0=zeros(self.ecodim*self.patchnbr)
            for g in range(self.ecodim*self.patchnbr):
                X0[g]=random.uniform(0.1,2)
            X = integrate.solve_ivp(self.dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
            tr=X.t
            trans=int(len(X.y[0])/4)
            mX=mean(X.y[:,trans:],axis=1)
    
        minTemp=[];maxTemp=[]
        for g in range(self.ecodim*self.patchnbr):
            self.minX,self.maxX=self.minmax(X.y[g][int(-len(t)/2):-1])
            minTemp.append(self.minX),maxTemp.append(self.maxX)
        self.minCtr.append(minTemp);self.maxCtr.append(maxTemp)
        self.mXctr.append(mX)  # storing either mean stock values ( with Iadj= 0 or 1) 
    
    
        if self.pr.jac[0]==1:
            self.jacAll=self.jacEq(mX,0)
    
        if sum(self.stddev)>0:
            stoch=zeros(self.ecodim*self.patchnbr)+1
            self.B = stoch*diag(mX)
            sde = sdeint.itoint(self.fdf, self.G, mX, t) # ito method
            #sde = sdeint.stratint(self.fdf, self.G, mX, t) # stratonovich method
            trans=int(len(t)/4)
            self.sdestn=sde[trans:,0:]         
            #sdestn = signal.sosfilt(sos, sdestn, axis=-1) # apply sos high-pass filter to stochastic time series
            self.sdecov=cov(transpose(self.sdestn))
            self.sdemean=mean(self.sdestn,axis=0)         
            self.Xcov.append(self.sdecov.flatten())
            self.Xstoch.append(transpose(self.sdemean))
    
        if sum(self.pr.lmdaAdj)>0:
            mXl=self.X_byPatch(mX)
            for pnumb in range(self.patchnbr):
                for sp in range(self.ecodim):
                    self.troph[pnumb,sp,:]=self.troph[pnumb,sp,:]*self.hoiVV(sp,'hab',mXl[pnumb],pnumb)*self.hoiVV(sp,'troph',mXl[pnumb],pnumb)
                    self.m[pnumb,sp]=self.m[pnumb,sp]*prod(self.hoiVV(sp,'loss',mXl[pnumb],pnumb)[self.hoiVV(sp,'loss',mXl[pnumb],pnumb)>0])
                    self.r[pnumb,sp]=self.r[pnumb,sp]*prod(self.hoiVV(sp,'drec',mXl[pnumb],pnumb)[self.hoiVV(sp,'drec',mXl[pnumb],pnumb)>0])          
    
            self.hoitroph=zeros((self.patchnbr,self.ecodim,self.ecodim))+1
            self.hoihab=zeros((self.patchnbr,self.ecodim,self.ecodim))
            self.hoiloss=zeros((self.patchnbr,self.ecodim,self.ecodim))+1
            self.hoidrec=zeros((self.patchnbr,self.ecodim,self.ecodim))+1
    
            X0=zeros(self.ecodim*self.patchnbr)
            for g in range(self.ecodim*self.patchnbr):
                X0[g]=random.uniform(0.1,2)
            X = integrate.solve_ivp(self.dMdf_dt, [t[0], t[-1]], X0,t_eval=t,dense_output=True,atol=1e-6, rtol=1e-6) # changed both tolerances from 10-9
            tr=X.t
            trans=int(len(X.y[0])/4)
            mX=mean(X.y[:,trans:],axis=1)
    
            minTemp=[];maxTemp=[]
            for g in range(self.ecodim*self.patchnbr):
                self.minX,self.maxX=self.minmax(X.y[g][int(-len(t)/2):-1])
                minTemp.append(self.minX),maxTemp.append(self.maxX)
            self.minAdj.append(minTemp);self.maxAdj.append(maxTemp) 
            self.mXadj.append(mX)    
    
            if self.pr.jac[0]==1:
                self.jacAllAdj=self.jacEq(mX,0)  
    
            if sum(self.stddev)>0:
                stoch=zeros(self.ecodim*self.patchnbr)+1
                self.B = stoch*diag(mX)
                sde = sdeint.itoint(self.fdf, self.G, mX, t) # ito method
                #sde = sdeint.stratint(self.fdf, self.G, mX, t) # stratonovich method
                self.sdestn=sde[trans:,0:]
                #savetxt('time_series.txt', array(sdestn)) # save adjusted stocahstic time series to file
                #sdestn = signal.sosfilt(sos, sdestn, axis=-1) # apply sos high-pass filter to stochastic time series
                self.sdecov=cov(transpose(self.sdestn))
                self.sdemean=mean(self.sdestn,axis=0)         
                self.XcovAdj.append(self.sdecov.flatten())
                self.XstochAdj.append(transpose(self.sdemean))
                
        stochArray=[self.mXctr,self.minCtr,self.maxCtr,self.Xstoch,self.Xcov,self.mXadj,self.minAdj,self.maxAdj,self.XstochAdj,self.XcovAdj,self.jacAll,self.jacAllAdj]        

        return stochArray
    
    
