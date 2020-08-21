#!/usr/bin/env python
# coding: utf-8

# # 2-pt correlation

# In[1]:


import numpy as np
import math
import matplotlib.pyplot as plt
data = np.loadtxt("pion_gamma15_p0_t0_1.txt")


# In[2]:


C2i = data[:,7]
N_conf = 10
C2i = np.reshape(C2i,(64,10),order='F')


# In[3]:


t = np.linspace(1,64,num=64)


# In[4]:


C2 = np.mean(C2i,1)
C2err = np.std(C2i,axis=1)/math.sqrt(N_conf-1)


# In[5]:


mi = np.zeros((63,10))
a0 = 0.197/0.12
for row in range(0,63):
    for column in range(0,10):
        mi[row,column] = a0 * math.log(C2i[row,column]/C2i[row+1,column])
        
m = np.mean(mi,1)
merr = np.std(mi,axis=1)/math.sqrt(N_conf-1)
# print(m)
# print(merr)


# In[6]:


plt.xlim([1,21])
plt.errorbar(t[2:20],m[2:20],yerr=merr[2:20],fmt='.',capsize=3,elinewidth=2)
plt.legend(['PionMass'])
plt.plot(t,0.31*np.ones(64),'r--')
plt.xlabel('Time index',fontsize=14)
plt.ylabel('Average effective mass',fontsize=14)
plt.title('2-point correlation',fontsize=16)
plt.show()
plt.close()


# # Jacknife Resampling

# In[7]:


C2mat = np.tile(np.reshape(C2,(64,1)),(1,10))
C2mat = C2mat * N_conf


# In[8]:


C2iprime = (C2mat - C2i) / (N_conf - 1)
C2prime = np.mean(C2iprime,1)
C2errprime = np.std(C2iprime,axis=1)/math.sqrt(N_conf-1)
C2covprime = np.cov(C2iprime,rowvar=True,bias=True)


# In[9]:


miprime = np.zeros((63,10))
for row in range(0,63):
    for column in range(0,10):
        miprime[row,column] = a0 * math.log(C2iprime[row,column]/C2iprime[row+1,column])
        
mprime = np.mean(miprime,1)
merrprime = np.std(miprime,axis=1)*math.sqrt(N_conf-1)
# print(mprime)
# print(merrprime)


# In[10]:


plt.xlim([1,21])
plt.errorbar(t[2:20],mprime[2:20],yerr=merrprime[2:20],fmt='.',capsize=3,elinewidth=2)
plt.legend(['PionMass'])
plt.plot(t,0.31*np.ones(64),'r--')
plt.xlabel('Time index',fontsize=14)
plt.ylabel('Average effective mass',fontsize=14)
plt.title('2-point correlation',fontsize=16)
plt.show()
plt.close()


# # Data Curve Fitting

# In[11]:


import gvar as gv
import lsqfit


# In[12]:


xmin = 4
xmax = 20
prior = {'c0':gv.gvar(0.02,0.5),'m0':gv.gvar(0.3,1.),'c1':gv.gvar(1.,100.),'deltam':gv.gvar(0.5,10.)}


# In[13]:


xfit = np.arange(xmin,xmax)
yfit = gv.gvar(C2prime[xmin:xmax],C2errprime[xmin:xmax])
# yfit = gv.gvar(C2prime[xmin:xmax],np.sqrt(C2covprime[xmin:xmax,xmin:xmax].diagonal()))
# yfit = gv.gvar(C2prime[xmin:xmax],C2covprime[xmin:xmax,xmin:xmax])


# In[14]:


def fit1state(x, p):
    ans = p['c0']*(np.exp(-x * p['m0'] / a0))
    return ans

fit1 = lsqfit.nonlinear_fit(data=(xfit,yfit),svdcut=1e-3,prior=prior,fcn=fit1state)
print(fit1)


# In[15]:


def fit2state(x, p):
    ans = p['c0']*(np.exp(-x * p['m0'] / a0))*(1+p['c1']*(np.exp(-x* p['deltam']/a0)))
    return ans

fit2 = lsqfit.nonlinear_fit(data=(xfit,yfit),svdcut=1e-3,prior=prior,fcn=fit2state)
print(fit2)
