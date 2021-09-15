# -*- coding: utf-8 -*-
"""
Created on Wed May  5 11:41:52 2021

@author: Filip
"""

import numpy as np
import numpy.matlib

# ZADANI: w - vahy, C - kapacita batohu, Q - matice cen

w=np.array([0, 1, 1, 1])
C=2
Q=np.matrix([[10,0,0],[0,1,10],[0,10,1]])

# ALOKACE: f - matice profitu, S - vektor reseni

f=np.full((len(w),C+1),-np.inf)
f[0,0]=0


jednicky=[[-1]]
for j in range(C):
    jednicky1=jednicky.append([-1])
    
S=[[]]*(len(w)+1)
for i in range(0,len(w)+1):
    S[i]=jednicky.copy()
    
# HLAVNI CAST

for k in range(1,len(w)):
    for r in range(0,C+1):
        if f[k-1,r] > f[k,r]:
            f[k,r]=f[k-1,r]
            S[k][r]=S[k-1][r]
        if r+w[k] <= C:
            if S[k-1][r][0]==-1:
                beta=Q[k-1,k-1]
            else:
                SS=np.union1d(S[k-1][r],k)-1
                nuly=np.matlib.zeros((1,len(w)-1))
                nuly[0,SS.astype(int)]=1
                beta=nuly*Q*np.transpose(nuly)
            if beta > f[k,r+w[k]]:
                f[k,r+w[k]]=beta
                if S[k-1][r][0]==-1:
                    S[k][r+w[k]]=[k]
                else:
                    S[k][r+w[k]]=np.union1d(S[k-1][r],(k))
                    
r_star=np.argmax(f)
r_star=r_star%(C+1)
print('Vezmeme predmet(y)', S[len(w)-1][r_star], 'a celkovy profit bude', f[len(w)-1,r_star])