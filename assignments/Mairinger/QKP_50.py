# -*- coding: utf-8 -*-
"""
Created on Wed May  5 13:12:42 2021

@author: Filip
"""

import numpy as np
import numpy.matlib

# ZADANI: w - vahy, C - kapacita batohu, Q - matice cen

w=np.array([0, 13, 14, 14, 15, 15, 9, 26, 24, 13, 11, 9, 12, 25, 12, 26])
C=50
Q=np.array([
7	,12	,7	,6	,13	,8	,11	,7	,15	,23	,14	,15	,17	,9	,15,
12	,6	,15	,13	,10	,15	,9	,10	,8	,17	,11	,13	,12	,16	,15,
7	,15	,13	,11	,16	,6	,8	,14	,13	,4	,14	,8	,15	,9	,16,
6	,13	,11	,16	,10	,13	,14	,14	,17	,15	,14	,6	,24	,13	, 4,
13	,10	,16	,10	,5	,9	,7	,25	,12	,6	,6	,16	,10	,15	,14,
8	,15	,6	,13	,9	,10	,2	,13	,12	,16	,9	,11	,23	,10	,21,
11	,9	,8	,14	,7	,2	,9	,8	,18	,4	,13	,14	,14	,17	,15,
7	,10	,14	,14	,25	,13	,8	,23	,9	,16	,12	,3	,14	,14	,27,
15	,8	,13	,17	,12	,12	,18	,9	,18	,15	,16	,13	,14	,7	,17,
23	,17	,4	,15	,6	,16	,4	,16	,15	,12	,28	,5	,19	,6	,18,
14	,11	,14	,14	,6	,9	,13	,12	,16	,28	,9	,13	,4	,13	,16,
15	,13	,8	,6	,16	,11	,14	,3	,13	,5	,13	,22	,11	,19	,13,
17	,12	,15	,24	,10	,23	,14	,14	,14	,19	,4	,11	,17	,15	,12,
9	,16	,9	,13	,15	,10	,17	,14	,7	,6	,13	,19	,15	,32	,16,
15	,15	,16	,4	,14	,21	,15	,27	,17	,18	,16	,13	,12	,16	, 8
    ])

Q=Q.reshape(15,15)

# ALOKACE: f - matice profitu, S - vektor reseni

f=np.full((len(w),C+1),-np.inf)
f[0,0]=0

jednicky=[[-1]]
for j in range(C):
    jednicky1=jednicky.append([-1])
    
S=[[]]*len(w+1)
for i in range(0,len(w)):
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
print('Vezmeme predmety', S[len(w)-1][r_star], 'a celkovy profit bude', f[len(w)-1,r_star])