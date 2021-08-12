# -*- coding: utf-8 -*-
"""Propensity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1nVA0x11ILHth5Z9lVAFgumZeNM_ZYWGC
"""


import sys,os
import json,csv,numpy as np
from pathlib import Path


def removeXAA(seq):
    NonAaSeq=["B","J","O","U","X","Z"];Sq=seq
    for aa in seq:
        if aa in NonAaSeq:
            Sq=Sq.replace(aa,'')
    return Sq

def loadSEQdata(path):
    PSq=[];c=0;SL=[];c1=1
    fi=open('{}/Propen/cystein_pos_19.txt'.format(path),"r")
    for line in fi:
      seq=line.split()[2]
      if "*" not in seq:
        PSq.append(seq)
        c+=1
    n=len(PSq)#1850#
    #print("Total Pos Seq = {} {}".format(c,len(PSq)))
    l1=[]
    fi=open('{}/Propen/cystein_neg_19.txt'.format(path),"r")
    for line in fi:
      seq=line.split()[2]
      if "*" not in seq:
        l1.append(seq)
        if c1!= 0 and c1%n==0:
            SL.append(l1)
            #print('Neg Seq={}'.format(len(l1)))
            l1=[];
        c1+=1
    #print("Total Neg Seq = {}".format(len(SL)))
    return PSq,SL



def loadposiDataMAT(path,PSq,ln):
    global alpha
    MAT=np.zeros((20,(ln*2+1)),dtype=float)
    seq=[];c=0
    for seq in PSq:
        for j in range(ln*2+1):
            for i in range(20):
                if alpha[i]==seq[j] and j!=ln:
                    MAT[i][j]+=1
    return MAT



def getRow_ij(i,j,FMAT):
    Lij=[]
    for M in FMAT:
        Lij.append(M[i][j])
    Lij=np.array(Lij)
    mn=np.mean(Lij)
    std=np.std(Lij)
    return mn,std

def writeMAT(fname,pathl,MAT):
    global alpha
    m,n=MAT.shape
    npath="{}Propen".format(pathl)
    s=''
    for i in range(m):
        s+="{},{}\n".format(alpha[i],",".join(["{:.3f}".format(x) for x in MAT[i]]))
    f=open("{}/{}.csv".format(npath,fname),"w")
    f.write(s);f.close()
    print("{} write completed".format(fname))



def PropencityComputation(path,ln):
    global alpha;

    PSq,SL=loadSEQdata(path)
    posD=loadposiDataMAT(path,PSq,ln)

    FMAT=[]
    for NSq in SL:
        MAT=loadposiDataMAT(path,NSq,ln);
        FMAT.append(MAT)
    dim=(ln*2+1)
    meanMAT=np.zeros((len(alpha),dim),dtype=float)
    stdMAT=np.zeros((len(alpha),dim),dtype=float)

    for i in range(len(alpha)):
        for j in range(dim):
            m,s=getRow_ij(i,j,FMAT)
            meanMAT[i][j]=m
            stdMAT[i][j]=s

    PropM=np.zeros((len(alpha),dim),dtype=float)
    for i in range(len(alpha)):
        for j in range(dim):
            if j!=ln:
                PropM[i][j]=(posD[i][j]-meanMAT[i][j])/max(stdMAT[i][j],1)
            else:
                PropM[i][j]=0
    norPropM=np.zeros((len(alpha),dim),dtype=float)
    for i in range(len(alpha)):
        mx=max([abs(x) for x in PropM[i]])
        norPropM[i]=PropM[i]/mx
    print("........."*10)
    writeMAT("PositiveMat",path,posD)
    writeMAT("NegativeAvgMat",path,meanMAT)
    writeMAT("NegativeStdMat",path,stdMAT)
    writeMAT("AApropencity",path,PropM)
    writeMAT("AApropencityNorm",path,norPropM)
    print("........."*10)

alpha=['A', 'C', 'D', 'E', 'F', 'G', 'H','I',\
             'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S',\
             'T', 'V', 'W', 'Y']
          
PropencityComputation(path,19)
