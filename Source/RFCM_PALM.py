#!/usr/bin/env python
# coding: utf-8

# # 1. Import Section

# In[16]:


import sys,os
from Bio import SeqIO
import json,csv,numpy as np
from Bio.Seq import Seq
from pathlib import Path
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,auc,precision_recall_curve,         recall_score, confusion_matrix, classification_report,         accuracy_score, f1_score,roc_auc_score,matthews_corrcoef
import warnings
warnings.filterwarnings('ignore')


# ##  C3: Parse Fasta and Sub-Seqs. Data Gen.:

# In[17]:


def removeXAA(seq):
    NonAaSeq=["B","J","O","U","X","Z"];Sq=seq
    for aa in seq:
        if aa in NonAaSeq:
            Sq=Sq.replace(aa,'')
    return Sq
def ParseSeq(path,inF):
    SqD={}
    fi=SeqIO.parse('{}/InputFile/{}'.format(path,inF),"fasta")
    for fasta in fi:
        name, sequence = fasta.id, str(fasta.seq)
        Pid=fasta.id.split("|")[1]
        SqD[Pid]=sequence
    return SqD
def GenSub(S,ln):
    xi =[l for l, ltr in enumerate(S) if ltr == 'C']
    SubL=[]
    for idx in xi:
        if idx>=ln and (len(S)-idx)<ln:
            b=ln-(len(S)-idx)+1
            si=S[idx-ln:]+"*"*b
        elif idx<ln:
            a=ln-idx
            si="*"*a+S[:idx+ln+1]
        elif idx>=ln and idx<len(S):
            si=S[idx-ln:idx+ln+1]
        SubL.append([idx,si])
    return SubL,len(SubL)
def SubSqGen(path,inF,l):
    PdS={};cnt=0
    SqD=ParseSeq(path,inF)
    for p in SqD:
        PdS[p],c=GenSub(SqD[p],l);cnt+=c
    print("{} Proteins with : {} : cystene sites".format(len(PdS),cnt))
    return PdS


# ## C4: DataGen + FET-(Physico & Propensity)

# In[18]:


def getPHYdataX(seq,Aindex,PropD):
    A=[];B=[]
    ln=int(len(seq)/2)
    for j in range(len(seq)):
        ke = seq[j]
        if j<ln:
            if ke in Aindex:
                A.append(Aindex[ke]*PropD[ke][str(j)])
            else:
                A.append(0.0)
        elif j>ln:
            if ke in Aindex:
                B.append(Aindex[ke]*PropD[ke][str(j)])
            else:
                B.append(0.0)
    A=np.array(A);B=np.array(B);
    A=(A)/max(max(abs(A)),1)
    B=(B)/max(max(abs(B)),1)
    C=np.concatenate((A,B),axis=0)
    return C

def dataGenX(PropD,PHYCHdi,SubSqD,FetS):
    c=0;D=[];PrPsD={};Lb=[]
    for k in SubSqD:
        PrPsD[k]={}
        for pid in SubSqD[k]:
            pd,Seq=pid
            M=np.array([])
            for fet in FetS:
                Am=getPHYdataX(Seq,PHYCHdi[fet],PropD)
                M=np.concatenate((M,Am),axis=0)
            PrPsD[c]=[k,pd]; c+=1
            D.append(M)
    return D,PrPsD


# ## C6:Consensus

# In[19]:


def ConsenSusRes(Rl):
    S1=[];S2=[];S3=[]
    for i,k in enumerate(Rl['KB']):
    #print(Rl['KB'][i],Rl['GA'][i],Rl['UN'][i])
        sk=(Rl['KB'][i]+Rl['GA'][i]+Rl['UN'][i])
        if sk>=3:S3.append(1)
        else:S3.append(0)
        if sk>=2:S2.append(1)
        else:S2.append(0)
        if sk>=1:S1.append(1)
        else:S1.append(0)
    return S1,S2,S3


# ##  C5: Classification

# In[23]:


def RunClassifierX(path, Dts,DataT,FetT,PrPsD):
    cf=load('{}/BestModelFile/{}_{}_BM.joblib'.format(path,DataT,FetT))
    Rl=cf.predict(Dts)
    return Rl


# ##  C2: Main.:

# In[21]:


def main(path,DataT,inF,ln):
    with open("{}/DataFile/Propensity.json".format(path),'r') as f:
        PropD=json.load(f)
    with open("{}/DataFile/AAIscore.json".format(path),'r') as f:
        PHYCHdi=json.load(f)
    with open("{}/DataFile/Feature_List.json".format(path),'r') as f:
        FetD=json.load(f)
    
    SubSqD=SubSqGen(path,inF,ln)
    Rl={}
    for FetT in ["KB","GA","UN"]:
        FetS=FetD[DataT][FetT]
        DataM,PrPsD=dataGenX(PropD,PHYCHdi,SubSqD,FetS)
        Rl[FetT]=RunClassifierX(path, DataM,DataT,FetT,PrPsD)
    S1,S2,S3=ConsenSusRes(Rl)
    outPath="{}/Output/".format(path)
    if not os.path.exists(outPath):
        os.makedirs(outPath)
        
    S='Protein,Position,Pred(1*),Pred(2*),Pred(3*)\n'
    for k,dg in enumerate(S1):
        S+="{},{},{},{},{}\n".format(PrPsD[k][0],PrPsD[k][1],S1[k],S2[k],S3[k])
    
    f=open("{}{}_Res.csv".format(outPath,DataT),'w')
    f.write(S);f.close()


# ## C1:: Main

# In[22]:


if __name__=="__main__":
    path=Path(os.getcwd()).parent
    ln=19
    if len (sys.argv) != 3 :
        print("@@@@@"*15)
        print("   Usage       : python RFCM_PALM.py <filename> <dataType>")
        #print("   Example     : python RFCM_PALM.py inputSeq.fasta Male")
        print("   <filename>  : input fasta file     : example: xyz.fasta")
        print("   <dataType>  : Male/Female/Combined : example: Male")
        print("@@@@@"*15)
        sys.exit (1)
    else:
        print("----------"*5)
        fname="{}".format(sys.argv[1].strip())
        DataT="{}".format(sys.argv[2].strip())
        print("\tYour input fasta file : {}  \n\tYour data Choice      : {}".format(fname,DataT))
        if DataT not in ["Male","Female", "Combined"]:
            print("\tInvalid <dataType> @@^@@ ! \n\tPlease enter any of the valid data type <Male/Female/Combined> ") 
            sys.exit (1)
        main(path,DataT,fname,ln)
        print("{} \n\t Hi!  Your prediction is ready.... \n\t Please check the Output Directory!\n{}\n".format("*****"*10,"*****"*10))
        #print("-------"*5)

# In[ ]:




