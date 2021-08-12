# -*- coding: utf-8 -*-




import sys,os
import json,csv,numpy as np,operator
from pathlib import Path
#from joblib import dump, load
from collections import OrderedDict as OD
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score,auc,precision_recall_curve, \
        recall_score, confusion_matrix, classification_report, \
        accuracy_score, f1_score,roc_auc_score,matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

def removeXAA(seq):
    NonAaSeq=["B","J","O","U","X","Z"];Sq=seq
    for aa in seq:
        if aa in NonAaSeq:
            Sq=Sq.replace(aa,'')
    return Sq

def loadSEQdata(path,rc):
    PSq=[];c=0;SL=[];c1=1
    fi=open('{}/KB/cystein_pos_21.txt'.format(path),"r")
    for line in fi:
      seq=line.split()[2]
      if "*" not in seq:
        PSq.append(seq)
        c+=1
        #if c==rc:break
    n=len(PSq)#1850#
    print("Total Pos Seq = {} {}".format(c,len(PSq)))
    l1=[]
    fi=open('{}/KB/cystein_neg_21.txt'.format(path),"r")
    for line in fi:
      seq=line.split()[2]
      if "*" not in seq:
        l1.append(seq)
        if c1!= 0 and c1%n==0:
            SL.append(l1)
            #print('Neg Seq={}'.format(len(l1)))
            l1=[];
        c1+=1
    print("Total Neg Seq = {}".format(len(SL[4])))
    return PSq,SL[0]

def loadProp(path,ln):
  PropD={}
  with open("{}/KB/PROPENSITY/{}.csv".format(path,ln), newline='') as cf:
      csvR = csv.reader(cf, delimiter=',', quotechar='|')
      for r in csvR:
          PropD[r[0]]={}
          for i,aa in enumerate(r[1:]):
              PropD[r[0]][i]=float(aa)
  with open("{}/KB/PROPENSITY/Pro_{}.json".format(path,ln),'w') as f:
      json.dump(PropD,f)
  return PropD

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

def KBdataGen(PropD,PHY,PSq,NSq,ln,rx):
  D=[];L=[]
  for Seq in PSq:
    seq=Seq[rx:len(Seq)-rx]
    M=getPHYdataX(seq,PHY,PropD)
    D.append(M);
    L.append(1)
  for Seq in NSq:
    seq=Seq[rx:len(Seq)-rx]
    M=getPHYdataX(seq,PHY,PropD)
    D.append(M);
    L.append(0)
  D=np.array(D);L=np.array(L)
  return D,L

def Evaluate(Dtr,Ltr):
  cf=RandomForestClassifier(n_estimators=300,min_samples_leaf =20)
  skf = StratifiedKFold(n_splits=5,shuffle=False)
  AUCL=0.0
  for train_index, test_index in skf.split(Dtr, Ltr):
      X_train, X_test = Dtr[train_index], Dtr[test_index]
      y_train, y_test = Ltr[train_index], Ltr[test_index]
      cf.fit(X_train, y_train)
      prTsd_RF=cf.predict(X_test)
      probas_RF = cf.predict_proba(X_test)
      AUCL+=roc_auc_score(y_test,probas_RF[:, 1])
  return AUCL/5.0

def KB_IntersectionUnion(ResAn):
  IB10=IB15=IB25=IB50=IB75=IB100=[x for x in ResAn[19]];UB10=UB15=UB25=UB50=UB75=UB100=[]
  XX={10:{"I":IB10,"U":UB10},10:{"I":IB10,"U":UB10},15:{"I":IB15,"U":UB15},15:{"I":IB15,"U":UB15},25:{"I":IB25,"U":UB25},50:{"I":IB50,"U":UB50},75:{"I":IB75,"U":UB75},100:{"I":IB100,"U":UB100}}
  FKB={}
  for rr in [10,15,25,50,75,100]:
    for ln in ResAn:
      XX[rr]["I"]=set(XX[rr]["I"]) & set(ResAn[ln][:rr])
      XX[rr]["U"]=set(XX[rr]["U"]).union(set(ResAn[ln][:rr]))
    FKB[rr]={'KBI':XX[rr]["I"],'KBU':XX[rr]["U"]}
  return FKB

def KBestFetSelection(path):
  ProP={};rc=200
  for ln in range(15,22):
    ProP[ln]=loadProp(path,ln)
  with open(path+"/DataFile/AAIscore.json",'r') as f:
    PHYCHdi=json.load(f)
  PSq,NSq=loadSEQdata(path,rc)

  maxln=21
  ResAn={}
  for ln in range(15,22):
    rx=maxln-ln
    PropD=ProP[ln]
    AIRes={}
    for i,fet in enumerate(PHYCHdi):
      Dat,L=KBdataGen(PropD,PHYCHdi[fet],PSq,NSq,ln,rx)
      AUC=Evaluate(Dat,L)
      AIRes[fet]=AUC
    dsort=OD(sorted (AIRes.items(),key=lambda x:x[1],reverse=True))
    ResAn[ln]=[x for x in dsort]
  KBF=KB_IntersectionUnion(ResAn)
  with open("{}/KB/KBF.json".format(path),'w') as f:
      json.dump(KBF,f)
  return KBF
path='./'
KBF=KBestFetSelection(path)

