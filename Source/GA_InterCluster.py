#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from sklearn import model_selection
from sklearn.metrics import precision_score, \
        recall_score, confusion_matrix, classification_report, \
        accuracy_score, f1_score,roc_auc_score,matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
import numpy as np
import datetime
from collections import OrderedDict
import multiprocessing as mp
import json

def parseAAI(path):
    with open("{}/AAIscore.json".format(path),'r') as f:
        PHYCHdi=json.load(f)
    return(PHYCHdi)

def readTrData(ix,dtype,aikey,length):
    
    if dtype=='pos':
        fh=open("C:/Lab/LEN_DATA/%s/L%s/%s%s/%s.txt"%(gender,length,dtype.upper(),length,aikey),"r")
    else:
        fh=open("C:/Lab/LEN_DATA/%s/L%s/%s%s/%s.txt"%(gender,length,dtype.upper(),length,aikey),"r")
    for i,line in enumerate(fh):
        if i==ix:
            c=np.array([float(x) for x in line.split('\n')[0].split(',')])
            break
    fh.close()
    return c

def dataPart(Xi,Yi):
    s0=[];s1=[];s2=[];s3=[]#;s4=[];s5=[]
    cs0=[];cs1=[];cs2=[];cs3=[]#;cs4=[];cs5=[]
#    np=len(Xi)/(2*5)
    #part=inst/3
    try:
        a,b=Xi.shape
    except:
        print(len(Xi))
        
    set1_pos=[];set2_pos=[];set3_pos=[]
    set1_neg=[];set2_neg=[];set3_neg=[];setNone_neg=[]
    with open("C:/Lab/LEN_DATA/%s/L%s/cystein_%s_%s.txt"%(gender,length,"pos",length),"r") as f:
        for i,line in enumerate(f):
            pid=line.split("\n")[0].split()[0]
            if i<200:
                set1_pos.append(pid)
            elif i<400:
                set2_pos.append(pid)
            else:
                set3_pos.append(pid)
        with open('./Random_Neg.json') as h:
            negIndex=json.load(h)
        negIndex.sort()
        negIndex=negIndex[:inst]
        for index,lineNo in enumerate(negIndex):
            g=open("C:/Lab/LEN_DATA/%s/L%s/cystein_%s_%s.txt"%(gender,length,"neg",length),"r")
            for i,line in enumerate(g):
                if i == lineNo:
                    pid=line.split("\n")[0].split()[0]
                    if pid in set1_pos:
                        if pid in set2_pos or pid in set3_pos:
                            #print(pid)
                            pass
                        if len(set1_neg)<200:
                            set1_neg.append(index)
                        else:
                            setNone_neg.append(index)
                    elif pid in set2_pos:
                        if pid in set1_pos or pid in set3_pos:
                            #print(pid)
                            pass
                        if len(set2_neg)<200:
                            set2_neg.append(index)
                        else:
                            setNone_neg.append(index)
                    elif pid in set3_pos:
                        if pid in set1_pos or pid in set2_pos:
                            #print(pid)
                            pass
                        if len(set3_neg)<200:
                            set3_neg.append(index)
                        else:
                            setNone_neg.append(index)
                    else:
                        setNone_neg.append(index)
            g.close()
                
    set1_neg=np.array(set1_neg)+inst
    set2_neg=np.array(set2_neg)+inst
    set3_neg=np.array(set3_neg)+inst
    setNone_neg=np.array(setNone_neg)+inst
    #print(len(setNone_neg))
          
    for i in range(a):
        if i<200:
            s1.append(Xi[i]);cs1.append(Yi[i])
        elif i<400:
            s2.append(Xi[i]);cs2.append(Yi[i])
        elif i<600:
            s3.append(Xi[i]);cs3.append(Yi[i])
        else:
            if i in set1_neg:
                s1.append(Xi[i]);cs1.append(Yi[i])
            if i in set2_neg:
                s2.append(Xi[i]);cs2.append(Yi[i])
            if i in set3_neg:
                s3.append(Xi[i]);cs3.append(Yi[i])
            if i in setNone_neg:
                s0.append(Xi[i]);cs0.append(Yi[i])
    
    for i in range(len(s0)):
        if len(s1)<400:
            s1.append(s0[i]);cs1.append(cs0[i])
        elif len(s2)<400:
            s2.append(s0[i]);cs2.append(cs0[i])
        else:
            s3.append(s0[i]);cs3.append(cs0[i])
    
    K=[s1,s2,s3];L=[cs1,cs2,cs3];M=[s0,cs0]
    return (K,L,M)

def get_split(ix,I,J,K):
    #print(ix,len(I),len(J),len(K))
    trainX=[];testX=[];trainY=[];testY=[]
    for i in range(len(I)):
        if i!=ix:
            trainX+=I[i]
            trainY+=J[i]
        else:
            testX=I[i]
            testY=J[i]
    
    trainX=np.array(trainX)
    trainY=np.array(trainY)
    testX=np.array(testX)
    testY=np.array(testY)
    print(trainX.shape,trainY.shape,testX.shape,testY.shape)
    return trainX,trainY,testX,testY

def getNegInd():
    Ni=[]
    #fh=open("D:\\Google Drive\\library\\JU\\Thesis\\Mouse\\bestFinalNegRes.csv",'r')
    fh=open("./bestFinalNegRes.csv","r")
    for i,line in enumerate(fh):
        if i>0:
            Ni.append(int(line.split(",")[0]))
    return(Ni)
    
def GetRandomBits(x) :
	bits = ""
	for i in range(x) :
		if (random.random() > 0.5) :
			bits += "1"
		else :
			bits += "0"
	return bits

def Roulette(total_fitness,population_sorted) :
	pick = random.uniform(0,total_fitness)
	FitnessSoFar = 0.0
	for x,y in population_sorted.items() :
		FitnessSoFar += y
		if FitnessSoFar >= pick :
			return x


def Crossover(population,TotalFitness,offspring1,offspring2) :
	if (random.random() < CROSSOVER_RATE) :  # ***if (random.random() < fitness) :
		crossover = random.randrange(CHROMO_LENGTH)
		one_cut = offspring1[crossover :]#; print(type(one_cut))
		two_cut = offspring2[crossover :]
		offspring1 = offspring1[:crossover] + two_cut
		offspring2 = offspring2[:crossover] + one_cut
	while(offspring1==offspring2):
		offspring2=Roulette(TotalFitness,population)
	return offspring1,offspring2

def Crossover_uni(population,TotalFitness,offspring1,offspring2):
    if (random.random() < CROSSOVER_RATE):
        zygote1=[offspring1[i] if random.random()<0.5 else offspring2[i] for i in range(len(offspring1))]
        zygote2=[offspring1[i] if random.random()<0.5 else offspring2[i] for i in range(len(offspring1))]
        while(zygote1==zygote2):
            zygote2=[offspring1[i] if random.random()<0.5 else offspring2[i] for i in range(len(offspring1))]
        zygote1=''.join(zygote1)
        zygote2=''.join(zygote2)
        offspring1=zygote1
        offspring2=zygote2
    return offspring1,offspring2


def Mutate(bits) :
	temp = [i for i in bits]
	for j,k in enumerate(temp) :
		if (random.uniform(0.00,0.1) < MUTATION_RATE) :
			if k == "1" :
				temp[j] = "0"
			else :
				temp[j] = "1"
	s = ''.join(temp)
	return s

def AssignFitness(bits,XML,Y):
    global GlBest_auc
    temp = []
    for i,j in enumerate(bits):
        if j == "1" :
            temp.append(i)
	#del X
    if len(temp)<1:
        return (0,0,0)
    if len(temp)<2:
        Data=XML[:,temp[0]]
    elif len(temp)<3:
        Data=np.concatenate((XML[:,temp[0]],XML[:,temp[1]]),axis=1)	
    else:
        Data=np.concatenate((XML[:,temp[0]],XML[:,temp[1]]),axis=1)
        for i in temp[2:]:
            Data=np.concatenate((Data,XML[:,i]),axis=1)
            
    ro,co=(Data.shape);#incr=0.000;gm=((1.0/co)+incr)
    print("\ntraining data dim = %s,%s"%(ro,co))
    
    XL,YL,ML=dataPart(Data,Y)
    print("\nData Partition completed")
    scores=[]
    pool=mp.Pool(processes=5)
    scores=pool.starmap(CrossVal, [(j,XL,YL,ML) for j in range(len(XL))])
    pool.close()
    f1_scores=0;auc_scores=0; precision_scores,recall_scores,accuracy_scores=0,0,0;mcc_scores=0
    for f1,auc,mcc in scores:    
        f1_scores+=f1
        auc_scores+=auc
        mcc_scores+=mcc
        
    avg_f1=f1_scores/5
    avg_auc=auc_scores/5
    avg_mcc=mcc_scores/5
    return (avg_f1,avg_auc,avg_mcc)

def CrossVal(j,XL,YL,ML):
    X_train,y_train,X_test,y_test=get_split(j,XL,YL,ML)
    ro,co=X_train.shape
    #gm=1/co
    #clf =SVC(kernel='poly',gamma=0.002,C=1)
    clf = RandomForestClassifier(n_estimators=220,min_samples_leaf=10)
    #clf =svm.SVC(kernel='rbf',gamma=gm)
    clf.fit(X_train, y_train)
    xx=clf.predict(X_test)
    mcc=matthews_corrcoef(y_test, xx)
    return (f1_score(y_test, xx),roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]),mcc)

def AssFitParalel_initial(y,XML,Y,results_queue_initial):
    avg_f1,avg_auc,avg_mcc=AssignFitness(y,XML,Y)
    print(y)
    results_queue_initial.put((y,avg_f1,avg_auc,avg_mcc))

def AssFitParalel(y,XML,Y,results_queue):
    avg_f1,avg_auc,avg_mcc=AssignFitness(y,XML,Y)
    results_queue.put((y,avg_f1,avg_auc,avg_mcc))

def paralel_result(result):
    Fitness.append(result)

gender='male' # 'Female' # 'Combined'
length=19
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
GENE_LENGTH = 1
MAX_ALLOWABLE_GENERATIONS = 100

inst=600

if __name__ == '__main__':
    Fln=0.1;Dt='AAI_cluster_'
    #Fln=25_19;Dt='IB';
    fname="%s%s"%(Dt,Fln)

    aikey=[]
    with open('./results/%s_res_%s_intraCluster.txt'%(fname,gender),'r') as f:
        for line in f:
            tmp=line.split('\n')[0].split(',')[1].rsplit('\t',1)[0].split('\t')
            #print(tmp)
            for i in tmp:
                aikey.append(i)

    f=open("./results/%s_res_%s_interCluster.txt"%(fname,gender),'w');f.close()
    fitness=0
    POP_SIZE=0
    CHROMO_LENGTH=0
    
    while(True):
        Laikey=len(aikey)
        if Laikey==1:
            break
        CHROMO_LENGTH=Laikey
        if Laikey<=3:
            POP_SIZE=2**Laikey
        else:
            POP_SIZE=15
        initial=1
        NoDval=length*2+1+length*2
        XML=np.empty([initial,Laikey,NoDval],dtype=float)
        XMLy=[]
        dtype="pos"
        rng=[i for i in range(1870)]
        random.seed(0)
        for m,i in enumerate(rng):
            for j,k in enumerate(aikey):
                xl=readTrData(i,dtype,k,length)
                XML[m][j]=xl
            XMLy.append(1)
            initial+=1
            addRow = np.empty([1,Laikey,NoDval],dtype=float)
            XML = np.concatenate([XML,addRow])
            if m%(inst//10)==0:
                print("%s pos complete"%m)
            if m==600:
                break

        dtype="neg"
        with open('./Random_Neg.json') as f:
            negIndex=json.load(f)
        negIndex.sort()
        #Ni=getNegInd()
        for m,i in enumerate(negIndex):
            for j,k in enumerate(aikey):
                xl=readTrData(i,dtype,k,length)
                XML[initial-1][j]=xl
            XMLy.append(0)
            initial+=1
            addRow = np.empty([1,Laikey,NoDval],dtype=float)
            XML = np.concatenate([XML,addRow])
            if m%(inst/10)==0:
                print("%s neg complete"%m)
            if m==600:
                break

        Y=np.array(XMLy)
        print("%s Data Read completed"%(length))
        XML.resize((initial-1,Laikey,NoDval),refcheck=False)

        # constants
        GlBest_Fit=0.0;GlBest_chro="";GlBest_auc=0.0;GlCount=0;GlBest_mcc=0.0
        population = {}
        bits=['1' for _ in range(CHROMO_LENGTH)]
        bits=''.join(bits)
        population[bits]=0
        while (len(population) < POP_SIZE):#for i in range(POP_SIZE):
                bits = GetRandomBits(CHROMO_LENGTH)
                population[bits] = 0

        print("Started assigning fitness to initial population\n")

        results_queue_initial = mp.Queue()
        processes_initial = [mp.Process(target=AssFitParalel_initial, args=(x,XML,Y,results_queue_initial,)) for x,_ in population.items()]
        for p in processes_initial:
            p.start()
        for p in processes_initial:
            p.join()

        Fitness_initial = [results_queue_initial.get() for p in processes_initial]
        results_queue_initial.close()

        for r,f1,roc,mcc in Fitness_initial:
            population[r]=mcc
            if roc>GlBest_auc:
                GlBest_auc=roc
            if mcc>GlBest_mcc:
                GlBest_mcc=f1

        print("Finished assigning fitness to initial population\n")


        gen = 0;saturation=0

        while (gen<MAX_ALLOWABLE_GENERATIONS and saturation<=50):


           TotalFitness = 0.0;Best_fitness=0.0;Best_chro="";count=0;Fitness=[];cnt1=0;cnt2=0;Best_auc=0.0;Best_mcc=0.0

           for x,y in population.items():
               TotalFitness += y
               if y>Best_fitness:
                   Best_fitness=float(y)
                   Best_chro=x
               if Best_fitness>GlBest_Fit:
                   GlBest_Fit=Best_fitness
                   GlBest_chro=Best_chro

           child_population = {}
           while (len(child_population) < POP_SIZE):

               "selection"
               parent1 = Roulette(TotalFitness,population)
               parent2 = Roulette(TotalFitness,population)
               while parent1==parent2:
                   parent2 = Roulette(TotalFitness,population)
               offspring1,offspring2 = Crossover(population,TotalFitness,parent1,parent2)
               if offspring1 in child_population or offspring2 in child_population:
                   if offspring1 not in child_population:
                       if offspring1==parent1:
                           child_population[offspring1] = 0
                       else:
                           offspring1 = Mutate(offspring1)
                           child_population[offspring1] = 0                        
                   if offspring2 not in child_population:
                       if offspring2==parent2:
                           child_population[offspring2] = 0
                       else:
                           offspring2 = Mutate(offspring2)
                           child_population[offspring2] = 0
                   continue
               if offspring1==parent1 and offspring2==parent2:
                   child_population[offspring1] = 0
                   child_population[offspring2] = 0
                   continue

               "mutation"
               offspring1 = Mutate(offspring1)
               offspring2 = Mutate(offspring2)
               child_population[offspring1] = 0
               child_population[offspring2] = 0


           print("=============================================================================")        

           results_queue = mp.Queue()
           processes = [mp.Process(target=AssFitParalel, args=(x,XML,Y,results_queue,)) for x,_ in child_population.items()]
               p.start()
           for p in processes:
               p.join()

           Fitness = [results_queue.get() for p in processes]
           results_queue.close()


           for r,f1,roc,mcc in Fitness:
               child_population[r]=mcc
               if roc>Best_auc:
                   Best_auc=roc
               if mcc>Best_mcc:
                   Best_mcc=f1

           Best_fitness_temp=0;Best_chro_temp="";Worst_fitness_temp=list(child_population.values())[0];Worst_chro=list(child_population.keys())[0]

           for x,y in child_population.items():
               if y>Best_fitness_temp:
                   Best_fitness_temp=float(y)
                   Best_chro_temp=x
               if y<Worst_fitness_temp:
                   Worst_fitness_temp=float(y)
                   Worst_chro=x

           if Best_fitness_temp<Best_fitness:
               child_population.pop(Worst_chro, None)
               child_population[Best_chro]=Best_fitness

           for k,l in enumerate(Best_chro_temp):
                   if l=="1":
                       count+=1

           population=child_population

           if Best_auc>GlBest_auc:
               GlBest_auc=Best_auc
           if Best_mcc>GlBest_mcc:
               GlBest_mcc=Best_mcc

           if Best_fitness_temp==GlBest_Fit:
               for k,l in enumerate(GlBest_chro):
                   if l=="1":
                       cnt1+=1
               for k,l in enumerate(Best_chro_temp):
                   if l=="1":
                       cnt2+=1
               if cnt2<cnt1:
                   GlBest_chro=Best_chro_temp
                   saturation=0
               else:
                   saturation+=1
           elif Best_fitness_temp>GlBest_Fit:
               GlBest_Fit=Best_fitness_temp
               GlBest_chro=Best_chro_temp
               saturation=0
           else:
               saturation+=1
           print("saturation:%s"%saturation)
           msg1 = "Best Fitness in generation number %s : %.3f || Number of features in best chromosome %s || Best Fitness till now: %.3f\n" % (gen+1,Best_fitness_temp,count,GlBest_Fit)
           print(msg1)
           print("Best F1 till now %s || Best auc till now %s"%(GlBest_mcc,GlBest_auc))
           gen+=1

        for k,l in enumerate(GlBest_chro):
            if l=="1":
               GlCount+=1
        msg3="Global Best Fitness: %.3f || Number of features in global best Chromosome: %s\n" % (GlBest_Fit,GlCount)
        print(msg3)

        if GlBest_Fit>fitness:
            fitness=GlBest_Fit
            aikey=[aikey[k] for k,l in enumerate(aikey) if GlBest_chro[k]=='1']
        elif GlBest_Fit==fitness:
            if GlCount<Laikey:
                aikey=[aikey[k] for k,l in enumerate(aikey) if GlBest_chro[k]=='1']
            else:
                break
        else:
            break


    with open("./results/%s_res_%s_interCluster.txt"%(fname,gender),'a') as f:
        sp=""
        for i in aikey:
            sp+="%s\n"%i
        f.write('%s'%(sp))
        f.write("%s %s %s"%(fitness,GlBest_auc,GlBest_mcc))

    print("----------COMPLETE-----------")
