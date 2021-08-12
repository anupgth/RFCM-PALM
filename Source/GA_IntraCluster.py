#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
#import pandas
from sklearn import model_selection
from sklearn.metrics import precision_score, \
        recall_score, confusion_matrix, classification_report, \
        accuracy_score, f1_score,roc_auc_score,matthews_corrcoef
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
#import os
import numpy as np
import datetime
from collections import OrderedDict
import multiprocessing as mp

def parseAAI(path):
    with open("{}/AAIscore.json".format(path),'r') as f:
        PHYCHdi=json.load(f)
    return(PHYCHdi)

def readTrData(ix,dtype,aikey,length):
    
    if dtype=='pos':
        fh=open("J:/Lab/LEN_DATA/%s/L%s/%s%s/%s.txt"%(gender,length,dtype.upper(),length,aikey),"r")
    else:
        fh=open("J:/Lab/LEN_DATA/%s/L%s/%s%s/%s.txt"%(gender,length,dtype.upper(),length,aikey),"r")
    for i,line in enumerate(fh):
        if i==ix:
            ll1=[float(x) for x in line.split('\n')[0].split(',')]

            c=np.array(ll1)

            return c

def dataPart(Xi,Yi,rt):
    s1=[];s2=[];s3=[];s4=[];s5=[]
    cs1=[];cs2=[];cs3=[];cs4=[];cs5=[]
#    np=len(Xi)/(2*5)
    try:
        a,b=Xi.shape
    except:
        print(len(Xi))
    for i in range(a):
        if i<rt:
            s1.append(Xi[i]);cs1.append(Yi[i])
        elif i<2*rt and i>=rt:
            s2.append(Xi[i]);cs2.append(Yi[i])
        elif i<3*rt and i>=2*rt:
            s3.append(Xi[i]);cs3.append(Yi[i])
        elif i<4*rt and i>=3*rt:
            s4.append(Xi[i]);cs4.append(Yi[i])
        elif i<5*rt and i>=4*rt:
            s5.append(Xi[i]);cs5.append(Yi[i])
        #**************************************
        elif i<6*rt and i>=5*rt:
            s1.append(Xi[i]);cs1.append(Yi[i])
        elif i<7*rt and i>=6*rt:
            s2.append(Xi[i]);cs2.append(Yi[i])
        elif i<8*rt and i>=7*rt:
            s3.append(Xi[i]);cs3.append(Yi[i])
        elif i<9*rt and i>=8*rt:
            s4.append(Xi[i]);cs4.append(Yi[i])
        elif i<10*rt and i>=9*rt:
            s5.append(Xi[i]);cs5.append(Yi[i])

    K=[s1,s2,s3,s4,s5];L=[cs1,cs2,cs3,cs4,cs5]
    return (K,L)

def get_split(I,J,ix):
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
    return trainX,trainY,testX,testY

def getNegInd():
    Ni=[]
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
	#population_sorted = sorted(population.items(), key=lambda t:t[1])
	if total_fitness==0.0:
         picky=random.randrange(len(population_sorted))
         for i,j in enumerate(population_sorted):
             if i==picky:
                 return j
	else:
     	 #for x,y in population.items() :
     	 for x,y in population_sorted.items() :
    		#print(x,y)
    		  FitnessSoFar += y
    		  if FitnessSoFar >= pick :
    			  return x


def Crossover(population,TotalFItness,offspring1,offspring2) :
	if (random.random() < CROSSOVER_RATE) :  
		crossover = random.randrange(CHROMO_LENGTH)
		one_cut = offspring1[crossover :]#; print(type(one_cut))
		two_cut = offspring2[crossover :]
		offspring1 = offspring1[:crossover] + two_cut
		offspring2 = offspring2[:crossover] + one_cut
	while(offspring1==offspring2):
		offspring2=Roulette(TotalFitness,population)
	return offspring1,offspring2

def Crossover_uni(offspring1,offspring2):
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
            
    ro,co=(Data.shape);
    
    XL,YL=dataPart(Data,Y,rt)
    scores=[]
    pool=mp.Pool(processes=5)
    scores=pool.starmap(CrossVal, [(j,XL,YL) for j in range(len(XL))])
    pool.close()
    f1_scores=0;auc_scores=0; precision_scores,recall_scores,accuracy_scores=0,0,0;mcc_scores=0   
    for f1,auc,mcc in scores:    
        f1_scores+=f1
        auc_scores+=auc
        mcc_scores+=mcc
        
    avg_f1=f1_scores/5
    avg_auc=auc_scores/5
    avg_mcc=mcc_scores/5
    return (avg_mcc,avg_f1,avg_auc)

def CrossVal(j,XL,YL):
    X_train,y_train,X_test,y_test=get_split(XL,YL,j)
    ro,co=X_train.shape
    #gm=1/co
    #clf =SVC(kernel='poly',gamma=0.02,C=1)
    clf =RF(n_estimators=300,min_samples_leaf =20)
    

    clf.fit(X_train, y_train)
    xx=clf.predict(X_test)
    yy=clf.decision_function(X_test)
    mcc=matthews_corrcoef(y_test, xx)
    
    return (f1_score(y_test, xx),roc_auc_score(y_test,yy),mcc)

def AssFitParalel_initial(y,XML,Y,results_queue_initial):
    try:
        avg_f1,avg_auc,avg_mcc=AssignFitness(y,XML,Y)
    except ValueError:
        print(y)
    results_queue_initial.put((y,avg_f1,avg_auc,avg_mcc))

def AssFitParalel(y,XML,Y,results_queue):
    avg_f1,avg_auc,avg_mcc=AssignFitness(y,XML,Y)
    results_queue.put((y,avg_f1,avg_auc,avg_mcc))

def paralel_result(result):
    Fitness.append(result)

gender='male'
if __name__ == '__main__':
    length=19
    CROSSOVER_RATE = 0.7
    MUTATION_RATE = 0.01
    #POP_SIZE = 10
    #CHROMO_LENGTH = 19
    GENE_LENGTH = 1
    MAX_ALLOWABLE_GENERATIONS = 100
    Fln=0.1;Dt='AAI_cluster_'
    fname="%s%s"%(Dt,Fln)
    aikey=[]
    clust_dict={}
    with open('./%s.csv'%fname,'r') as f:
        for line in f:
            tmp=line.split('\n')[0].split(',')
            clust_dict[tmp[0]]=tmp[1].rsplit(' ', 1)[0].split(' ')
    clust_dict.pop('singleton',None)


    f=open("./results/%s_res_male_intraCluster.txt"%fname,'w');f.close()
    for clust in clust_dict:
        if len(clust_dict[clust])==1:
            aikey=[i for i in clust_dict[clust]]
            Laikey=len(aikey)
            initial=1
            NoDval=length*2+1+length*2
            XML=np.empty([initial,Laikey,NoDval],dtype=float)
            XMLy=[]

            dtype="pos"
            rng=[i for i in range(1870)]
            random.seed(0)
            Ni=random.sample(rng,500)
            for m,i in enumerate(Ni):
                for j,k in enumerate(aikey):
                    xl=readTrData(i,dtype,k,length)
                    XML[m][j]=xl
                XMLy.append(1)
                initial+=1
                XML.resize((initial,Laikey,NoDval),refcheck=False)

            dtype="neg"
            #f=open("./LEN_DATA/femaleTrainNew/L%s/cystein_%s_%s.txt"%(length,dtype,length),"r")
            f=open("J:/Lab/LEN_DATA/%s/L%s/cystein_%s_%s.txt"%(gender,length,dtype,length),"r")
            g=f.readlines()
            f.close()
            neg_seq=len(g)
            rng=[i for i in range(neg_seq)]
            random.seed(0)
            Ni=random.sample(rng,500)
            #Ni=getNegInd()
            for i in Ni:
            #for i in range(1055):
                for j,k in enumerate(aikey):
                    xl=readTrData(i,dtype,k,length)
                    #XML.append(xl)
                    XML[initial-1][j]=xl
                XMLy.append(0)
                initial+=1
                XML.resize((initial,Laikey,NoDval),refcheck=False)

            Y=np.array(XMLy)
            print("%s Data Read completed"%(length))
            XML.resize((initial-1,Laikey,NoDval),refcheck=False)
            with open("./results/%s_res_male_intraCluster.txt"%fname,'a') as f:
                f.write("%s,%s,%s\n"%(clust,clust_dict[clust][0],AssignFitness('1',XML,Y)))

        else:
            fitness=0
            aikey=[i for i in clust_dict[clust]]
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
                    POP_SIZE=10
                initial=1
                NoDval=length*2+1+length*2
                XML=np.empty([initial,Laikey,NoDval],dtype=float)
                XMLy=[]

                dtype="pos"
                rng=[i for i in range(1870)]
                random.seed(0)
                Ni=random.sample(rng,500)
                for m,i in enumerate(Ni):
                    for j,k in enumerate(aikey):
                        xl=readTrData(i,dtype,k,length)
                        #XML.append(xl)
                        XML[m][j]=xl
                    XMLy.append(1)
                    initial+=1
                    XML.resize((initial,Laikey,NoDval),refcheck=False)

                dtype="neg"
                #f=open("./LEN_DATA/femaleTrainNew/L%s/cystein_%s_%s.txt"%(length,dtype,length),"r")
                f=open("J:/Lab/LEN_DATA/%s/L%s/cystein_%s_%s.txt"%(gender,length,dtype,length),"r")
                g=f.readlines()
                f.close()
                neg_seq=len(g)
                rng=[i for i in range(neg_seq)]
                random.seed(0)
                Ni=random.sample(rng,500)
                for i in Ni:
                    for j,k in enumerate(aikey):
                        xl=readTrData(i,dtype,k,length)
                        XML[initial-1][j]=xl
                    XMLy.append(0)
                    initial+=1
                    XML.resize((initial,Laikey,NoDval),refcheck=False)

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
                # initial population

                results_queue_initial = mp.Queue()
                processes_initial = [mp.Process(target=AssFitParalel_initial, args=(x,XML,Y,results_queue_initial,)) for x,_ in population.items()]
                #Fitness=[pool.apply_async(AssFitParalel, args=(x,)) for x,_ in population.items()]
                for p in processes_initial:
                    p.start()
                for p in processes_initial:
                    p.join()

                Fitness_initial = [results_queue_initial.get() for p in processes_initial]
                results_queue_initial.close()

                best=0;best1=""
                for r,s,roc,mcc in Fitness_initial:
                    if s>best:
                        best=s
                        best1=r
                    population[r]=s
                    if roc>GlBest_auc:
                        GlBest_auc=roc
                    if mcc>GlBest_mcc:
                        GlBest_mcc=mcc

                print("Finished assigning fitness to initial population\n")

                if POP_SIZE==2**Laikey:
                    aikey=[aikey[k] for k,l in enumerate(aikey) if best1[k]=='1']
                    GlBest_Fit=best
                    break

                gen = 0;saturation=0

                while (gen<MAX_ALLOWABLE_GENERATIONS and saturation<=35):

                    #print (i)
                    # getting the fitness scores of present population
                   TotalFitness = 0.0;Best_fitness=0.0;Best_chro="";count=0;Fitness=[];cnt1=0;cnt2=0;Best_auc=0.0;Best_mcc=0.0
                   #TotalFitness=0.0

                   for x,y in population.items():
                       TotalFitness += y
                       if y>Best_fitness:
                           Best_fitness=float(y)
                           Best_chro=x
                       if Best_fitness>GlBest_Fit:
                           GlBest_Fit=Best_fitness
                           GlBest_chro=Best_chro

                   child_population = {}
                   satu=0
                   while (len(child_population) < POP_SIZE):

                       parent1 = Roulette(TotalFitness,population)
                       parent2 = Roulette(TotalFitness,population)

                       "crossover"
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

                       satu+=1

                   print("=============================================================================")        
                   #print("Started assigning fitness to current population")
                   results_queue = mp.Queue()
                   processes = [mp.Process(target=AssFitParalel, args=(x,XML,Y,results_queue,)) for x,_ in child_population.items()]
                   for p in processes:
                       p.start()
                   for p in processes:
                       p.join()

                   Fitness = [results_queue.get() for p in processes]
                   results_queue.close()


                   for r,s,roc,mcc in Fitness:
                       child_population[r]=s
                       if roc>Best_auc:
                           Best_auc=roc
                       if mcc>Best_mcc:
                           Best_mcc=mcc

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

        #	population.update(child_population)
                   population=child_population

                   if Best_auc>GlBest_auc:
                       GlBest_auc=Best_auc
                   if Best_mcc>GlBest_mcc:
                       GlBest_mcc=Best_mcc

                   if Best_fitness_temp==GlBest_Fit:
                       #cnt1=0;cnt2=0
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
                   gen+=1
                   #break

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

                    #break

            with open("./results/%s_res_male_intraCluster.txt"%fname,'a') as f:
                sp=""
                for i in aikey:
                    sp+="%s\t"%i
                f.write('%s,%s,%s\n'%(clust,sp,(GlBest_Fit,GlBest_auc,GlBest_mcc)))

        print("%s complete"%clust)
        #break

    print("----------COMPLETE-----------")