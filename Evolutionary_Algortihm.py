import numpy as np
from Auxiliar_functions import *
from numpy_neural_network import *
from operator import itemgetter
from random import random,randint,uniform, sample, shuffle,gauss
from Survivors import *
from Crossover import *
from Mutation import *
from utils import *
import pickle


def weights_transform(indiv,size_hidden_layer1,size_hidden_layer2):
    x=np.split(indiv,[13*size_hidden_layer1, 13*size_hidden_layer1+size_hidden_layer1*size_hidden_layer2],axis=0)# divide os pesos do individuo por camada
    #pesos_l1=np.split(x[0],[i for i in range(0,13*size_hidden_layer1,size_hidden_layer1)])
    pesos_l1=np.resize(x[0],(13,size_hidden_layer1))
    pesos_l2=np.resize(x[1],(size_hidden_layer1,size_hidden_layer2))
    pesos_l3=np.resize(x[2],(size_hidden_layer2,1))
    
    return pesos_l1,pesos_l2,pesos_l3

def gera_indiv_float(size_hidden_layer1,size_hidden_layer2): # gera os pesos para cada camada aleatoriamente e devolve um vetor de floats com todos os pesos
    pesos_l1=np.random.uniform(low=-1, high=1, size=(13,size_hidden_layer1))
    pesos_l2=np.random.uniform(low=-1, high=1, size=(size_hidden_layer1,size_hidden_layer2))
    pesos_l3=np.random.uniform(low=-1, high=1, size=(size_hidden_layer2,1))
    
    pesos_l1=np.concatenate(pesos_l1, axis=0)
    pesos_l2=np.concatenate(pesos_l2, axis=0)
    pesos_l3=np.concatenate(pesos_l3, axis=0)
    indiv=np.concatenate([pesos_l1,pesos_l2,pesos_l3],axis=0)
    
    return indiv

def gera_pop(size_pop,hid_l1,hid_l2):
    return [(gera_indiv_float(hid_l1,hid_l2),0) for i in range(size_pop)]


def fitness_function(indiv, size_hidden_layer1, size_hidden_layer2):
    X_train=np.load('X_train.npy',mmap_mode='r')
    X_test=np.load('X_test.npy',mmap_mode='r')
    y_train=np.load('y_train.npy',mmap_mode='r')
    y_test=np.load('y_test.npy',mmap_mode='r')

    pesos_l1,pesos_l2,pesos_l3=weights_transform(indiv,size_hidden_layer1,size_hidden_layer2)
    
    neural_network = NeuralNetwork(epochs=1000,batch_size=3,hidden_layer1_size=size_hidden_layer1,
                                   hidden_layer2_size=size_hidden_layer2,function_a='sigmoid',
                                   weights_l1=pesos_l1,weights_l2=pesos_l2,weights_l3=pesos_l3);
    neural_network.train(X_train, y_train)
    predicted = neural_network.predict(X_test)
    fitness=neural_network.accuracy(predicted, y_test)
    
    return fitness

def best_pop(populacao):
    populacao.sort(key=itemgetter(1),reverse=True)
    return populacao[0]
    
def average_pop(populacao):
    return sum([fit for cromo,fit in populacao])/len(populacao)


def sea_nn(populacao,numb_generations,hid_l1,hid_l2,pop_size,prob_cross,sel_parents,prob_mut,sel_survivors,recombination,mutation,fitness_func):
    best_gen=[]
    avg_gen=[]
    #populacao=gera_pop(pop_size,hid_l1,hid_l2)
    #populacao=[(indiv[0], fitness_func(indiv[0],hid_l1,hid_l2)) for indiv in populacao] 
    for i in range(numb_generations):
        mate_pool = sel_parents(populacao)
        progenitores = []
        for j in range(0,pop_size-1,2):
            indiv_1= mate_pool[j]
            indiv_2 = mate_pool[j+1]
            filhos = recombination(indiv_1,indiv_2, prob_cross)
            progenitores.extend(filhos) 
        
        descendentes = []
        for cromo,fit in progenitores:
            novo_indiv = mutation(cromo,prob_mut)
            descendentes.append((novo_indiv,fitness_func(novo_indiv,hid_l1,hid_l2)))
       
        populacao = sel_survivors(populacao,progenitores)
      
        populacao = [(indiv[0], fitness_func(indiv[0],hid_l1,hid_l2)) for indiv in populacao]  
        avg_gen.append(average_pop(populacao))
        best_gen.append(best_pop(populacao)[1])
        
    return best_pop(populacao),best_gen,avg_gen
   
def run(numb_runs,numb_generations,hid_l1,hid_l2,pop_size,prob_cross,sel_parents,prob_mut,sel_survivors,recombination,mutation,fitness_func):
    statistics = []
    bests=[]
    
    with open('ListaPopulations', 'rb') as fp:
        populacoes=pickle.load(fp)
     
    for i in range(numb_runs):
        print('Run->'+str(i))
        best,stat_best,stat_aver = sea_nn(populacoes[i],numb_generations,hid_l1,hid_l2,pop_size,prob_cross,sel_parents,prob_mut,sel_survivors,recombination,mutation,fitness_func)
        statistics.append(stat_best)
        bests.append(best)
    stat_gener = list(zip(*statistics))
    boa = [max(g_i) for g_i in stat_gener] 
    aver_gener =  [sum(g_i)/len(g_i) for g_i in stat_gener]
    return boa,aver_gener,bests


if __name__ == '__main__':
    number_runs=5
    number_gen=30
    hidden_layer1=15
    hidden_layer2=15
    pop_size=50
    prob_cross=0.8
    prob_muta=0.6
    tourn_sel=5 #numero de progenitores escolhidos
    elite_percentage=0.1
 
    # DESCOMENTAR LINHAS 120 A 122 PARA CORRER APENAS UMA RUN    
#    populacao=gera_pop(100,15,15)
#    populacao=[(indiv[0], fitness_function(indiv[0],15,15)) for indiv in populacao] 
#    
#    best_all,best_gen,avg_gen=sea_nn(populacao,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
#                                   uniform_cross,muta_float_custom,fitness_function)
#    display_stat_1([x for x in best_gen],avg_gen)
##        
#    boa,aver_gener,bests=run(number_runs,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
#           order_cross,muta_float_custom,fitness_function)
    
    opt=2# Escolher qual a configuracao a correr
    if(opt==1):  
        boa,aver_gener,bests=run(number_runs,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
           uniform_cross,muta_float_custom,fitness_function)
        
        crossover='uniform'
        mutation='float_custom'
        fig_name=crossover+'+'+mutation
        
        display_stat_n_save(boa,aver_gener,fig_name)
        
        with open('Best_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(boa, fp)
        
        with open('Avg_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(aver_gener, fp)
        
        with open('Bests_ByRun-'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(bests, fp)
    elif(opt==2):  
        boa,aver_gener,bests=run(number_runs,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
           uniform_cross,muta_perm_scramble,fitness_function)
        
        crossover='uniform'
        mutation='perm_scramble'
        fig_name=crossover+'+'+mutation
        
        display_stat_n_save(boa,aver_gener,fig_name)
        
        with open('Best_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(boa, fp)
        
        with open('Avg_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(aver_gener, fp)
        
        with open('Bests_ByRun-'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(bests, fp)
    elif(opt==3):  
        boa,aver_gener,bests=run(number_runs,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
           two_points_cross,muta_perm_scramble,fitness_function)
        
        crossover='TwoPoint'
        mutation='perm_scramble'
        fig_name=crossover+'+'+mutation
        
        display_stat_n_save(boa,aver_gener,fig_name)
        
        with open('Best_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(boa, fp)
        
        with open('Avg_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(aver_gener, fp)
        
        with open('Bests_ByRun-'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(bests, fp)
    elif(opt==4):  
        boa,aver_gener,bests=run(number_runs,number_gen,hidden_layer1,hidden_layer2,pop_size,prob_cross,tour_sel(tourn_sel),prob_muta,sel_survivors_elite(elite_percentage),
           two_points_cross,muta_float_custom,fitness_function)
        
        crossover='TwoPoint'
        mutation='float_custom'
        fig_name=crossover+'+'+mutation
        
        display_stat_n_save(boa,aver_gener,fig_name)
        
        with open('Best_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(boa, fp)
        
        with open('Avg_ByGenarion_ByRun+'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(aver_gener, fp)
        
        with open('Bests_ByRun-'+crossover+'+'+mutation, 'wb') as fp:
            pickle.dump(bests, fp)
            
        #    l1=15
        #    l2=15
        #    list_pop=[]
        #    for i in range(0,30):
            #populacao=gera_pop(2000,15,15)
            #populacao=[(indiv[0], fitness_function(indiv[0],15,15)) for indiv in populacao] 
        #        list_pop.append(populacao)
                
            
        #    populacao=[(indiv[0], fitness_function(indiv[0],l1,l2)) for indiv in populacao] 
        #    scores=[x[1] for x in populacao]
        #    scores.sort(reverse=True)