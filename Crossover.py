# -*- coding: utf-8 -*-
"""
Created on Sat May 12 20:31:46 2018

@author: G_N17
"""

# Conjunto de funções para Crossover
from random import random,randint,uniform, sample, shuffle,gauss
import numpy as np
from operator import itemgetter
import random



def two_points_cross(indiv_1, indiv_2,prob_cross):
    size = len(indiv_1[0])
    value = random.random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        pc= sample(range(size),2)
        pc.sort()
        pc1,pc2 = pc
        f1= np.vstack([cromo_1[:pc1,None],cromo_2[pc1:pc2,None],cromo_1[pc2:,None]])
        f2= np.vstack([cromo_2[:pc1,None],cromo_1[pc1:pc2,None],cromo_2[pc2:,None]])
        f1=f1[:,0]
        f2=f2[:,0]
        return ((f1,0),(f2,0))
    return  indiv_1,indiv_2    
    
def uniform_cross(indiv_1, indiv_2,prob_cross):
    size = len(indiv_1[0])
    value = random.random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1=[]
        f2=[]
        for i in range(0,size):
            if random.random() < 0.5:
                f1.append(cromo_1[i])
                f2.append(cromo_2[i])
            else:
                f1.append(cromo_2[i])
                f2.append(cromo_1[i])
        return ((f1,0),(f2,0))
    return  indiv_1,indiv_2    

# OX - order crossover

def order_cross(indiv_1, indiv_2,prob_cross):
    size = len(indiv_1[0])
    value = random.random()
    if value < prob_cross:
        cromo_1 = indiv_1[0]
        cromo_2 = indiv_2[0]
        f1 = [None]* size
        f2 = [None] * size
        pc= sample(range(size),2)
        pc.sort()
        pc1,pc2 = pc
        f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
        f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
        for j in range(size):
            for i in range(size):
                if (cromo_2[j] not in f1) and (f1[i] == None):
                    f1[i] = cromo_2[j]
                    break
            for k in range(size):
                if (cromo_1[j] not in f2) and (f2[k] == None):
                    f2[k] = cromo_1[j]
                    break
        return ((f1,0),(f2,0))
    return  indiv_1,indiv_2    
    
    
def pmx_cross(indiv_1, indiv_2,prob_cross):
    size = len(indiv_1[0])
    value = random()
    if value < prob_cross:
        cromo_1 = indiv_1[0] 
        cromo_2 = indiv_2[0]
        pc= sample(range(size),2)
        pc.sort()
        pc1,pc2 = pc
        f1 = [None] * size
        f2 = [None] * size
        f1[pc1:pc2+1] = cromo_1[pc1:pc2+1]
        f2[pc1:pc2+1] = cromo_2[pc1:pc2+1]
        # primeiro filho
        # parte do meio
        for j in range(pc1,pc2+1):
            if cromo_2[j] not in f1:
                pos_2 = j
                g_j_2 = cromo_2[pos_2]
                g_f1 = f1[pos_2]
                index_2 = cromo_2.index(g_f1)
                while f1[index_2] != None:
                    index_2 = cromo_2.index(f1[index_2])
                f1[index_2] = g_j_2
        # restantes
        for k in range(size):
            if f1[k] == None:
                f1[k] = cromo_2[k]
        # segundo filho    
        # parte do meio
        for j in range(pc1,pc2+1):
            if cromo_1[j] not in f2:
                pos_1 = j
                g_j_1 = cromo_1[pos_1]
                g_f2 = f2[pos_1]
                index_1 = cromo_1.index(g_f2)
                while f2[index_1] != None:
                    index_1 = cromo_1.index(f2[index_1])
                f2[index_1] = g_j_1
        # parte restante
        for k in range(size):
            if f2[k] == None:
                f2[k] = cromo_1[k]                
        return ((f1,0),(f2,0))
    return  indiv_1,indiv_2  


# Tournament Selection
def tour_sel(t_size):
    def tournament(pop):
        size_pop= len(pop)
        mate_pool = []
        for i in range(size_pop):
            winner = tour(pop,t_size)
            mate_pool.append(winner)
        return mate_pool
    return tournament

def tour(population,size):
    """Minimization Problem.Deterministic"""
    pool = sample(population, size)
    pool.sort(key=itemgetter(1),reverse=True)
    return pool[0]

