from random import random,randint,uniform, sample, shuffle,gauss
import numpy as np
import random as r

def muta_float_custom(indiv, prob_muta):
    cromo=np.array(indiv,dtype='float64')
    value = random()
    if value < prob_muta:
        index = sample(range(len(indiv)),round(np.size(indiv)*0.2))
        index.sort()
        for i in range(np.size(index)):
            j=index[i]
            cromo[j]=(r.uniform(-1,1))
    return cromo

def muta_perm_scramble(indiv,prob_muta):
    cromo = np.array(indiv,dtype='float64')
    size=len(indiv)
    value = random()
    if value < prob_muta:
        index = sample(range(size),2)
        index.sort()
        i1,i2 = index
        scramble = cromo[i1:i2+1]
        shuffle(scramble) 
        cromo = np.hstack([cromo[:i1],scramble,cromo[i2+1:]])
    return cromo


def muta_perm_swap(indiv, prob_muta):
    cromo = indiv[:]
    value = random()
    if value < prob_muta:
        index = sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        cromo[i1],cromo[i2] = cromo[i2], cromo[i1]
    return cromo

def muta_perm_inversion(indiv,prob_muta):
    cromo = indiv[:]
    value = random()
    if value < prob_muta:
        index = sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        inverte = []
        for elem in cromo[i1:i2+1]:
            inverte = [elem] + inverte
            cromo = cromo[:i1] + inverte + cromo[i2+1:]
    return cromo

def muta_perm_insertion(indiv, prob_muta):
    cromo = indiv[:]
    value = random()
    if value < prob_muta:
        index = sample(range(len(cromo)),2)
        index.sort()
        i1,i2 = index
        gene = cromo[i2]
        for i in range(i2,i1,-1):
            cromo[i] = cromo[i-1]
            cromo[i1+1] = gene
    return cromo





def muta_float_gaussian(indiv, prob_muta, domain, sigma):
    cromo = indiv[:]
    for i in range(len(cromo)):
        cromo[i] = muta_float_gene(cromo[i],prob_muta, domain[i], sigma[i])
    return cromo

def muta_float_gene(gene,prob_muta, domain_i, sigma_i):
    value = random()
    new_gene = gene
    if value < prob_muta:
        muta_value = gauss(0,sigma_i)
        new_gene = gene + muta_value
        if new_gene < domain_i[0]:
            new_gene = domain_i[0]
        elif new_gene > domain_i[1]:
            new_gene = domain_i[1]
    return new_gene