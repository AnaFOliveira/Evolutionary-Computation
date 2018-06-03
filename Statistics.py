
from stat_functions import *
from utils import *
import pickle
import matplotlib.pyplot as plt
import numpy as np

crossover1='uniform'
mutation1='float_custom'
crossover2='TwoPoint'
mutation2='perm_scramble'
 
   
with open('Results/Best_ByGenarion_ByRun+'+crossover1+'+'+mutation1,'rb') as fp:
    boa1 = pickle.load(fp)
    
with open('Results/Best_ByGenarion_ByRun+'+crossover1+'+'+mutation2,'rb') as fp:
    boa2 = pickle.load(fp)

with open('Results/Best_ByGenarion_ByRun+'+crossover2+'+'+mutation2,'rb') as fp:
    boa3 = pickle.load(fp)

with open('Results/Best_ByGenarion_ByRun+'+crossover2+'+'+mutation1,'rb') as fp:
    boa4 = pickle.load(fp)
    



data=[boa1,boa2,boa3,boa4]
plt.boxplot(data)
describe_data(data)

test_normal(data,2) #p-values dao todos menores que 0.05 logo temos de fazer testes nao parametricos, visto que as amostras nao seguem uma distribuicao noraml

print(friedman_chi(data))#Friedmans anova(matched, non parametric)


#como pval menor que 0.05 as as amostras apresentam diferencas estatisticas significativas, assim vamos compar a opcao 2 com a 3 atraves da analise do boxplot para saber qual a melhor configuracao

print(wilcoxon(data[1],data[2]))
#como pvalue <0.05 concluimos que existem diferencas estatisticas entre a configuracao 2 e 3, logo do ponto de vista estatistico a configuracao 3 e a melhor!

