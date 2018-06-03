# Conjunto de funções para selecionar os survivors

from operator import itemgetter

# Elitism
def sel_survivors_elite(elite):
    def elitism(parents,offspring):
        size = len(parents)
        comp_elite = int(size* elite)
        offspring.sort(key=itemgetter(1),reverse=True)
        parents.sort(key=itemgetter(1),reverse=True)
        new_population = parents[:comp_elite] + offspring[:size - comp_elite]
        return new_population
    return elitism

# Generational
def survivors_generational(parents,offspring):
    return offspring

