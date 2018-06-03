import numpy as np
import pickle



data_path = 'C:/Users/Asus/Documents/UNIVERSIDADE/4º ANO/2º SEMESTRE/Computação evolucionária/Projeto/Data/'  
file=open(data_path+'processed.cleveland.data.txt','r')
file=list(file.readlines())
for i in range(len(file)):
	file[i]=file[i].split(',')
data=np.array(file,dtype =None)
missing=np.where(data=='?')
data=np.delete(data,list(missing[0]),0)
data=data.astype('float64')
np.place(data[:,-1], data[:,-1]!=0, 1)

with open('Data_processed', 'wb') as f:
    pickle.dump(data, f)
f.close


