import pickle
from sklearn.model_selection import train_test_split


def split_data():
    #Open file
    file = open("Data_processed",'rb')
    data = pickle.load(file)
    file.close
    #Divide data train and test
    X=data[:,0:-1]
    Y=data[:,-1]
    #unique, counts = np.unique(Y, return_counts=True)
    #print (np.asarray((unique, counts)).T)
    X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
    
    np.save('X_train',X_train.astype('float64'))
    np.save('X_test',X_test.astype('float64'))
    np.save('y_train',y_train.astype('float64'))
    np.save('y_test',y_test.astype('float64'))
    
    return X_train,X_test,y_train,y_test



