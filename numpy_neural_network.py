"""
    Adapted from https://github.com/lucko515/fully-connected-nn
"""
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    
    def __init__(self,epochs,batch_size,hidden_layer1_size,hidden_layer2_size,function_a,weights_l1,weights_l2,weights_l3):
        ''' 
        This constructor is used to initilize hyperparams for our network
        Inputs: learning_rata -  how fast are you going to train the network
                Epochs -  how many times are you going to run forward and backward pass
                batch_size -  how many samples are you feeding into netowrk at ones
        '''
        self.epochs = epochs
        self.batch_size = batch_size
        self.hidden_layer1_size=hidden_layer1_size;
        self.hidden_layer2_size=hidden_layer2_size;
        self.function_a=function_a; 
        #define the weights and bias
        #self.weigts_one = np.random.randn(13, self.hidden_layer1_size)
        self.weigts_one = weights_l1
        self.bias_one = np.zeros((1, self.hidden_layer1_size))
        #self.weigts_two = np.random.randn(self.hidden_layer1_size, self.hidden_layer2_size)
        self.weigts_two = weights_l2
        self.bias_two = np.zeros((1, self.hidden_layer2_size))
        #self.weighs_three = np.random.randn(self.hidden_layer2_size, 1)
        self.weighs_three = weights_l3
        self.bias_three = np.zeros((1, 1))
        

    def function_act(self,x):
        if(self.function_a=='sigmoid'):
            return 1 / (1 + np.exp(-x))
        elif(self.function_a=='tanh'):
            return np.tanh(x)
        elif(self.function_a=='relu'):
            return np.maximum(x, 0)
        elif(self.function_a=='softmax'):
            return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        elif(self.function_a=='gaussian'):
            return np.exp(np.power((-x), 2))
        elif(self.function_a=='sin'):
            return np.sin(x)
        
    def train(self, X ,y):
        y_train = np.reshape(y, (len(y), 1))
        
        #Training loop
        for i in range(self.epochs):
            
            idx = np.random.choice(len(X), self.batch_size, replace=True)
            X_batch = X[idx, :]
            y_batch = y_train[idx, :]
            l1, l2, scores = self.forward(X_batch)
            
            cost = y_batch - scores
                      
    def forward(self, X):
        l1 = self.function_act((np.dot(X, self.weigts_one) + self.bias_one))
        l2 = self.function_act((np.dot(l1, self.weigts_two) + self.bias_two))
        scores = self.function_act((np.dot(l2, self.weighs_three) + self.bias_three))
        return l1, l2, scores
    
    def predict(self, X):
        l1, l2, scores = self.forward(X)
        pred = []
        for i in range(len(scores)):
            if scores[i] >= 0.5:
                pred.append(1)
            else:
                pred.append(0)
        return pred
    
    def accuracy(self, pred, y_test):
        assert len(pred) == len(y_test)
        true_pred = 0
        
        cm=confusion_matrix(y_test, pred)
        #print(cm)
               
        for i in range(len(pred)):
            if pred[i] == y_test[i]:
                true_pred += 1
                
        acc=(true_pred/len(pred))*100
        #print(acc, "%")
        return acc
  