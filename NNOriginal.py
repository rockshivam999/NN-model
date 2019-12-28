# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:42:19 2019

@author: h4ck3rsh4d0w
"""
import numpy as np
import matplotlib.pyplot as plt
class nn:
    def __init__(self,inputN,hidden,output,lr=0.3):
        self.inputN=inputN
        self.hidden=hidden
        self.output=output
        self.lr=lr
        #initilizing weights for each layer
        self.wh=np.random.rand(self.hidden,self.inputN)-0.5
        self.wo=np.random.rand(self.output,self.hidden)-0.5
        self.activation=lambda x : 1/(1+np.exp(-x))
        self.cost=[]
        
        
    def train(self,inputs,targets):
    #feed forward section
        inputs=np.array(inputs,ndmin=2).T
        targets=np.array(targets,ndmin=2).T
        hiddinInput=np.dot(self.wh,inputs)
        hiddenOut=self.activation(hiddinInput)
        finalInput=np.dot(self.wo,hiddenOut)
        finalOut=self.activation(finalInput)
        
        #errors in output layer
        errorO=finalOut-targets
        self.cost.append(-abs(errorO[0]))
        #error in hidden layer
        errorInHidden=np.dot(self.wo.T,errorO)
        
        #updation
        self.wo-=self.lr*np.dot(errorO*finalOut*(1-finalOut),hiddenOut.T)
        self.wh-=self.lr*np.dot(errorInHidden*hiddenOut*(1-hiddenOut),inputs.T)
        
        
        
        
        
    def test(self,inputs):
    #feed forward only
        inputs=np.array(inputs,ndmin=2).T
        hiddenInput=np.dot(self.wh,inputs)
        hiddenOut=self.activation(hiddenInput)
        finalInput=np.dot(self.wo,hiddenOut)
        finalOutput=self.activation(finalInput)
        return finalOutput
        
nn=nn(2,4,1)#size of input layer,size of hidden layer , size of output layer
#training data
x = np.array([[0,0],
                [1,1],
                [1,0],
                [0,1],
                [1,1],
                [1,0],
                [0,1]])

y = np.array([[0],
                 [0],
                 [1],
                 [1],
                 [0],
                 [1],
                 [1]])
for i in range(50000):
    index=np.random.randint(len(x))
    item=x[index]
    nn.train(item,y[index])
    
    
print(nn.test(x[6]))
plt.plot(nn.cost)


