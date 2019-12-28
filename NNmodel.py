# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 20:03:17 2019

@author: h4ck3rsh4d0w
"""

import numpy as np
import matplotlib.pyplot as plt
class nn:
    def __init__(self,Ninput,Nhidden,Noutput,lr=0.3):
        self.Ninput=Ninput
        self.Nhidden=Nhidden
        self.Noutput=Noutput
        self.wh=np.random.randn(self.Nhidden,self.Ninput)-0.5
        self.wo=np.random.randn(self.Noutput,self.Nhidden)-0.5
        self.lr=lr
        self.activation=lambda x:1/(1+np.exp(-x))#x:max(x,0)
        self.cost=[]
    def train(self,inputD,outputD):
        inputD=np.array(inputD,ndmin=2).T
        outputD=np.array(outputD,ndmin=2).T
        hidden_input=np.dot(self.wh,inputD)
        output_hidden=self.activation(hidden_input)
        input_outer=np.dot(self.wo,output_hidden)
        output_outer=self.activation(input_outer)
        
        error=output_outer-outputD
        self.cost.append(error[0])
        error_hidden=np.dot(self.wo.T,error)
        
        self.wo-=self.lr*np.dot(error*output_outer*(1-output_outer),output_hidden.T)
        self.wh-=self.lr*np.dot(error_hidden*output_hidden*(1-output_hidden),inputD.T)
     
        
        
        
        
    def test(self,inputD):
        inputD=np.array(inputD,ndmin=2).T
        hidden_input=np.dot(self.wh,inputD)
        output_hidden=self.activation(hidden_input)
        input_outer=np.dot(self.wo,output_hidden)
        output_outer=self.activation(input_outer)
        return output_outer
        
        
        
        
nn=nn(2,4,1)
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

