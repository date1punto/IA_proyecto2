__authors__ = ['1632398', '1633405', '1630320']
__group__ = 'DJ.12'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        
    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.train_data = np.random.randint(8, size=[10, 14400])
    
        self.train_data=np.float32(train_data.reshape(train_data.shape[0], -1))
        
        
    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.neighbors = np.random.randint(k, size=[test_data.shape[0], k])
        
        test_data = np.float32(test_data.reshape(-1, self.train_data.shape[1]))
        distancies=cdist(test_data, self.train_data)
        self.neighbors=self.labels[np.argsort(distancies, axis=1)[:, :k]]
        return self.neighbors

    def get_class(self):
        """
        Get the class by maximum voting
        : return : numpy array of Nx1 elements .
            For each of the rows in self . neighbors gets the most
            voted value ( i . e . the class at which that row belongs )

        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #return np.random.randint(10, size=self.neighbors.size), np.random.random(self.neighbors.size)
        """guanyadors=[]
        #percentatges=[]
        for i in self.neighbors:#falta mirar si se produce empate quien gana
            a,b=np.unique(i, return_counts=1) 
            maxim=np.max(b)
            fi=0
            if len(a[maxim==b])>1:
                for x in i:
                    if fi==1:
                        break
                    for y in a[b==maxim]:
                        if x==y:
                            guanya=y
                            fi=1
                            break
            else:
                guanya=a[np.argmax(b)]           
            #percentatges.append(max(b)/sum(b)*100)
            guanyadors.append(guanya)
            
        return np.array(guanyadors)"""
       
        classe=[]
        #percentatge=[]
        for aux in self.neighbors:
            d={}
            for i in aux:
                if i in d:
                    d[i]+=1
                else:
                    d[i]=1
            
            maxValue=max(d.values())
            
            for i,j in d.items():
                if j==maxValue:
                   # percentatge.append(j/len(self.neighbors)*100)
                    classe.append(i)
                    break;
        return np.array(classe)
            
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class  1st the class
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
