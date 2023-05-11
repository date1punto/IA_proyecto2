__authors__ = ['1632398', '1633405', '1630320']
__group__ = 'DJ.12'

import numpy as np
import utils



class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        #self.X = np.random.rand(100, 5)
        
        self.X=np.float32(X)
        if X.ndim==3 and X.shape[2]==3:
           self.X=np.reshape(X,(-1, 3))
        else:
           self.X=np.reshape(X, (-1, X.ndim))#en el cas que no sigui imatge N X D



    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################
        
            

    def _init_centroids(self): #posibles fallos con el old_centroids
        """
        Initialization of centroids
        """
    
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        

        if self.options['km_init'].lower() == 'first':
           valors, index=np.unique(self.X, return_index=True, axis=0)
           index=np.sort(index)
           self.centroids=np.array((self.X[index[:self.K]]), np.float64, copy=True)
         
        
        elif self.options['km_init'].lower() == 'custom':
         #los una de las diagonales seran 255-n,255-n, 255-n donde n puede tener cualquier valor positivo (empezar en un vertice y acabar en el contrario)
         #se podra tambien empezar des de 0 o 255 ejemplo (0,255, 0)  
             diagonals=np.random.choice([0,1],size=(self.K, self.X.shape[1]))
             contrari=np.ones((self.K, self.X.shape[1])) + diagonals*-2
             self.centroids=diagonals*255 + (contrari* np.random.rand(self.K).reshape(self.K,-1)*255)
             
        #  punts distribuïts sobre la diagonal del hipercub de les dades?
        else:#255 porque los valors van de [0, 255] y random.rand devuelve de [0,1]
            self.centroids = np.random.rand(self.K, self.X.shape[1])*255 #no asegura que no este repetido
         
        self.old_centroids = np.full([self.K, 3], np.nan) 
        #self.old_centroids=np.empty([self.K, 3]) 
        #self.old_centroids[:]=np.nan

    def get_labels(self):
        """        Calculates the closest centroid of all points in X
        and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distancies=distance(self.X, self.centroids)
        self.labels = np.argmin(distancies, axis=1)

    
    
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids=np.array(self.centroids, copy=True)
        
        punts_per_clase=np.bincount(self.labels).reshape(-1, 1)
        
        suma_dist=[np.sum(self.X[self.labels == k], axis=0) for k in range(self.K)]
        
        self.centroids= np.divide(suma_dist,punts_per_clase)
    
    
    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        return np.allclose(self.old_centroids, self.centroids, rtol=self.options['tolerance'])
        

    
    
    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number
        of iterations is smaller than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        self._init_centroids()
        self.num_iter = 0
        
        while (not self.converges() and self.num_iter < self.options['max_iter']): #comprovem si convergeix
            self.get_labels() #trobem quin es el centroide mes proper
            self.get_centroids() #calculem els nous centres
            self.num_iter = self.num_iter + 1

    
    
    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """
    
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################

        return np.multiply(np.divide(1,self.X.shape[0]), np.sum(np.square(np.min(distance(self.X, self.centroids), axis=1))))
    
    
    def find_bestK(self, max_K):
        """
         sets the best k anlysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
       
        trobat=False
        
        self.K=2
        self.fit()
        anterior=self.withinClassDistance()
        for K in range(3,max_K+1):
            self.K=K
            self.fit()
            actual=self.withinClassDistance()
            
            if (100-100*actual/anterior) < 20:
                trobat=True
                self.K=K-1
                break
                
            anterior=actual
        
        #if trobat==False:
           # self.k=max_K
            



def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    dist=np.zeros((X.shape[0], C.shape[0]))
    for i in range(C.shape[0]): #vamos por columnas
        dist[:, i]=np.linalg.norm((X-C[i]), axis=1)
      
    return dist
   


"""
    dist1=np.sum(np.abs(X-C[0]), axis=1)
    dist2=np.sum(np.abs(X-C[1]), axis=1)  
    dist3=np.sum(np.abs(X-C[2]), axis=1)
    dist4=np.sum(np.abs(X-C[3]), axis=1)
  
    return np.dstack((dist1, dist2, dist3, dist4)).reshape(-1, C.shape[0])
#dstack une los vectores en vertical
#el reshape es porque si no devolveria dimensiones (1,4800,3) y queremos (4800,3)
"""

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    
    matriu=utils.get_color_prob(centroids)
    indices=np.argmax(matriu, axis=1)
    lista = [utils.colors[i] for i in indices]
    return lista

