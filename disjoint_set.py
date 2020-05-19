import numpy as np
from collections import defaultdict

class disjoint_set:
    '''
    disjoint_set object implementation
    '''

    def __init__(self, num_of_nodes):
        self._num = num_of_nodes             #total number of nodes in img(rows*cols)
        self._arr = np.empty((self._num, 3), dtype=np.int64)
        for i in range(self._num):
            self._arr[i][0] = i              #parents
            self._arr[i][1] = 0              #rank
            self._arr[i][2] = 1              #size
        self._INT = np.empty((self._num, ), dtype=np.float64) #internal difference
        
    @property
    def num(self):
        return self._num

    @property
    def arr(self):
        return self._arr
    
    @property
    def INT(self):
        return self._INT
    
    def update_INT(self, xp, yp, w):
    #update the internel difference 
        
        new_INT = max(w, self._INT[xp], self._INT[yp])
        
        self._INT[yp] = new_INT
        self._INT[xp] = new_INT
    
    def find(self, x):
    #find the parents of x and return it
    #compress path
    
        if self._arr[x][0] == x:
            return x
        else:
            self._arr[x][0] = self.find(self._arr[x][0])
            
            return self._arr[x][0]
    
    def union(self, x, y):
    #union two set by rank
      
        xp = self.find(x)
        yp = self.find(y)
        
        if self._arr[xp][1] < self._arr[yp][1]:
        #rand of xp less than yp
        #attach xp to yp
            self._arr[xp][0] = yp
            self._arr[yp][2] += self._arr[xp][2]
    
        elif self._arr[xp][1] > self._arr[yp][1]:
        #rank of xp greater than yp
        #attach yp to xp
            self._arr[yp][0] = xp
            self._arr[xp][2] += self._arr[yp][2]
    
        else:
        #same rank, attach xp to yp
        #increase rank
    
            self._arr[xp][0] = yp
            self._arr[yp][1] += 1
            self._arr[yp][2] += self._arr[xp][2]
    
    def is_same_parent(self, x, y):
    #if x and y in same component
        
        return self.find(x) == self.find(y)
    
    def conclusion(self, show=False):
    #for debug
    
        con = defaultdict(list)
        

        for i in range(self._num):
             con[self.find(self._arr[i][0])].append(i)

        if show:
            for key in con:
                print(f'Set ID: {key}: number of members: {len(con[key])}')
                
        return con