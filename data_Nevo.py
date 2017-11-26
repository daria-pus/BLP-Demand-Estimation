import scipy.io as io
import numpy as np


#%%            
class Data(object):
    ''' Synthetic data for Nevo (2000b)
    The file iv.mat contains the variable iv which consists of an id column
    (see the id variable above) and 20 columns of IV's for the price
    variable. The variable is sorted in the same order as the variables in
    ps2.mat.
    '''
    def __init__(self):
                ### Variable loadings and definitions. Much of the text is borrowed from
        ### http://emlab.berkeley.edu/users/bhhall/e220c/readme.html
        ps2 = io.loadmat('ps2.mat')
        iv = io.loadmat('iv.mat')
        
        self.X1_names = ['Price']
        self.X2_names = ['Sugar', 'Mushy']
        self.X3_names = []
        self.D_names = ['Income', 'Income^2', 'Age', 'Child']
        
        # x1 variables enter the linear part of the estimation and include the 
        # price variable and 24 brand dummies. 
        self.x1 = ps2['x1']
        
        # x2 variables enter the non-linear part and include a constant, price,
        # sugar content, a mushiness dummy. 
        self.x2 = ps2['x2']
        
        # the market share of brand j in market t
        self.s_jt = ps2['s_jt']

        # Random draws. For each market 80 iid normal draws are provided.
        # They correspond to 20 "individuals", where for each individual
        # there is a different draw for each column of x2. 
        self.v = ps2['v']

        T = self.v.shape[0]                        # number of markets = (# of cities)*(# of quarters)  
        J = int(self.x2.shape[0]/T)           # number of brands per market. if the numebr differs by market this requires some "accounting" vector
        
        # this vector relates each observation to the market it is in
        self.cdid = np.kron(np.array([i for i in range(T)], ndmin=2).T, np.ones((J, 1)))
        self.cdid = self.cdid.reshape(self.cdid.shape[0]).astype('int')

        ## this vector provides for each index market the last observation index in data matrix
        ## here all brands appear in all markets. if this 
        ## is not the case the two vectors, cdid and cdindex, have to be   
        ## created in a different fashion but the rest of the program works fine.
        ##cdindex = [nbrn:nbrn:nbrn*nmkt]';
        self.cdindex = np.array([i for i in range((J - 1),J * T, J)])



        # draws of demographic variables from the CPS for 20 individuals in each
        # market. The first 20 columns give the income, the next 20 columns the
        # income squared, columns 41 through 60 are age and 61 through 80 are a
        # child dummy variable. 
        self.demogr = ps2['demogr']
        
        
        self.IV = np.matrix(np.concatenate((iv['iv'][:, 1:], self.x1[:, 1:].todense()), 1))
        
        self.theta2w =  np.array([0.377, 3.089, 0, 1.186, 0,
                         1.848, 16.598, -0.659, 0, 11.625,
                         0.004, -0.193, 0, 0.029, 0,
                         0.081, 1.468, 0, -1.514, 0], ndmin=2)