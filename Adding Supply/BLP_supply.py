import numpy as np
from scipy.optimize import minimize
import pandas as pd
import time
import scipy.linalg
#%%      
class BLP():
    """BLP Class
    Random coefficient logit model
    Parameters
    ----------
    data : object
        Object containing data for estimation. It should contain:
        v  :     Random draws given for the estimation with (T by ns*(k1+k2)) dimension
        demogr:  Demeaned draws of demographic variables with (T by ns*D) dimension
        x1 :     variables enter the linear part of the estimation
        x2 :     variables enter the non-linear part (includes price)
        p :      price vector
        W :      cost characterisctics
        IVd:     instrumental variables including exogenous variables from x1
        IVs:     instrumental variables including exogenous variables from W
        s_jt:    Observed market shares (TJ by 1) dimension 
        cdindex: Index of the last observation for a market (T by 1)
        cdid:    Vector of market indexes (TJ by 1)
        own:     ownership matrix
        
        p_index2: location of price in x2 
        
        theta2w: Starting values
        
        TJ is number of observations (TJ = T*J if all products are observed in all markets)
        
        """
            
    def __init__(self,data):      
        # x1 variables enter the linear part of the estimation
        self.x1 = data.x1
        
        # x2 variables enter the non-linear part (includes price)
        self.x2 = data.x2
        
        self.p = data.p
        
        # the market share of brand j in market t
        self.s_jt = data.s_jt
        
        # cost characteristics
        self.W = data.W
        
        # Random draws. For each market ns*K2 iid normal draws are provided.
        # They correspond to ns "individuals", where for each individual
        # there is a different draw for each column of x2. 
        self.v = data.v

        self.cdindex = data.cdindex
        self.cdid = data.cdid

        self.vfull = self.v[self.cdid,:]
        self.demogr = data.demogr
        self.dfull = self.demogr[self.cdid,:]
        # ownership matrix
        self.own = data.own
        
        # instrumental variables including exogenous variables from x1, W
        self.IVd = data.IVd
        self.IVc = data.IVc
        
        # location of price in x2
        self.p_index2 = data.p_index2                    
        
        self.K1 = self.x1.shape[1]
        self.K2 = self.x2.shape[1]
        self.ns = int(self.v.shape[1]/self.K2)          # number of simulated "indviduals" per market 
        self.D = int(self.dfull.shape[1]/self.ns)       # number of demographic variables
        self.T = self.cdindex.shape[0]                  # number of markets = (# of cities)*(# of quarters)  
        self.TJ = self.x1.shape[0]
        self.J = int(self.TJ/self.T)
                


        ## create weight matrix
        self.invAd = np.linalg.inv(self.IVd.T @ self.IVd)
        self.invAc = np.linalg.inv(self.IVc.T @ self.IVc)
        
        self.IV = scipy.linalg.block_diag(self.IVd, self.IVc)
        self.invAcomb = np.linalg.inv(self.IV.T @ self.IV)
        
        ## compute the outside good market share by market
        temp = np.cumsum(self.s_jt)
        sum1 = temp[self.cdindex]
        sum1[1:] = np.diff(sum1.T)
        outshr = (1 - sum1[self.cdid]).reshape(self.TJ,1)
        
        ## compute logit results and save the mean utility as initial values for the search below
        y = np.log(self.s_jt) - np.log(outshr)
        X = np.concatenate((self.p.todense(), self.x1.todense()),1)
        mid = X.T @ self.IVd @ self.invAd @ self.IVd.T
        t = np.linalg.inv(mid @ X) @ mid @ y
        self.d_old = X @ t
        self.d_old = np.exp(self.d_old)
        
        self.alpha = t[0]
        
        self.gmmvalold = 0
        self.gmmdiff = 1

        self.iter = 0

    def init_theta(self, theta2w):
        ## starting values for theta2. zero elements in the matrix 
        ## correspond to coeff that will not be max over,i.e are fixed at zero.
        ## The rows are coefficients for X1 varibles
        ## The columns represent SD, and interactions with demographic variables.
        theta2w = theta2w.reshape((self.K2, self.D+1))
        ##% create a vector of the non-zero elements in the above matrix, and the %
        ##% corresponding row and column indices. this facilitates passing values % 
        ##% to the functions below. %
        self.theti, self.thetj = list(np.where(theta2w != 0))
        self.theta2 = theta2w[np.where(theta2w != 0)]
        self.theta2_alpha = np.append(self.theta2, self.alpha)
        return self.theta2_alpha

    def gmmobj(self, theta2_alpha):
        alpha = theta2_alpha[-1]
        theta2 = theta2_alpha[:-1]
        # compute GMM objective function
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
                theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind]
        delta = self.meanval(theta2w)
        self.theta2=theta2
        self.alpha = alpha
        ##% the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(delta)) == 1:
            f = 1e+10
        else:
            # calculate theta1
            y_d = delta - alpha*self.p
            temp1 = self.x1.T @ self.IVd
            temp2 = y_d.T @ self.IVd
            self.theta1 = np.linalg.inv(temp1 @ self.invAd @ temp1.T) @ temp1 @ self.invAd @ temp2.T
            gmmresid_d = y_d - self.x1 @ self.theta1
            # calculate markup
            m = self.markup(alpha, theta2w)
            # estimate cost parameters
            mc = self.p - m
            temp1 = self.W.T @ self.IVc
            temp2 = mc.T @ self.IVc
            self.w = np.linalg.inv(temp1 @ self.invAc @ temp1.T) @ temp1 @ self.invAc @ temp2.T
            gmmresid_s = mc - self.W @ self.w
            # calculate ojective function
            self.gmmresid = np.concatenate((gmmresid_d, gmmresid_s),0)
            temp1 = self.gmmresid.T @ self.IV
            f = temp1 @ self.invAcomb @ temp1.T
        self.gmmvalnew = f[0,0]
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        self.iter += 1
        if self.iter % 10 == 0:
            print('gmm objective:', f[0,0])
        return(f[0, 0])

    def meanval(self, theta2w):
        # computes delta using contraction mapping
        # starting value for delta is computed delta from last gmm iteration
        if self.gmmdiff < 1e-6:
            etol = self.etol = 1e-11
        elif self.gmmdiff < 1e-3:
            etol = self.etol = 1e-9
        else:
            etol = self.etol = 1e-7
        norm = 1
        i = 0
        while norm > etol:
            pred_s_jt = self.mktsh(theta2w)
            self.d_new = np.multiply(self.d_old,self.s_jt) / pred_s_jt # (13) in Nevo 
            t = np.abs(self.d_new - self.d_old)
            norm = np.max(t)
            self.d_old = self.d_new
            i += 1
#        print('# of iterations for delta convergence:', i)
        return np.log(self.d_new)

    def mufunc(self, theta2w):
        # computes deviation from mean utility
        mu = np.zeros((self.TJ, self.ns))
        for i in range(self.ns):
            v_i = self.vfull[:, np.arange(i, self.K2 * self.ns, self.ns)]
            d_i = self.dfull[:, np.arange(i, self.D * self.ns, self.ns)]
            temp = d_i @ theta2w[:, 1:(self.D+1)].T
            mu[:, i]=(np.multiply(self.x2, v_i) @ theta2w[:, 0]) + np.multiply(self.x2, temp) @ np.ones((self.K2))
        return mu

    def mktsh(self, theta2w):
        # compute the market share for each product
        temp = self.ind_sh(theta2w).T
        f = (sum(temp) / float(self.ns)).T
        f = f.reshape(self.TJ,1)
        return f
    
    def ind_sh(self, theta2w): # need to incorporate simulation variance reduction
        # computes individual market shares
        eg = np.multiply(np.exp(self.mufunc(theta2w)), np.kron(np.ones((1, self.ns)), self.d_old))
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[self.cdid, :]
        return np.multiply(eg, denom)

    def markup(self, alpha, theta2w):
        # computes price-cost markup for bertrand compedtition
        # uses calculated shares everywhere
        m = np.zeros((self.TJ,1))
        shares = self.ind_sh(theta2w)
        s = np.sum(shares,axis=1)
        t_d = np.ones((self.T,self.ns))
        for d in range(self.D):
            t_d = t_d + theta2w[self.p_index2,1+d]*self.demogr[:,self.ns*d:self.ns*(d+1)]
        alpha_i = alpha + theta2w[self.p_index2,0]*self.v[:,self.ns*self.p_index2: self.ns*(self.p_index2+1)] + t_d  
        n = 0
        for i in range(self.T):
            temp = shares[n:(self.cdindex[i] + 1), :]
            temp1 = np.multiply(temp, np.kron(np.ones((self.J,1)), alpha_i[i,:]))
            H1 = temp1 @ temp.T
            H = (np.diag(np.array(sum(temp1.T)).flatten())-H1) / self.ns
            f = np.multiply(H, self.own)
            f = np.linalg.inv(f)*s[n:(self.cdindex[i] + 1), :]
            m[n:(self.cdindex[i]+1), :] = f
            n = self.cdindex[i] + 1
        m = - m
        return m

    def iterate_optimization(self, opt_func, param_vec, args=(), method='Nelder-Mead'):
        success = False
        while not success:
            res = minimize(opt_func, param_vec, args=args, method=method)
            param_vec = res.x
            if res.success:
                return res.x


#%%
if __name__ == '__main__':
    starttime = time.time()
    data = Data()
    blp = BLP(data)
    init_theta = blp.init_theta(data.theta2w)
    res = blp.iterate_optimization(opt_func=blp.gmmobj,
                                   param_vec=init_theta,
                                   args=())
    

    
    
    
 
    

    
    
    
      
    
    
    
