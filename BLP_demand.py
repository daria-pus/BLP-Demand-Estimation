import numpy as np
from scipy.optimize import minimize
import pandas as pd
import time
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
        X1 :     The variables that vary over products and markets and have random coefficient 
                 (TJ by k1) dimension
        X2 :     The variables that only vary over products and do not change from market 
                 to market and have a random coefficient (TJ by k2) dimension
        X3 :     The variables that vary over products and markets and do not have a 
                 random coefficient (TJ by k3) dimension

        prod_d : Product/brand dummies (TJ by J) dimension
        Z  :     Instruments with (TJ by K) dimension 
        s_jt:    Observed market shares (TJ by 1) dimension 
        cdindex: Index of the last observation for a market (T by 1)
        cdid:    Vector of market indexes (TJ by 1)
        
        X1_names, X2_names, X3_names, D_names: names of variables from X1, X2, X3 
                 and demographic variables respectively
        
        theta2w: Starting values
        
        x1 = [X1, X2]: variables enter the linear part of the estimation
        x2 = [X1, X3]: variables enter the non-linear part
        TJ is number of observations (TJ = T*J if all products are observed in all markets)
        
        """
            
    def __init__(self,data):      
        # x1 variables enter the linear part of the estimation: X1, X3
        self.x1 = data.x1
        
        # x2 variables enter the non-linear part: X1, X2
        self.x2 = data.x2
        
        # the market share of brand j in market t
        self.s_jt = data.s_jt

        # Random draws. For each market ns*K2 iid normal draws are provided.
        # They correspond to ns "individuals", where for each individual
        # there is a different draw for each column of x2. 
        self.v = data.v

        self.cdindex = data.cdindex
        self.cdid = data.cdid

        self.vfull = self.v[self.cdid,:]
        self.dfull = data.demogr[self.cdid,:]
        
        # instrumental variables including exogenous variables from x1
        self.IV = data.IV
        
        # arrays with names of variables 
        self.X1_names = data.X1_names 
        self.X2_names = data.X2_names  
        self.X3_names = data.X3_names                  
        self.D_names = data.D_names                       
        
        self.K1 = self.x1.shape[1]
        self.K2 = self.x2.shape[1]
        self.ns = int(self.v.shape[1]/self.K2)          # number of simulated "indviduals" per market 
        self.D = int(self.dfull.shape[1]/self.ns)       # number of demographic variables
        self.T = self.cdindex.shape[0]                  # number of markets = (# of cities)*(# of quarters)  
        self.TJ = self.x1.shape[0]
        self.J = int(self.TJ/self.T)
                


        ## create weight matrix
        self.invA = np.linalg.inv(self.IV.T @ self.IV)
        
        ## compute the outside good market share by market
        temp = np.cumsum(self.s_jt)
        sum1 = temp[self.cdindex]
        sum1[1:] = np.diff(sum1.T)
        outshr = (1 - sum1[self.cdid]).reshape(self.TJ,1)
        
        ## compute logit results and save the mean utility as initial values for the search below
        y = np.log(self.s_jt) - np.log(outshr)
        mid = self.x1.T @ self.IV @ self.invA @ self.IV.T
        t = np.linalg.inv(mid @ self.x1) @ mid @ y
        self.d_old = self.x1 @ t
        self.d_old = np.exp(self.d_old)

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
        return self.theta2

    def gmmobj(self, theta2):
        # compute GMM objective function
        delta = self.meanval(theta2)
        self.theta2=theta2
        ##% the following deals with cases where the min algorithm drifts into region where the objective is not defined
        if max(np.isnan(delta)) == 1:
            f = 1e+10
        else:
            temp1 = self.x1.T @ self.IV
            temp2 = delta.T @ self.IV
            self.theta1 = np.linalg.inv(temp1 @ self.invA @ temp1.T) @ temp1 @ self.invA @ temp2.T
            self.gmmresid = delta - self.x1 @ self.theta1
            temp1 = self.gmmresid.T @ self.IV
            f = temp1 @ self.invA @ temp1.T
        self.gmmvalnew = f[0,0]
        self.gmmdiff = np.abs(self.gmmvalold - self.gmmvalnew)
        self.gmmvalold = self.gmmvalnew
        self.iter += 1
        if self.iter % 10 == 0:
            print('gmm objective:', f[0,0])
        return(f[0, 0])

    def meanval(self, theta2):
        if self.gmmdiff < 1e-6:
            etol = self.etol = 1e-11
        elif self.gmmdiff < 1e-3:
            etol = self.etol = 1e-9
        else:
            etol = self.etol = 1e-7
        norm = 1
        i = 0
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
                theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind]
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
        eg = np.multiply(np.exp(self.mufunc(theta2w)), np.kron(np.ones((1, self.ns)), self.d_old))
        temp = np.cumsum(eg, 0)
        sum1 = temp[self.cdindex, :]
        sum2 = sum1
        sum2[1:sum2.shape[0], :] = np.diff(sum1.T).T
        denom1 = 1. / (1. + sum2)
        denom = denom1[self.cdid, :]
        return np.multiply(eg, denom)

    def jacob(self, theta2): # Jacobian needs to be adjusted if supply is added
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = self.theta2[ind]
        shares = self.ind_sh(theta2w)
        f1 = np.zeros((self.cdid.shape[0] , self.K2 * (self.D + 1)))
        # calculate derivative of shares with respect to the first column of theta2w (variable that are not interacted with demogr var, sigmas)
        for i in range(self.K2):
            xv = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1, self.ns)),
                             self.v[self.cdid, self.ns*i:self.ns * (i + 1)])
            temp = np.cumsum(np.multiply(xv, shares), 0)
            sum1 = temp[self.cdindex, :]
            sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
            f1[:,i] = (np.mean((np.multiply(shares, xv - sum1[self.cdid,:])),1)).flatten()
        # If no demogr comment out the next part
        # calculate derivative of shares with respect to all but first column of theta2w (where x2 is interacted with demographic variables)
        for j in range(self.D):
            d = self.dfull[:, self.ns * j:(self.ns * (j+1))]   
            temp1 = np.zeros((self.cdid.shape[0], self.K2))
            for i in range(self.K2):
                xd = np.multiply(self.x2[:, i].reshape(self.TJ, 1) @ np.ones((1, self.ns)), d)
                temp = np.cumsum(np.multiply(xd, shares), 0)
                sum1 = temp[self.cdindex, :]
                sum1[1:sum1.shape[0], :] = np.diff(sum1.T).T
                temp1[:, i] = (np.mean((np.multiply(shares, xd-sum1[self.cdid, :])), 1)).flatten()
            f1[:, (self.K2 * (j + 1)):(self.K2 * (j + 2))] = temp1

        self.rel = self.theti + self.thetj * (max(self.theti)+1)
        f = np.zeros((self.cdid.shape[0],self.rel.shape[0]))
        n = 0

        for i in range(self.T):
            temp = shares[n:(self.cdindex[i] + 1), :]
            H1 = temp @ temp.T
            H = (np.diag(np.array(sum(temp.T)).flatten())-H1) / self.ns
            f[n:(self.cdindex[i]+1), :] = np.linalg.inv(H) @ f1[n:(self.cdindex[i] + 1), self.rel]
            n = self.cdindex[i] + 1
        return f

    def varcov(self, theta2):
        Z = self.IV.shape[1]
        temp = self.jacob(theta2)                                              
        a = np.concatenate((self.x1.todense(), temp), 1).T @ self.IV
        IVres = np.multiply(self.IV, self.gmmresid @ np.ones((1, Z)))
        b = IVres.T @ IVres
        inv_aAa = np.linalg.inv(a @ self.invA @ a.T)
        f = inv_aAa @ a @ self.invA @ b @ self.invA @ a.T @ inv_aAa       
        return f

    def iterate_optimization(self, opt_func, param_vec, args=(), method='Nelder-Mead'):
        success = False
        while not success:
            res = minimize(opt_func, param_vec, args=args, method=method)
            param_vec = res.x
            if res.success:
                return res.x

    def results(self, theta2):
        var = self.varcov(theta2)
        se_all = np.sqrt(var.diagonal())
        
        # the Minimum Distance estimates
        # If brand dummies were included, they absorb any product 
        # characteristics that are constant across markets.
        if ((self.K1-len(self.X1_names+self.X3_names))>0) & (len(self.X2_names)>0):
            omega = np.linalg.inv(var[self.K1-self.J:self.K1,self.K1-self.J:self.K1])
            xmd = self.x2[0:self.J,[0]+list(range(self.K2-len(self.X2_names),self.K2))]
            ymd = self.theta1[self.K1-self.J:self.K1]
        
            beta = np.linalg.inv(xmd.T @ omega @ xmd) @ xmd.T @ omega @ ymd
            resmd = ymd - xmd @ beta;
            semd = np.sqrt((np.linalg.inv(xmd.T @ omega @ xmd)).diagonal())
            
            Rsq = 1-((resmd-np.mean(resmd)).T @ (resmd-np.mean(resmd))) @ np.linalg.inv((ymd-np.mean(ymd)).T @ (ymd-np.mean(ymd)))
            Rsq_G = 1-(resmd.T @ omega @ resmd) @ np.linalg.inv((ymd-np.mean(ymd)).T @ omega @ (ymd-np.mean(ymd)))
        else:
            beta = []
            semd = []
            Rsq = []
            Rsq_G = []
        
        #mcoef = np.concatenate((beta[0,0], self.theta1[0,0:self.X1.shape[1]], beta[0,1:], self.theta1[0,self.X1.shape[1]:self.K1-self.J]),0)
        #semcoef = np.concatenate((semd[0,0], se_all[0,0:self.X1.shape[1]], semd[0,1:], se_all[0,self.X1.shape[1]:self.K1-self.J]),0)
        
        mcoef = np.concatenate((beta[0,:], self.theta1[0,0:len(self.X1_names)], beta[1:,:]),0)
        semcoef = np.concatenate((semd[:,0], se_all[0,0:len(self.X1_names)], semd[:,1:].T),0)
        
        index = []
        names = ['Constant'] + self.X1_names + self.X2_names + self.X3_names
        for var in names:
            index.append(var)
            index.append('')

        table_results = pd.DataFrame(data=np.zeros((self.K2 * 2, 2 + self.D)), columns=['Mean', 'SD'] + self.D_names)
        se_theta2 = se_all[:,self.K1:]
        se_theta2w = np.zeros((1,self.K2*(1 + self.D)))
        for i in range(se_theta2.shape[1]):
            se_theta2w[0,self.rel[i]] = se_theta2[0,i]
        se_theta2w = se_theta2w.reshape((self.K2,1+self.D), order='F')
        for i in range(len(names)):
            table_results.loc[2*i+1, 'Mean'] = semcoef[i,0]
            table_results.loc[2*i+1, ['SD'] + self.D_names] = se_theta2w[i,:]

        table_results.index=index
        for i in range(len(names)):
            table_results.loc[names[i],'Mean']= mcoef[i]
        theta2w = np.zeros((self.K2, self.D+1))
        for ind in range(len(self.theti)):
            theta2w[self.theti[ind], self.thetj[ind]] = theta2[ind]
        table_results.loc[names,['SD'] + self.D_names]=theta2w        

        table_results = table_results.reset_index()

        
        
        
        print(table_results)
        print('GMM objective: {}'.format(self.gmmvalnew))   
        print('Min-Dist R-squared: {}'.format(Rsq[0,0]))
        print('Min-Dist weighted R-squared: {}'.format(Rsq_G[0,0]))   
        print('run time: {} (minutes)'.format((time.time() - starttime) / 60))   

#%%
if __name__ == '__main__':
    starttime = time.time()
    data = Data()
    blp = BLP(data)
    init_theta = blp.init_theta(data.theta2w)
    res = blp.iterate_optimization(opt_func=blp.gmmobj,
                                   param_vec=init_theta,
                                   args=())
    blp.results(res)
