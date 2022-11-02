import itertools
import numpy as np
import pywt
import time

from numpy.linalg import norm
from scipy.linalg import expm
from scipy.optimize import minimize
from statsmodels.tsa.api import VAR


#Single-Scale Causal Structure Learning
#paper
class SSCASTLE:
    
    def __init__(self, Y, lag=None, maxlags=None, criterion=None):
        """
        INPUT
        =====
        Y: np.array, TxN with T observations and N # of ts
        lag: AR lags, if None then estimate the max number of lags via VAR model
        maxlags: int, necessary only if lag=None. Maximum number of lags to check in order to 
                estimate lag
        criterion: str, one among ['aic', 'bic', 'fpe', 'hqic']
        """
        
        #check variables

        if lag==None:

            if maxlags==None: self.maxlags = 10
            elif isinstance(maxlags, int) and maxlags>0:
                self.maxlags = maxlags
            else:
              raise Exception('maxlags must be a strictly positive integer.')
              
            if criterion==None: self.criterion = 'bic'
            elif isinstance(criterion, str) and criterion.lower() not in ['aic', 'bic', 'fpe', 'hqic']:
                self.criterion = criterion
            else:
                raise Exception("criterion must be one among ['aic', 'bic', 'fpe', 'hqic']")
            
            best_value = np.inf
            nlags = 0
            result = None

            for l in range(1, self.maxlags + 1):
                model = VAR(Y)
                fitted = model.fit(maxlags=l , ic=self.criterion, trend='nc')

                value = getattr(fitted, self.criterion)
                if value < best_value:
                    best_value = value
                    nlags = fitted.k_ar

            self.lag = nlags

        elif isinstance(lag, int) and lag>=0:
            self.lag = lag
        else: raise Exception('lag must be a positive integer.')
            
        if isinstance(Y, np.ndarray):
            self.Y = Y
            self.T = Y.shape[0] - self.lag #cut the ts for kernel computation
            self.N = Y.shape[1]
        else: raise Exception('Y must be a numpy array.')
            
        for l in range(self.lag+1):
            if l==0:
                self.Y_bar=self.Y[self.lag:]
            else:
                self.Y_bar=np.concatenate([self.Y_bar,np.roll(self.Y,l,0)[self.lag:]],1)
      
    def _loss(self, W_bar):
        
        """ Evaluate value and gradient of loss.
        INPUT
        =====
        W_bar: np.ndarray, shape (N(L+1),N). Matrix of causal coefficients
        """

        #prediction 
        M = self.Y_bar @ W_bar

        #resid
        R = self.Y[self.lag:] - M

        loss = 0.5 / self.T * (R ** 2).sum()
        G_loss = - 1.0 / self.T * self.Y_bar.T @ R

        return loss, G_loss

    def _h(self, W_bar):
        """Evaluate value and gradient of acyclicity constraint on the first NxN block of W_bar"""
        
        if W_bar.ndim==1:
            W_bar = self._adj(W_bar)
            
        E = expm(W_bar[:self.N] * W_bar[:self.N])  # (Zheng et al. 2018)
        h = np.trace(E) - self.N
        G_h = E.T * W_bar[:self.N] * 2
        return h, np.concatenate((G_h,np.zeros((self.lag*self.N,self.N))))
    
    def _adj(self, w_bar):
        """Convert vector variable ([self.N*(self.lag+1)*self.N] array) back to original matrix ([self.N*(self.lag+1), self.N])."""
        return w_bar.reshape([self.N*(self.lag+1), self.N])
    
    def _func(self, w_bar, z, reg, rho2, alpha, beta, lmbd):
        """Evaluate value and gradient for the update of W_bar"""
        W_bar = self._adj(w_bar)
        loss, G_loss = self._loss(W_bar)
        _, G_h = self._h(W_bar)
        
        if reg==None:
            obj = loss + alpha*np.trace(G_h.T@W_bar)
            G = G_loss + alpha*G_h
            g_obj = np.concatenate(G, axis=0)
        else:
            v = w_bar - z + beta
            n = norm(v, ord=2)
            obj = loss + alpha*np.trace(G_h.T@W_bar) + .5 * rho2 * n * n
            G = G_loss + alpha*G_h + rho2 * self._adj(v)
            g_obj = np.concatenate(G, axis=0)
        
        return obj, g_obj
        
    
    def _soft_thresholding(self, v, lmbd1):
        """
        Function to apply soft-thresholding operator
        """
        
        return np.where(v>lmbd1, v-lmbd1, np.where(v<-lmbd1, v+lmbd1, 0.))
        

    def solver(self, reg=None, thresh=None, interval=(-1.,1.), lmbd=None, 
               rho1=None, rho2=None, 
               alpha=None, beta=None,
               ratio=None, h_tol=None, rho1_tol=None, 
               max_iter=None, verbose=False):
        """
        INPUT
        =====
        reg: string, default None, else choose one between ['l1','tv'], where 'tv' stands for 'total variation'.
        thresh: float, threshold for causal coeff, default=.05
        interval: tuple, (min, max) for causal coefficients, default=(-1.,1.) 
        lmbd: float, sparsity regularization strenght, default=.01
        rho1: float, penalty param of augmented lagrangian linked to acyclicity constraint, default=.01
        rho2: float, penalty param of augmented lagrangian linked to sparsity constraint, default=1.
        alpha: float, initial value of the Lagrange Multiplier for h(W), default=0.
        beta: np.ndarray, shape (N*(lag+1)*N), initial value of the Lagrange Multiplier 
              for sparsity constraint, default=np.zeros(N*(lag+1)*N)
        ratio: float, condition to increase the magnitude of rho1 by a factor equal to 10,
               it must be in (0.,1.), default=.2
        h_tol: float, maximum value to be a DAG, default=1e-12
        rho1_tol: int/float, maximum value of rho1, default = 1.e8
        max_iter: int, maximum number of iteration, default=100
        verbose: bool, whether to print opt info, default False
        
        OUTPUT
        ======
        Bs = np.ndarray, shape (lag+1, N, N), causal coefficients
        (it, t) = tuple, (#iterations, #time to execute)
        """
        
        #check variables
        if reg==None: pass
        else: 
            if not isinstance(reg, str) or (reg not in ['l1','tv']): 
                raise Exception("reg must be equal to either one between ['l1','tv'] or None.")
                
        if thresh==None: thresh=.05  
        else:
            if not isinstance(thresh, float) or thresh<0.:
                raise Exception("thresh must be a scalar >= than zero.") 

        assert isinstance(interval, tuple) and len(interval)==2 and all(isinstance(n,(int, float)) for n in interval), 'rng_diag must be a tuple of two real numbers.'
        
        if lmbd==None: lmbd=.01
        else:
            if not isinstance(lmbd, float) or lmbd<=0.:
                raise Exception("lmbd must be a scalar greater than zero.") 
        
        if rho1==None: rho1=.01
        else:
            if not isinstance(rho1, float) or rho1<=0.:
                raise Exception("rho1 must be a scalar greater than zero.") 
                
        if rho2==None: rho2=1.
        else:
            if not isinstance(rho2, float) or rho2<=0.:
                raise Exception("rho2 must be a scalar greater than zero.") 
                
        if alpha==None: alpha=0.
        else:
            if not isinstance(alpha, float) or alpha<0.:
                raise Exception("alpha must be a scalar >= than zero.") 
                
        if beta==None: beta=np.zeros(self.N*(self.lag+1)*self.N)
        else:
            if not isinstance(beta, np.ndarray) or np.sum(beta<=0.)>0 or beta.shape[0]!=self.N*(self.lag+1)*self.N:
                raise Exception("beta must be a np.ndarray of shape (N*(lag+1)*N) and with entries >= than zero.") 
        
        if ratio==None: ratio=.2
        else:
            if not isinstance(ratio, float) or (ratio<=0 or ratio>=1):
                raise Exception("ratio must be a scalar between 0 and 1")
        
        if h_tol==None: h_tol=1.e-8
        else:
            if not isinstance(h_tol, float) or h_tol<=0.:
                raise Exception("h_tol must be a scalar greater than zero.") 
        
        if rho1_tol==None: rho1_tol=np.inf
        else:
            if not isinstance(rho1_tol, (int, float)) or rho1_tol<=0.:
                raise Exception("rho1_tol must be a scalar greater than zero.") 
        
        if max_iter==None: max_iter=100
        else:
            if not isinstance(max_iter, int) or max_iter<=0.:
                raise Exception("max_iter must be a scalar greater than zero.") 
                
        if not isinstance(verbose, bool):
            raise Exception("verbose must be either True or False.")
        
        start = time.time()  

        interval=sorted(interval)      
        
        #Initialization
        #vector to optimize
        w_bar = np.zeros(self.N*(self.lag+1)*self.N) #vectorization of W_bar
        #ADMM variable
        z = np.zeros(self.N*(self.lag+1)*self.N) 
        #Dagness constraint
        h = np.inf
        #Scaled Lagrange multiplier
        beta /= rho2
        
        #build kernel for total variation reg
        if reg=='tv':
            #kernel
            Y_tilde = np.zeros([self.N*(self.lag+1), self.N])
            for (i,j) in itertools.product(range(self.N), range(self.N*(self.lag+1))):
                item = norm((np.abs(self.Y_bar[:,i])-np.abs(self.Y_bar[:,j])),2)
                Y_tilde[j,i]=item*item
            #vectorize the kernel
            y_tilde = Y_tilde.flatten()
            #build variable for soft-thresholding operator
            lmbd_bar = (lmbd/(2.*self.T*rho2))*y_tilde
            
            
        #boundaries for projection operator
        bnds = [(0, 0) if (i<self.N and j<self.N and i==j) else (interval[0], interval[1]) 
                for i in range(self.N*(self.lag+1)) for j in range(self.N)]
        
        for it in range(max_iter):
            
            if verbose: print('Started iteration number ', it)
            
            w_new, h_new = None, None
            
            sol = minimize(self._func, w_bar, args=(z, reg, rho2, alpha, beta, lmbd), 
                           method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, G_new = self._h(self._adj(w_new))
            
            if h_new/h>ratio: rho1*=10.
                
            w_bar, h = w_new, h_new
            alpha += rho1*h
            
            if reg=='l1':
                z = self._soft_thresholding(w_bar+beta, lmbd/rho2)
                beta += (w_bar-z)  
            
            elif reg=='tv':
                z = self._soft_thresholding(w_bar+beta, lmbd_bar)
                beta += (w_bar-z) 
            
            if h <= h_tol or rho1>=rho1_tol:
                if verbose: 
                    print('\nFinished at iteration number ', it, '. Dagness value equal to ', h,' (if DAG, h=0).')
                break
                
            if verbose: 
                print('##### OPTIMIZATION INFO #####')
                print('Current value of objective function: ', sol.fun)
                
                if it==0:
                    W_bar = self._adj(w_bar)
                    loss, _ = self._loss(W_bar)
                    ell1 = lmbd*np.sum(np.abs(w_bar))
                
                    print('Initial value of Frobenious norm: ', loss)
                    print('Initial value of reguralisation: ', ell1)
                
                print('|| W_bar-Z ||_F: ', norm(w_bar-z, 2))
                print('Ended iteration number ', it, '. Dagness value equal to ', h,' (if DAG, h=0).')
        
        W_bar = self._adj(w_bar)
        W_bar[np.abs(W_bar) <= thresh] = 0.
        
        Bs = np.zeros((self.lag+1, self.N, self.N))
        
        for l in range(self.lag+1):
            Bs[l,:,:] = (W_bar.T[:,l*self.N:(l+1)*self.N]).T
        
        t = time.time()-start
        if verbose:
            return Bs, (it, t), loss, ell1
        else:
            return Bs, (it, t)

#Multiscale Causal Structure Learning
#paper
class MSCASTLE:
    
    def __init__(self, Y, lag=None, maxlags=None, criterion=None, wavelet='db1', ndetails=None):
        """
        INPUT
        =====
        Y: np.array, shape TxN with T observations and N equal to the number of ts
        lag: AR lags, if None then estimate the max number of lags via VAR model
        maxlags: int, necessary only if lag=None. Maximum number of lags to check in order to 
                estimate lag
        criterion: str, one among ['aic', 'bic', 'fpe', 'hqic']
        wavelet: str, wavelet to use, see pywt discrete wavelet families
        ndetails: int, number of details of Stationary Wavelet Transform (SWT)
        """
        
        #check variables
            
        if isinstance(Y, np.ndarray):
            self.Y = Y
            self.N = Y.shape[1]
            
        else: raise Exception('Y must be a numpy array with length 2^(j), where j is a positive integer.')
            
        if ndetails is None:
            #max number of levels
            self.ndetails = None       
        elif isinstance(ndetails, int) and ndetails>0:
            self.ndetails = ndetails
        else: raise Exception('ndetails must be a positive integer.')
        
        if wavelet in pywt.wavelist(kind='discrete'):
            self.wavelet = wavelet
        else:
            raise Exception('wavelet must be one of the available pywt discrete wavelets')
        
        #Apply SWT. The details of the n-th level have indexes from n*self.ndetais to (n+1)*self.ndetais
        #decompose Y
        self.Y_dec= np.concatenate(np.array(pywt.swt(self.Y, 'db1', level=self.ndetails,
                                                     axis=0, trim_approx=True, norm=True))[1:], axis=-1)
        
        if self.ndetails is None:
            self.ndetails = self.Y_dec.shape[1]//self.N

        self.DN = self.N*self.ndetails
        
        #check everything is correct
        if self.DN != self.Y_dec.shape[1]:
            raise Exception('The number of columns of transformed dataset must be equal to Y.shape[1]*ndetails')
        
        #check out the lag.
        #if not given, it uses VAR model.
        #The selected lag is the one associated to the best value of bic 
        #according to VAR fit onto each time scale.
        #Thus, the lag could not be the highest: what matters is the value of the fit.
        if lag==None:

            if maxlags==None: self.maxlags = 10
            elif isinstance(maxlags, int) and maxlags>0:
                self.maxlags = maxlags
            else:
                raise Exception('maxlags must be a strictly positive integer.')
              
            if criterion==None: self.criterion = 'bic'
            elif isinstance(criterion, str) and criterion.lower() not in ['aic', 'bic', 'fpe', 'hqic']:
                self.criterion = criterion
            else:
                raise Exception("criterion must be one among ['aic', 'bic', 'fpe', 'hqic']")
            
            best_value = np.inf
            nlags = 0
            result = None
            
            for n in range(self.N):
                
                self.Y_level = np.copy(self.Y_dec[:,n*self.ndetails:(n+1)*self.ndetails])
                
                for l in range(1, self.maxlags + 1):
                    model = VAR(self.Y_level)
                    fitted = model.fit(maxlags=l , ic=self.criterion, trend='nc')

                    value = getattr(fitted, self.criterion)
                    #according to this condition
                    #nlags can be lowered as well
                    if value < best_value:
                        best_value = value
                        nlags = fitted.k_ar

            self.lag = nlags

        elif isinstance(lag, int) and lag>=0:
            self.lag = lag
        else: raise Exception('lag must be a positive integer.')
            
        self.T = self.Y.shape[0] - self.lag
        
        for l in range(self.lag+1):
            if l==0:
                self.Y_bar=self.Y_dec[self.lag:]
            else:
                self.Y_bar=np.concatenate([self.Y_bar,np.roll(self.Y_dec,l,0)[self.lag:]],1)
        
    def _loss(self, W_bar):
        
        """ Evaluate value and gradient of loss.
        INPUT
        =====
        W_bar: np.ndarray, shape (self.DN(self.lag+1), self.DN). Matrix of causal coefficients. 
               [W0.T,W1.T,...,WL.T].T where Wl is block diagonal and contains the causal coeff
               wij^l !=0 if i^l->j.
               
        OUTPUT
        ======
        loss: float, value of the part of the loss function associated with Frobenious norm
        G_loss: np.ndarray, shape(self.DN, self.DN), gradient of the loss         
        """

        #Y_hat 
        M = self.Y_bar @ W_bar

        #resid
        R = self.Y_dec[self.lag:] - M

        loss = 0.5 / self.T * (R ** 2).sum()
        G_loss = - 1.0 / self.T * self.Y_bar.T @ R
        
        return loss, G_loss

    def _h(self, W_bar):
        """ Evaluate value and gradient of acyclicity constraint (Zheng et al. 2018).
        
        INPUT
        =====
        W_bar: np.ndarray, shape (self.DN(self.lag+1), self.DN). Matrix of causal coefficients. 
               [W0.T,W1.T,...,WL.T].T where Wl is block diagonal and contains the causal coeff
               wij^l !=0 if i^l->j.
               
        OUTPUT
        ======
        h: float, value of the acyclicity constraint
        G_h: np.ndarray, shape(self.DN, self.DN), gradient of the acyclicity constraint
        """
        
        #check dimension
        if W_bar.ndim==1:
            W_bar = self._adj(W_bar)
            
        E = expm(W_bar[:self.DN] * W_bar[:self.DN])  # (Zheng et al. 2018)
        h = np.trace(E) - self.DN
        G_h = E.T * W_bar[:self.DN] * 2
        
        return h, np.concatenate((G_h,np.zeros((self.lag*self.DN,self.DN))))
    
    def _adj_prior(self, vec):
        """
        Function to compute a block diag prior.
        It reshapes the vector of length (self.lag+1)*self.ndetails*self.N*self.N into
        the block diag matrix V of shape (self.DN(self.lag+1), self.DN)
        
        INPUT
        =====
        vec: np.ndarray, shape ((self.lag+1)*self.ndetails*self.N*self.N,)
        
        OUTPUT
        ======
        V: np.ndarray, shape (self.DN(self.lag+1), self.DN)
        """ 
        for lag in range(self.lag+1):

            V_=vec[lag*(self.ndetails*self.N*self.N):(lag+1)*(self.ndetails*self.N*self.N)].reshape(self.ndetails,self.N,self.N)

            arrs = [np.atleast_2d(a) for a in [V_[i,:,:] for i in range(self.ndetails)]]

            shapes = np.array([a.shape for a in arrs])
            out_dtype = np.find_common_type([arr.dtype for arr in arrs], [])
            out = np.zeros(np.sum(shapes, axis=0), dtype=out_dtype)

            r, c = 0, 0
            for i, (rr, cc) in enumerate(shapes):
                out[r:r + rr, c:c + cc] = arrs[i]
                r += rr
                c += cc

            if lag==0:
                V=out
            else:
                V=np.concatenate([V,out],axis=0)

        return V
        
    def _adj(self, w_bar):
        """
        Reshape the vector of length (self.lag+1)*self.DN*self.DN into
        the block diag matrix W_bar of shape (self.DN(self.lag+1), self.DN)
        
        INPUT
        =====
        w_bar: np.ndarray, shape ((self.lag+1)*self.ndetails*self.N*self.N,)
        
        OUTPUT
        ======
        W_bar: np.ndarray, shape (self.DN(self.lag+1), self.DN)
        """ 
        return w_bar.reshape(self.DN*(self.lag+1), self.DN)
    
    def _func(self, w_bar, z, reg, rho2, alpha, beta, lmbd):
        """ Evaluate value and gradient for the update of W_bar. 
        
        INPUT
        =====
        w_bar:np.ndarray, shape ((self.lag+1)*self.DN*self.DN,). 
              Vectorized form of W_bar row-wise: the first self.DN elements
              of w_bar correspond to the first row of W_bar, etc...
        z: np.ndarray, shape ((self.lag+1)*self.DN*self.DN,). Additional variable for sparsity constraints
        reg: string, default None, else 'l1'
        rho2: float, penalty param of augmented lagrangian linked to sparsity constraint, default=1.0
        alpha: float, initial value of the Lagrange Multiplier for h(W), default=0.0
        beta: np.ndarray, shape ((self.lag+1)*self.DN*self.DN,), initial value of the Lagrange Multiplier 
              for sparsity constraint, default=np.zeros((self.lag+1)*self.DN*self.DN)
        lmbd: float, sparsity regularization strenght, default=.05. Please notice that only 
            causal coefficients with magnitude >lmbd2/rho2 will survive
            
        
        OUTPUT
        ======
        obj : float, value of the Augmented Lagrangian necessary for the update of w_bar
              according to the specified regularization
        g_obj: np.ndarray, vectorized form of the gradient of obj, shape equal to (self.DN*(self.lag+1)*self.DN,)
        """
        W_bar = self._adj(w_bar)
        
        loss, G_loss = self._loss(W_bar)
        _, G_h = self._h(W_bar)
        
        if reg==None:
            obj = loss + alpha*np.trace(G_h.T@W_bar)
            G = G_loss + alpha*G_h
            g_obj = np.concatenate(G, axis=0)
        else:
            #choosing l1 or l1-l12 leads to the same objective for w_bar
            v = w_bar - z + beta
            n = norm(v, ord=2)
            obj = loss + alpha*np.trace(G_h.T@W_bar) + .5 * rho2 * n * n 
            G = G_loss + alpha*G_h + rho2 * self._adj(v)
            g_obj = np.concatenate(G, axis=0)
        
        return obj, g_obj
        
    
    def _soft_thresholding(self, v, lmbd1):
        """
        Function to apply soft-thresholding operator element-wise
        
        INPUT
        =====
        v: np.ndarray, vector/matrix to be soft-thresholded
        lmbd1: float, lmbd/rho2
        
        OUTPUT
        ======
        Soft-thresholded vector/matrix
        """
        
        return np.where(v>lmbd1, v-lmbd1, np.where(v<-lmbd1, v+lmbd1, 0.))
    
    def _prior(self, bounds):
        """
        Construct optimization constraints according to specified bounds.

        INPUT
        =====
        bounds: tuple, (min, max) of causal coefficients different from zero
        
        OUTPUT
        ======
        prior :  list of tuples, constraints for each causal coefficient
        """
        
        entries = np.ones((self.lag+1)*self.ndetails*self.N*self.N)
        mask = self._adj_prior(entries)
        np.fill_diagonal(mask, 0.)
        prior=[bounds if i!=0 else (0,0) for i in mask.flatten()]
        
        return prior
    
    def _solver(self, reg=None, 
                thresh=None, bounds=None, 
                lmbd=None, 
                rho1=None, rho2=None, 
                alpha=None, beta=None,
                ratio=None, h_tol=None, rho1_tol=None, 
                max_iter=None, verbose=False):
        """
        Function to solve the optimization problem.
        
        INPUT
        =====
        reg: string, default None, else 'l1'
        thresh: float, threshold for causal coeff, default=.0
        bounds: tuple, boundaries for causal coefficient different from zero, default=(-1,1)
        lmbd: float, sparsity regularization strenght, default=.05. Please notice that only 
            causal coefficients with magnitude >lmbd2/rho2 will survive
        rho1: float, penalty param of the Augmented Lagrangian linked to acyclicity constraint, default=.01
        rho2: float, penalty param of Augmented Lagrangian linked to sparsity constraint, default=1.0
        alpha: float, initial value of the Lagrange Multiplier for h(W), default=0.0
        beta: np.ndarray, shape ((self.lag+1)*self.DN*self.DN,), initial value of the Lagrange Multiplier 
              for sparsity constraint, default=np.zeros((self.lag+1)*self.DN*self.DN,)
        ratio: float, condition to increase the magnitude of rho1 by a factor equal to 10,
               it must be in (0.,1.), default=.2
        h_tol: float, maximum value to be a DAG, default=1e-8
        rho1_tol: int/float, maximum value of rho1, default = 1.e8
        max_iter: int, maximum number of iteration, default=100
        verbose: bool, whether to print opt info, default False
        
        OUTPUT
        ======
        C = np.ndarray, shape (self.DN, self.DN), causal coefficients
        (it, t) = tuple, (#iterations, #time to execute)
        """
        
        #check variables
        if reg==None: pass
        else: 
            if not isinstance(reg, str) or (reg!='l1'): 
                raise Exception("reg must be either None or 'l1'.")
                
        if thresh==None: thresh=.0  
        else:
            if not isinstance(thresh, float) or thresh<0.:
                raise Exception("thresh must be a scalar >= than zero.") 
                
        if bounds==None: bounds=(-1,1)
        else:
            if not isinstance(bounds, tuple):
                raise Exception("bounds must be a tuple.") 
        
        if lmbd==None: lmbd=.05
        else:
            if not isinstance(lmbd, float) or lmbd<=0.:
                raise Exception("lmbd must be a scalar greater than zero.")
        
        if rho1==None: rho1=.01
        else:
            if not isinstance(rho1, float) or rho1<0.:
                raise Exception("rho1 must be a scalar greater than zero.") 
                
        if rho2==None: rho2=1.
        else:
            if not isinstance(rho2, float) or rho2<0.:
                raise Exception("rho2 must be a scalar greater than zero.") 
                
        if alpha==None: alpha=0.
        else:
            if not isinstance(alpha, float) or alpha<0.:
                raise Exception("alpha must be a scalar >= than zero.") 
                
        if beta==None: beta=np.zeros((self.lag+1)*self.DN*self.DN)
        else:
            if not isinstance(beta, np.ndarray) or np.sum(beta<=0.)>0 or beta.shape[0]!=(self.lag+1)*self.DN*self.DN:
                raise Exception("beta must be a np.ndarray of shape ((self.lag+1)*self.DN*self.DN,) and with entries >= than zero.") 
        
        if ratio==None: ratio=.2
        else:
            if not isinstance(ratio, float) or (ratio<=0 or ratio>=1):
                raise Exception("ratio must be a scalar between 0 and 1")
        
        if h_tol==None: h_tol=1.e-8
        else:
            if not isinstance(h_tol, float) or h_tol<=0.:
                raise Exception("h_tol must be a scalar greater than zero.") 
        
        if rho1_tol==None: rho1_tol=np.inf
        else:
            if not isinstance(rho1_tol, (int, float)) or rho1_tol<=0.:
                raise Exception("rho1_tol must be a scalar greater than zero.") 
        
        if max_iter==None: max_iter=100
        else:
            if not isinstance(max_iter, int) or max_iter<=0.:
                raise Exception("max_iter must be a scalar greater than zero.") 
        
        if not isinstance(verbose, bool):
            raise Exception("verbose must be either True or False.")
        
        start = time.time()        
        
        #Initialization
        #vector to optimize
        w_bar = np.zeros((self.lag+1)*self.DN*self.DN) #vectorization of W_bar
        #ADMM variable
        z = np.zeros((self.lag+1)*self.DN*self.DN) 
        #Dagness constraint
        h = np.inf
        #Scaled Lagrange multiplier
        beta /= rho2

        bounds=sorted(bounds)            
            
        #boundaries for causal coeff
        bnds = self._prior(bounds=bounds)
        
        for it in range(max_iter):
            
            if verbose: print('\nStarted iteration number ', it)
            
            w_new, h_new = None, None
            
            sol = minimize(self._func, w_bar, args=(z, reg, rho2, alpha, beta, lmbd), 
                           method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, G_new = self._h(self._adj(w_new))
            
            if h_new/h>ratio: rho1*=10.
                
            w_bar, h = w_new, h_new
            alpha += rho1*h
            
            if reg=='l1':
                #apply sof-thresholding
                z = self._soft_thresholding(w_bar+beta, lmbd/rho2)
                #update the Lagrange multiplier
                beta += (w_bar-z)  
            else:
                pass
            
            #check whether the thresholds have been exceeded 
            if h <= h_tol or rho1>=rho1_tol:
                print('\n\n\n\n##### OPTIMIZATION RESULTS #####')
                print('Optimizer successfully retrieved the matrix of coefficients W_bar: ', sol.success)
                print('Final value of objective function: ', sol.fun)
                print('|| W_bar-Z ||_F: ', norm(w_bar-z, 2))
                print('\nFinished at iteration number ', it, '. Dagness value equal to ', h,' (if DAG, h=0).')
                break
                
            if verbose: 
                print('##### OPTIMIZATION INFO #####')
                print('Current value of objective function: ', sol.fun)
                
                if it==0:
                    W_bar = self._adj(w_bar)
                    loss, _ = self._loss(W_bar)
                    ell1 = lmbd*np.sum(np.abs(w_bar))
                
                    print('Initial value of Frobenious norm: ', loss)
                    print('Initial value of reguralisation: ', ell1)

                print('|| W_bar-Z ||_F: ', norm(w_bar-z, 2))
                print('Ended iteration number ', it, '. Dagness value equal to ', h,' (if DAG, h=0).')
        
        #print('w-z: ', norm(w_bar-z, 2))
        #reshape the result
        W_bar = self._adj(w_bar)
        W_bar[np.abs(W_bar) <= thresh] = 0.
        
        C = np.zeros((self.lag+1, self.DN, self.DN))
        
        for l in range(self.lag+1):
            C[l,:,:] = (W_bar.T[:,l*self.DN:(l+1)*self.DN]).T
        
        t = time.time()-start
        
        if verbose:
            return C, (it, t), loss, ell1
        else:    
            return C, (it, t)