import math
import numpy as np
import pyro.contrib.gp as gp
import pyro.distributions as dist
import torch

from scipy.signal import fftconvolve
from utils import _discretewv
from distributions.Plackett_Luce import PlackettLuce, make_permutation_matrix 

class Generator():
    
    def __init__(self, T, N, multiscale, nonstationarity, density, kernel=None):
        """
        INPUT
        =====
        T: int, number of observations
        N: int, number of variables
        multiscale: float, it determines the number of contemporaneous scales
            contributing to the process
        nonstationarity: float, it determines the nonstationarity of causal coefficients
        density: float, density of causal structure
        kernel: object, a valid (combination of) Pyro kernel(s)
        """
        
        assert isinstance(T, int) and T>0, 'T must be a positive integer'
        assert isinstance(N, int) and N>0, 'N must be a positive integer'
        assert isinstance(multiscale, (float,int)) and 0<=multiscale<=1, "multiscale must be in [0,1]"
        assert isinstance(nonstationarity, (float,int)) and 0<=nonstationarity<=1, "nonstationarity must be in [0,1]"
        assert isinstance(density, (float,int)) and 0<=density<=1, "density must be in [0,1]"
        
        self.T=T
        self.N=N
        self.multiscale=multiscale
        self.nonstationarity=nonstationarity
        self.density=density
        
        self.zero = torch.zeros(1)
        self.one = torch.ones(1)
        self.D = torch.eye(self.N,self.N)
        
        if kernel==None:
            self.kernel = gp.kernels.RBF(1, variance=.1*self.one, lengthscale=1/torch.tensor(self.nonstationarity))
        else:
            self.kernel=kernel

    def sample_noise(self):
        """
        Function to generate i.i.d noise tensor Z, size (J,T,N)
        """
        #here we sample the number of contributing scales
        J = self.one + dist.Binomial(int(np.log2(self.T))-1, self.multiscale).sample()

        #store the number of scales
        #for future use
        self.J = J.long()
        
        #additional observations needed 
        #for valid convolution computation
        self.to_add = 2**(self.J)-1
        
        #latent variable is 
        #distributed according 
        #a multivariate normal N(0,I)
        self.Z = dist.Normal(self.zero, self.one).expand([self.J,self.T+self.to_add,self.N]).sample()
        
    def sample_GP(self):
        xs = 2*math.pi*torch.linspace(0,1,self.T+self.to_add.item())
        self.cov=self.kernel.forward(xs)+.001*self.one.expand([self.T+self.to_add.item()]).diag()
        CR = dist.MultivariateNormal(
            self.zero.expand([self.T+self.to_add.item()]), covariance_matrix=self.cov
        ).expand([self.J, self.N, self.N]).sample()
        
        return CR
    
    def causal_matrix(self):
        """
        Function to generate the causal structure.
        
        OUTPUT
        ======
        C: torch.tensor, size [self.J, self.N, self.N]
        """
        
        #small constant to ensure strict
        #positivity of the scale of the noise
        #associated with the mean reversion
        #process
        eps = 1.e-20
        
        #the causal structure is defined by three components:
        # - C0: this is a constant equal for all scales and at all timestamps
        # - CM: this tensor varies across scales and is constant along time
        # - C0M: is the summation of C0, CM
        # - CR: within each scale, it implies perturbations along time to causal structure.
        #   This components is a GP. 
        
        #instead of having 2 distinct components, 
        #compute a unique distribution
        var = self.one + self.multiscale*self.multiscale
        C0M = self.one.expand([self.J,self.T+self.to_add,1,1])*dist.Normal(self.zero,torch.sqrt(var)).expand([self.J,1,self.N, self.N]).sample()
        
        #this must be a smooth function
        
        mask = dist.Bernoulli(self.density).expand([self.J, 1, self.N, self.N]).sample()
        
        #Mask for managing the structure density
        ones = torch.tril(self.one.expand([self.N,self.N]), diagonal=-1).expand([self.J,self.T+self.to_add,self.N,self.N])
        Mask = mask*ones

        #Causal ordering
        self.b = dist.Uniform(self.zero.expand([self.N]), self.N*self.one.expand([self.N])).to_event(1).sample()
        self.ordering = PlackettLuce(self.b).sample()
        P = self.one.expand([1,self.T+self.to_add,1,1])*make_permutation_matrix(self.ordering.expand([self.J,1,self.N]))
        P_prime = P.transpose(-2,-1)
        
        #Make products:
        #-C lower triangular
        #-introduce the causal ordering
        self.C0M = Mask*C0M
        CR = self.sample_GP().transpose(1,-1)
        self.CR = Mask*CR
        self.C_bar = Mask*(C0M + self.nonstationarity*CR)
        C_prime = torch.einsum('jtmn,jtnl,jtlp->jtmp',P_prime, self.C_bar, P)

        return C_prime
        
    def mixing_function(self):
        """
        Function to compute the mixing function from C.
        
        OUTPUT
        ======
        M: torch.tensor, size [self.J, self.N, self.N] 
        
        """
        
        self.C = self.causal_matrix()
        self.M = torch.linalg.pinv(self.D.expand([self.J, self.T+self.to_add, self.N, self.N])-self.C)
        
    def convolve(self):
        """
        Function to generate time series components at each scale from EWS
        """
        #components over each scale
        self.X_j = torch.tensor([])
        #undecimated discrete wavelets
        discretewv = _discretewv(self.N, 'db1', self.J.item())
        
        for j in range(self.J):
                        
            item = torch.tensor(fftconvolve(self.random_S_sqrt[j], discretewv[j], mode='valid', axes=0)).unsqueeze(0)
            self.X_j = torch.cat((self.X_j, item), axis=0)
            
    def generate(self):
        """
        Function to generate the dataset X of size [T,N].
        
        OUTPUT
        ======
        X: torch.tensor, size [self.T, self.N]
        """
        
        self.sample_noise() 
        self.mixing_function()
        self.S = torch.einsum("jtnm, jtmk -> jtnk", self.M, self.M.transpose(-2,-1))[:,-self.T:]
        
        self.random_S_sqrt = torch.einsum("jtnm, jtm -> jtn", self.M, self.Z)
        
        self.convolve()
        
        return self.X_j.sum(axis=0)