import torch
import gpytorch
import pyro
import pyro.distributions as dist

from distributions.Plackett_Luce import PlackettLuce, make_permutation_matrix 
from torch.distributions import constraints

class CausalOrder():
    def __init__(self,
                 T,
                 N,
                 scale_cc=1.,
                 name_prefix="CausalOrder",
                 device='cpu'):
        
        self.T=T
        self.N=N
        self.scale_cc=scale_cc
        self.name_prefix=name_prefix
        self.device=device
        self.zero = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.one = torch.ones(1, dtype=torch.float32,device=self.device)
        self.pl_init = .5*self.one.expand(self.N)
        self.D = torch.eye(self.N,self.N,device=self.device)
            
    def model(self, X):
        #dataset size
        T, N = X.size()
        
        #sample the causal ordering
        ordering = pyro.sample(self.name_prefix+".causal_ordering", PlackettLuce(self.pl_init))
        
        B=torch.tril(self.one.new_ones([N,N]), diagonal=-1)
        P_=make_permutation_matrix(ordering).to(self.device)
        B=(P_.T@B@P_).bool()
        
        nodes_axis1 = pyro.plate(self.name_prefix+".Nodes1", N, dim=-2, device=self.device)
        nodes_axis2 = pyro.plate(self.name_prefix+".Nodes2", N, dim=-1, device=self.device)

        with nodes_axis1,nodes_axis2:
            c0 = pyro.sample(self.name_prefix+".c0", dist.Normal(self.zero,
                                               self.one).expand([N,N]).mask(B))
            
        IC=self.D-B*c0
        
        #condition on X
        with pyro.plate(self.name_prefix+".Observed Data", T, dim=-1, device=self.device):
            pyro.sample(self.name_prefix+".X", dist.MultivariateNormal(self.zero.expand(N), precision_matrix=IC.T@IC), obs=X)
            
    def guide(self, X):

        #dataset size
        T, N = X.size()
        
        b = pyro.param(self.name_prefix+".theta", self.pl_init, constraint=constraints.real_vector)
        
        ordering = pyro.sample(self.name_prefix+".causal_ordering", PlackettLuce(b, validate_args=True),
                                   infer=dict(baseline={'use_decaying_avg_baseline': True,
                                             'baseline_beta': 0.95})
                                  )
        
        B=torch.tril(self.one.new_ones([N,N]), diagonal=-1)
        P_=make_permutation_matrix(ordering).to(self.device)
        B=(P_.T@B@P_).bool()
        
        #mark conditional independence 
        #among variables and scales.
        #Please notice that we 
        #sample tensors, the 
        #plate is useful to mark the
        #independent dimensions

        nodes_axis1 = pyro.plate(self.name_prefix+".Nodes1", N, dim=-2, device=self.device)
        nodes_axis2 = pyro.plate(self.name_prefix+".Nodes2", N, dim=-1, device=self.device)

        with nodes_axis1,nodes_axis2:
            c0_loc = pyro.param(self.name_prefix+".c0_loc", self.zero.expand(N,N))
            c0_scale = pyro.param(self.name_prefix+".c0_scale", self.scale_cc*self.one.expand(N,N), constraint=constraints.positive)
            c0 = pyro.sample(self.name_prefix+".c0", dist.Normal(c0_loc, c0_scale).mask(B))

            
class CausalCoeff(gpytorch.models.ApproximateGP):
    def __init__(self,
                J,
                T,
                N,
                mask,
                kernel=None,
                mean_prior=None,
                beta=1.,
                Chol_init_std=1.e-3,
                frac_inducing=.64,
                variational_dist=0, 
                outputscale_prior=None,
                name_prefix="CausalCoeff",
                device='cpu'):
        """
         INPUT
        =====
        J: int, number of scales
        T: int, number of observations
        N: int, number of variables
        kernel_spec: dict, kernel choice. It is supposed to contain the following keys:
            - kernel: class, a valid (combination) of gpytorch.kernels.Kernel. Default: RBF()
            - lengthscale_constraint: class, instance of gpytorch.constraints. Default: None
            - lengthscale_prior: class, one of gpytorch priors. Default: None
        frac_inducing: float, fraction of total points used for variational inference. It belongs to (0,1.]
        variational_dist: int, 0-Cholesky; 1-MeanField; 2-Delta. 
        name_prefix: str, prefix of sites' names
        device: either 'cpu' or 'gpu'. Default: 'cpu'
        """
        
        assert isinstance(J, int) and J>0, "J is the numebr of scales"
        assert isinstance(T,int) and T>0, "T is the number of samples"
        assert isinstance(N, int) and N>1, "N is the number of variables"
        assert isinstance(beta, float) and beta>0., "Beta scales KL"
        assert isinstance(variational_dist, int) and 0<=variational_dist<=2, "variational distribution must be in {0,1,2}"
        # assert set(['kernel', 'lengthscale_constraint', 'lengthscale_prior'])==set(kernel_spec.keys()), "Kernel dict must have as keys 'kernel', 'lengthscale_constraint', 'lengthscale_prior'"
        
        self.J = J
        self.T = T
        self.N = N
        self.mask=mask
        self.beta = beta
        self.device=device
        
        self.zero = torch.zeros(1, device=self.device)
        self.one = torch.ones(1, device=self.device)
        self.D = torch.eye(self.N,self.N,device=self.device)

        if kernel is None:
            kernel=gpytorch.kernels.MaternKernel(nu=2.5,lengthscale_constraint=gpytorch.constraints.GreaterThan(1.),lengthscale_prior=None)
            #print("None")
        else:
            kernel = kernel
            #print("Kernel", kernel)
        
        assert 0<frac_inducing<=1, "frac_inducing must be in (0,1]"
        assert isinstance(name_prefix, str), "name_prefix must be a string"
        self.name_prefix = name_prefix
        
        # Define all the variational stuff
        num_inducing = int(frac_inducing*self.T)
        inducing_points = torch.linspace(0, 1, num_inducing).view(1,1,1,-1,1).repeat(self.J,self.N,self.N,1,1)

        if variational_dist==0:
            var_dist=gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing,
                                                                batch_shape=torch.Size([self.J,self.N,self.N]),
                                                                mean_init_std=Chol_init_std)
        elif variational_dist==1:
            var_dist=gpytorch.variational.MeanFieldVariationalDistribution(num_inducing_points=num_inducing,
                                                                batch_shape=torch.Size([self.J,self.N,self.N]),
                                                                mean_init_std=Chol_init_std)
        elif variational_dist==2:
            var_dist=gpytorch.variational.DeltaVariationalDistribution(num_inducing_points=num_inducing,
                                                                batch_shape=torch.Size([self.J,self.N,self.N]),
                                                                mean_init_std=Chol_init_std)
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
        self, inducing_points,
        var_dist,
        learn_inducing_locations=True)
              

        # Standard initializtation
        super().__init__(variational_strategy)
               
        # Mean, covar, likelihood        
        self.mean_module = gpytorch.means.ConstantMean(constant_prior=mean_prior, batch_shape=torch.Size([self.J,self.N,self.N]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel,
                                                         batch_shape=torch.Size([self.J,self.N,self.N]),
                                                         outputscale_prior=outputscale_prior)        
        
    def forward(self, time_steps):
        mean = self.mean_module(time_steps)
        covar = self.covar_module(time_steps)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, time_steps, O):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(time_steps, beta=self.beta)

        J = O.shape[0]
        assert J==self.J, "The number of scales J must be equal to the left-most dimension of O"
        
        scale_axis = pyro.plate(self.name_prefix + ".scale_plate", self.J, dim=-4, device=self.device)
        node_axis1 = pyro.plate(self.name_prefix + ".node_plate1", self.N, dim=-3, device=self.device)
        node_axis2 = pyro.plate(self.name_prefix + ".node_plate2", self.N, dim=-2, device=self.device)
        time_axis = pyro.plate(self.name_prefix + ".data_plate", self.T, dim=-1, device=self.device)
       
        with scale_axis, node_axis1,node_axis2, time_axis:
            function_samples = pyro.sample(self.name_prefix + ".f(time_steps)", function_dist.mask(
                self.mask.view(self.J,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])))
        
    def model(self, time_steps, O):
        
        pyro.module(self.name_prefix + ".gp", self)

        function_dist = self.pyro_model(time_steps, beta=self.beta)

        
        J = O.shape[0]
        assert J==self.J, "The number of scales J must be equal to the left-most dimension of S"

        scale_axis = pyro.plate(self.name_prefix + ".scale_plate", self.J, dim=-4, device=self.device)
        node_axis1 = pyro.plate(self.name_prefix + ".node_plate1", self.N, dim=-3, device=self.device)
        node_axis2 = pyro.plate(self.name_prefix + ".node_plate2", self.N, dim=-2, device=self.device)
        time_axis = pyro.plate(self.name_prefix + ".data_plate", self.T, dim=-1, device=self.device)
        
        with scale_axis, node_axis1, node_axis2, time_axis:
            function_samples = pyro.sample(self.name_prefix + ".f(time_steps)", function_dist.mask(
                self.mask.view(self.J,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])))

        IC=(self.D.view(1,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])- \
            self.mask.view(self.J,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])*function_samples).transpose(1,3).transpose(2,3)
        
        #here condition on O
        scale_axis_O = pyro.plate(self.name_prefix + ".scale_plate_O", J, dim=-4, device=self.device)
        time_axis_O = pyro.plate(self.name_prefix + ".data_plate_O", self.T, dim=-3, device=self.device)

        with scale_axis_O, time_axis_O:
            O_hat = pyro.sample(self.name_prefix+".O", dist.Normal(torch.einsum("jtnm,jtmk->jtnk",IC.transpose(-2,-1),IC),
                                                                   torch.ones(1)).to_event(2), obs=O)