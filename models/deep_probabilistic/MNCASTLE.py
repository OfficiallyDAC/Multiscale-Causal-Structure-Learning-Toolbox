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
                 name_prefix="CausalOrder",
                 device='cpu'):
        
        self.T=T
        self.N=N
        self.name_prefix=name_prefix
        self.device=device
        self.zero = torch.zeros(1, device=self.device)
        self.one = torch.ones(1, device=self.device)
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

        M = torch.linalg.pinv(self.D-B*c0)

        #condition on X
        with pyro.plate(self.name_prefix+".Observed Data", T, dim=-1, device=self.device):
            pyro.sample(self.name_prefix+".X", dist.MultivariateNormal(self.zero.expand(N), covariance_matrix=M@M.T), obs=X)

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
            c0_scale = pyro.param(self.name_prefix+".c0_scale", self.one.expand(N,N), constraint=constraints.positive)
            c0 = pyro.sample(self.name_prefix+".c0", dist.Normal(c0_loc, c0_scale).mask(B))


class CausalCoeff(gpytorch.models.ApproximateGP):
    def __init__(self,
                 J,
                 T,
                 N,
                 kernel=None,
                #  kernel_spec=None,
                 frac_inducing=.64, 
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
        name_prefix: str, prefix of sites' names
        device: either 'cpu' or 'gpu'. Default: 'cpu'
        """
        
        assert isinstance(J, int) and J>0, "J is the numebr of scales"
        assert isinstance(T,int) and T>0, "T is the number of samples"
        assert isinstance(N, int) and N>1, "N is the number of variables"
        # assert set(['kernel', 'lengthscale_constraint', 'lengthscale_prior'])==set(kernel_spec.keys()), "Kernel dict must have as keys 'kernel', 'lengthscale_constraint', 'lengthscale_prior'"
        
        self.J = J
        self.T = T
        self.N = N
        self.device=device
        
        self.zero = torch.zeros(1, device=self.device)
        self.one = torch.ones(1, device=self.device)
        self.D = torch.eye(self.N,self.N,device=self.device)

        if kernel is None:
            kernel = gpytorch.kernels.MaternKernel(nu=2.5,lengthscale_constraint=gpytorch.constraints.GreaterThan(1.),lengthscale_prior=None)
            print("None")
        else:
            kernel = kernel
            print("Kernel", kernel)
        # if kernel_spec['kernel'] is None:
        #     kernel = gpytorch.kernels.RBFKernel()
        # else:
        #     kernel = kernel_spec['kernel'](lengthscale_constraint=kernel_spec['lengthscale_constraint'],
        #                                   lengthscale_prior=kernel_spec['lengthscale_prior'])
        #     assert isinstance(kernel, gpytorch.kernels.Kernel), "Kernel must be a valid (combination of) gpytorch.kernels.Kernel"
        
        assert 0<frac_inducing<=1, "frac_inducing must be in (0,1]"
        assert isinstance(name_prefix, str), "name_prefix must be a string"
        self.name_prefix = name_prefix
        
        # Define all the variational stuff
        num_inducing = int(frac_inducing*self.T)
        inducing_points = torch.linspace(0, 1, num_inducing).view(1,1,1,-1,1).repeat(self.J,self.N,self.N,1,1)
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points,
            gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=num_inducing,
                                                                batch_shape=torch.Size([self.J,self.N,self.N]),
                                                                mean_init_std=1.e-3)
        )

        # Standard initializtation
        super().__init__(variational_strategy)

        # Mean, covar, likelihood        
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([self.J,self.N,self.N]))
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel,
                                                         batch_shape=torch.Size([self.J,self.N,self.N]))
        
    def forward(self, time_steps):
        mean = self.mean_module(time_steps)
        covar = self.covar_module(time_steps)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, time_steps, ordering, S ):
        # Get q(f) - variational (guide) distribution of latent function
        function_dist = self.pyro_guide(time_steps)
        J = S.shape[0]
        assert J==self.J, "The number of scales J must be equal to the left-most dimension of S"
        
        # ordering=pyro.deterministic(self.name_prefix + ".causal_ordering", ordering)
        B=torch.tril(self.one.new_ones([self.N,self.N]), diagonal=-1)
        P_=make_permutation_matrix(ordering).to(self.device)
        B=(P_.T@B@P_).bool()
 
        scale_axis = pyro.plate(self.name_prefix + ".scale_plate", self.J, dim=-4, device=self.device)
        node_axis1 = pyro.plate(self.name_prefix + ".node_plate1", self.N, dim=-3, device=self.device)
        node_axis2 = pyro.plate(self.name_prefix + ".node_plate2", self.N, dim=-2, device=self.device)
        time_axis = pyro.plate(self.name_prefix + ".data_plate", self.T, dim=-1, device=self.device)
       
        with scale_axis, node_axis1,node_axis2, time_axis:
            function_samples = pyro.sample(self.name_prefix + ".f(time_steps)", function_dist.mask(
                B.view(1,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])))
        """  
        scale_axis_S = pyro.plate(self.name_prefix + ".scale_plate_S", J, dim=-4, device=self.device)
        node_axis1_S = pyro.plate(self.name_prefix + ".node_plate1_S", self.N, dim=-2, device=self.device)
        node_axis2_S = pyro.plate(self.name_prefix + ".node_plate2_S", self.N, dim=-1, device=self.device)
        time_axis_S = pyro.plate(self.name_prefix + ".data_plate_S", self.T, dim=-3, device=self.device)
        
        with scale_axis_S, time_axis_S, node_axis1_S, node_axis2_S:
            s_scale = pyro.param(self.name_prefix + ".S_scale_loc", .05*self.one, constraint=constraints.positive)
            S_scale = pyro.sample(self.name_prefix + ".S_scale", dist.Delta(s_scale))
        """
        
    def model(self, time_steps, ordering, S):
        
        pyro.module(self.name_prefix + ".gp", self)
        
        function_dist = self.pyro_model(time_steps)
        
        J = S.shape[0]
        assert J==self.J, "The number of scales J must be equal to the left-most dimension of S"
        
        # ordering=pyro.deterministic(self.name_prefix + ".causal_ordering", ordering)
        B=torch.tril(self.one.new_ones([self.N,self.N]), diagonal=-1)
        P_=make_permutation_matrix(ordering).to(self.device)

        B=(P_.T@B@P_).bool()
        
        scale_axis = pyro.plate(self.name_prefix + ".scale_plate", self.J, dim=-4, device=self.device)
        node_axis1 = pyro.plate(self.name_prefix + ".node_plate1", self.N, dim=-3, device=self.device)
        node_axis2 = pyro.plate(self.name_prefix + ".node_plate2", self.N, dim=-2, device=self.device)
        time_axis = pyro.plate(self.name_prefix + ".data_plate", self.T, dim=-1, device=self.device)
        
        with scale_axis, node_axis1, node_axis2, time_axis:
            function_samples = pyro.sample(self.name_prefix + ".f(time_steps)", function_dist.mask(
                B.view(1,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])))
        
        M = torch.linalg.pinv((self.D.view(1,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])- \
                               B.view(1,self.N,self.N,1).expand([self.J, self.N,self.N,self.T])*function_samples).transpose(1,3).transpose(2,3))
        #here condition on S
        scale_axis_S = pyro.plate(self.name_prefix + ".scale_plate_S", J, dim=-4, device=self.device)
        node_axis1_S = pyro.plate(self.name_prefix + ".node_plate1_S", self.N, dim=-2, device=self.device)
        node_axis2_S = pyro.plate(self.name_prefix + ".node_plate2_S", self.N, dim=-1, device=self.device)
        time_axis_S = pyro.plate(self.name_prefix + ".data_plate_S", self.T, dim=-3, device=self.device)

        with scale_axis_S, time_axis_S, node_axis1_S, node_axis2_S:
            S_hat = pyro.sample(self.name_prefix+".S", dist.Normal(torch.einsum("jtnm,jtmk->jtnk", M, M.transpose(-2,-1)),.05), obs=S)