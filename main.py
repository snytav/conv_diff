import numpy as np

class Element:
    def __init__(self,ndof):
       self.M = np.zeros((ndof,ndof))
       self.C = np.zeros((ndof, ndof))
       self.K = np.zeros((ndof, ndof))
       self.s = np.zeros(2)
       self.f = np.zeros(2)





x_i=0                                  # Initial point
x_f=1                                  # Final point
a=1                                    # Convection velocity
sigma=0                                # Reaction coefficient

def s_fun(x):
    return 0                          # Source term
u_max=5.0/7.0                         # Peak of Gauss hille
x_0=2.0/15.0                          # Centre of Gauss hill
l=7*np.sqrt(2)/300.0                  # Width of Gauss hill

def u_0_fun(x):
    t = u_max*np.exp(-((x-x_0)/l),2.0)
    return t                            # Initial condition

def bound_cond_fun(t):
    return 0.0                          # Boundary condition
Pe=1                                    # Péclet number
Courant=1                               # Courant number
t_i=0                                   # Initial time
t_f=0.6                                 # Final time
animation_time=10                       # Animation time
n_el=150                                # Number of finite elements
n_gauss=2                               # Number of Gauss points
polynomial_degree=1                     # Shape functions polynomial degree
FE_type='Galerkin'                      # Type of FE (Galerkin or Upwind)
theta=1/2                               # Theta for time integration scheme
                                        # ( 0  = Forward Euler)
                                        # (1/2 = Crank-Nicolson)
                                        # (2/3 = Galerkin)
                                        # ( 1  = Backward Euler)

# # # Derived parameters

L=x_f-x_i                                # Domain length
n_np=polynomial_degree*n_el+1            # Number of nodal points
n_eq=polynomial_degree*n_el-1            # Number of equations
dof_el=polynomial_degree+1               # Number of DOFs per element
dof=n_np                                 # Total number of DOFs
L_el=L/n_el                              # Length of a finite element
h=L_el/polynomial_degree                 # Spatial step
x = np.linspace(x_i,x_f,n_el+1)          # Space vector
dx_p=L/n_el                              # Spatial step-analytical solution
x_p=np.arange(x_i,x_f,dx_p)              # Space vector-analytical solution
dx_e=L_el/10                             # Spatial step-numerical interp.
x_e=np.arange(0,L_el,dx_e)               # Space vector-numerical interp.
n_e=x_e.shape[0]                         # Number of points in space vector
s=s_fun(x)*np.ones(x.shape[0])           # Numerical value of source term
v=a*h/(2*Pe)                             # Diffusivity coefficient
dt=h/a*Courant                           # Time step
T=np.arange(t_i,t_f,dt)                  # Time vector

# Evaluation of beta
v_arr = v*np.ones(n_el)
a_arr = a*np.ones(n_el)

beta = 0

# Gauss parameters
from gauss import Gauss_parameters

xi,w=Gauss_parameters(n_gauss)

# Trasformation of coordinated for the Gauss integration points
x_gauss=L_el/2*(np.ones_like(xi)+xi)

# Jacobian of the transformation
J=h/2

# Computation of shape and test functions (and derivatives) at Gauss points
from gauss import shape_functions_Gauss_points,test_functions_Gauss_points
N,dN=shape_functions_Gauss_points(xi)
W,dW=test_functions_Gauss_points(xi,beta)


# Afference matrix
from afference import afference_matrix
A=afference_matrix(n_el,dof_el)

el = [Element(dof_el) for n in range(n_el)]

# Element mass matrix
from matrix import element_mass_matrix,element_convection_matrix,element_diffusion_matrix
for n in range(n_el):
    el[n].M=element_mass_matrix(dof_el,n_gauss,N,W,w,J)

# Element convection matrix
for n in range(n_el):
    el[n].C=element_convection_matrix(a_arr[n],dof_el,n_gauss,dN,W,w,J)

# Element diffusion matrix
for n in range(n_el):
    el[n].K=element_diffusion_matrix(v_arr[n],dof_el,n_gauss,dN,dW,w,J)

from matrix import element_load_vector
# Element load vector
for n in range(n_el):
    mlab_n = n+1
    mlab_first = (mlab_n - 1) * (dof_el - 1) + 1
    mlab_last  =       mlab_n * (dof_el - 1) + 1
    el[n].s=s[mlab_first-1:mlab_last]
    el[n].f=element_load_vector(el[n].s,dof_el,n_gauss,N,W,w,J)
    qq = 0




qq = 0






