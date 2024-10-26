# TODO: matrix A in line 317 is wrong: both size and values at A[0,:] cf A(1,:)

# %% 1D Unsteady convection-diffusion-reaction problem
# % Based on "Finite Element Methods for flow problems" of
# % Jean Donea and Antonio Huerta
#
# % Andrea La Spina
# % https://it.linkedin.com/in/andrealaspina
# % https://independent.academia.edu/AndreaLaSpina
# % https://www.researchgate.net/profile/Andrea_La_Spina
#
# %% Equation
#
# % u_t+a*u_x-v*u_xx+sigma*u=s(x)           in [x_i,x_f]x]t_i,t_f[
# % u(x,0)=u0(x)                            on [x_i,x_f]
# % u(x1,t)=u_1(t);                         on x1 for t in [t_i,t_f]
# % u(x2,t)=u_2(t);                         on x2 for t in [t_i,t_f]
import matplotlib.pyplot as plt
import numpy as np






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
    t = u_max*np.exp(-np.power((x-x_0)/l,2.0))
    return t                            # Initial condition
from sympy import *
t = symbols('t')
def bound_cond_fun(t):
    return 0.0*t                        # Boundary condition
#dof_constrained_string='[1,dof]';       # Degree of freedom constrained
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
steps_per_el = 10                        # number of steps per element
dx_e=L_el/10                             # Spatial step-numerical interp.
x_e=np.linspace(0,L_el,steps_per_el+1)   # Space vector-numerical interp.
n_e=x_e.shape[0]                         # Number of points in space vector
s=s_fun(x)*np.ones(x.shape[0])           # Numerical value of source term
v=a*h/(2*Pe)                             # Diffusivity coefficient
dt=h/a*Courant                           # Time step
T=np.arange(t_i,t_f,dt)                  # Time vector
dof_constrained = [0,dof-1]                # Degree of freedom constrained
# Evaluation of beta





if __name__ == '__main__':
    from conv_diff import unsteady_convection_diffusion_reaction
    xt,yt,d_yt = unsteady_convection_diffusion_reaction(
        n_gauss,
        L_el,
        n_el,
        dof_el,
        dof,
        T,
        a,
        v,
        x_i,
        x_e,
        sigma,
        dof_constrained,
        bound_cond_fun,
        x_p,
        x_0,
        u_max,
        l,
        dt,
        theta,
        n_e,
        x_f,
        u_0_fun,
        h,
        s,
        x
    )
    qq = 0


