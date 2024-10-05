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



import numpy as np

class TimeMoment:
    def __init__(self,ndof):
        self.u = np.zeros(ndof)


class Element:
    def __init__(self,ndof,nt):
       self.M    = np.zeros((ndof,ndof))
       self.C    = np.zeros((ndof, ndof))
       self.K    = np.zeros((ndof, ndof))
       self.s    = np.zeros(2)
       self.f    = np.zeros(2)
       self.x    = np.zeros(2)
       self.time = []
       for i in range(nt):
           self.time.append(TimeMoment(ndof))





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
aff_m = np.loadtxt('afference.txt')
q = A - aff_m +np.ones_like(A)
d_A_m = np.max(q)
# d_A_m = np.max(np.abs(aff_m - A + np.ones_like(A))) # - A.reshape(A.shape[0]*A.shape[1])))


el = [Element(dof_el,T.shape[0]) for n in range(n_el)]
time = [TimeMoment(dof) for n in range(T.shape[0])]

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

# Element abscissae
for n in range(n_el):
    el[n].x=x_i+n*L_el+x_e

# Assemblate matrices and vectors

# Assemblage of mass matrix
from mass_matrix import assemble_mass_matrix
M=assemble_mass_matrix(el,dof,n_el,dof_el,A)

# Assemblage of convection matrix
from convection_matrix import assemble_convection_matrix
C=assemble_convection_matrix(el,dof,n_el,dof_el,A)

# Assemblage of diffusion matrix
from diffusion_matrix import assemble_diffusion_matrix
K = assemble_diffusion_matrix(el,dof,n_el,dof_el,A)

# Convection+Diffusion+Reaction matrix
D=C+K+sigma*M

# Assemblage of load vector
from load_vector import assemble_load_vector
f=assemble_load_vector(el,dof,n_el,dof_el,A)

# Definition of the constrained DOFs
dof_free=dof-len(dof_constrained)
n_dof_constrained=len(dof_constrained)

constrain_der_fun = []
for n in range(n_dof_constrained):
    g = diff(bound_cond_fun(t), t)
    constrain_der_fun.append(g)


# Evaluation of boundary conditions over time
u_p        = np.zeros((n_dof_constrained,T.shape[0]+1))
u_der_p    = np.zeros((n_dof_constrained,T.shape[0]+1))

constrain     = np.zeros(n_dof_constrained)
constrain_der = np.zeros(n_dof_constrained)
t = symbols('t')

for k,ti in enumerate(T):
    for n in range(n_dof_constrained):
        constrain[n] = bound_cond_fun(ti)
        constrain_der[n] =  diff(bound_cond_fun(t), t)
    u_p[:, k] = constrain.T
    u_der_p[:, k] = constrain_der.T



u_p = np.array(u_p)

# Mass matrix
from constrain import constrain_matrix
[M_ff,M_fp,M_pf,M_pp]=constrain_matrix(M,dof_constrained)

# Convection matrix
[C_ff,C_fp,C_pf,C_pp]=constrain_matrix(C,dof_constrained)

# Diffusion matrix
[K_ff,K_fp,K_pf,K_pp]=constrain_matrix(K,dof_constrained)

# Convection+Diffusion matrix
[D_ff,D_fp,D_pf,D_pp]=constrain_matrix(D,dof_constrained)
# Load vector
from constrain import constrain_vector
[f_f,f_p]=constrain_vector(f,dof_constrained);


u_0=u_0_fun(x).T
from constrain import constrain_vector
u_0_f,_ = constrain_vector(u_0,dof_constrained)

# Unsteady convectio-diffusion-reaction solution
u_f = np.zeros((T.shape[0]+1,u_0_f.shape[0]))
# Time integration
u_f[0,:]=u_0_f
for k,t in enumerate(T):
    MM = (M_ff+dt*theta*D_ff)
    MM_m = np.loadtxt('TimeMatrix_m.txt')
    MM_m = MM_m.reshape(MM.shape)
    d_MM = np.max(np.abs(MM-MM_m))
    u_p_m = np.loadtxt('u_p' +'_'+ str(k+1) + '.txt')
    d_u_p = np.max(np.abs(u_p[:,k+1]-u_p_m))
    D_fp_m = np.loadtxt('D_fp' +'_'+  str(k+1) + '.txt')
    d_D_fp = np.max(np.abs(D_fp - D_fp_m.reshape(D_fp.shape)))
    u_der_p_m = np.loadtxt('u_der_p' +'_'+  str(k+1) + '.txt')
    d_u_der_p = np.max(np.abs(u_der_p_m - u_der_p[:,k+1]))
    M_fp_m = np.loadtxt('M_fp' + '_'+ str(k+1) + '.txt')
    d_M_fp = np.max(np.abs(M_fp_m.reshape(M_fp.shape) - M_fp))
    # M_fp_m = np.loadtxt('M_fp' +'_'+  str(k+1) + '.txt')
    # d_M_fp = np.max(np.abs(M_fp_m - M_fp))
    # M_fp_m = np.loadtxt('M_fp' +'_'+  str(k+1) + '.txt')
    # d_M_fp = np.max(np.abs(M_fp_m - M_fp))
    f_f_m = np.loadtxt('f_f_' + str(k+1) +  '.txt')
    d_f_f = np.max(np.abs(f_f_m - f_f))

    # the operation M_fp*u_der_p[:,k+1] probably needs matrix multiplication
    A = M_fp
    b = u_der_p[:,k+1]
    x = np.matmul(A,b)
    br = f_f - x
    upk = u_p[:,k+1].reshape(u_p[:,k+1].shape[0],1)
    res =  np.matmul(D_fp,upk)
    br -= res.reshape(res.shape[0]) #*u_p[:,k+1]
    # matlab dimensionality is (149,2) X( 2,1) resulting in 149,1
    bb = np.matmul((M_ff-dt*(1-theta)*D_ff),u_f[k,:])
    bb_m = np.loadtxt('bb_' + str(k + 1) + '.txt')
    uf_init_m = np.loadtxt('uf_init_m_' + str(k + 1) + '.txt')  # uf_init_m_
    d_uf_init = np.max(np.abs(u_f[k,:] - uf_init_m))
    # d_bb too big at k == 2
    d_bb_init = np.max(np.abs(bb-bb_m))
    # Matlab     dt*theta*(f_f-M_fp*u_der_p(:,k+1)-D_fp*u_p(:,k+1))
    bb_1 =  dt*theta*(np.matmul(M_fp,u_der_p[:,k+1].reshape(u_der_p[:,k+1].shape[0],1))
            +np.matmul(D_fp,u_p[:,k+1].reshape(u_p[:,k+1].shape[0],1)))
    #bb1 = dt*theta*(f_f- bb_1.reshape(bb_1.shape[0],) - bb_2.reshape(bb_2.shape[0],))

    bb1_m = np.loadtxt('bb_u_f_1_' + str(k + 1) + '.txt')
    d_bb1  = np.max(np.abs(bb_1 - bb1_m))

    # matlab bb_2 = dt*(1-theta)*(f_f-M_fp*u_der_p(:,k)-D_fp*u_p(:,k))
    bb_2 =  dt*(1.0-theta)*(np.matmul(M_fp,u_der_p[:,k].reshape(u_der_p[:,k].shape[0],1))
            +np.matmul(D_fp,u_p[:,k].reshape(u_p[:,k].shape[0],1)))
    bb_2_m = np.loadtxt('bb_u_f_2_' + str(k + 1) + '.txt')
    d_bb2 = np.max(np.abs(bb_2 - bb_2_m))

    bb_final_m = np.loadtxt('bb_final_' + str(k + 1) + '.txt')
    #d_bb_final = np.max(np.abs(bb_final_m))
    bb += bb_1.reshape(bb_1.shape[0])+bb_2.reshape(bb_2.shape[0])
    d_bb_final = np.max(np.abs(bb - bb_final_m))


    # bb += dt*(1-theta)*(f_f-np.matmul(M_fp,u_der_p[:,k].reshape(u_der_p[:,k].shape[0],1))
    #                     -np.matmul(D_fp,u_p[:,k].reshape(u_p[:,k].shape[0],1)))



    # bb2_1 = np.matmul(M_fp, u_der_p[:, k].reshape(u_der_p[:, k].shape[0], 1))
    # bb2_2 = np.matmul(D_fp, u_der_p[:, k+1].reshape(u_p[:, k+1].shape[0], 1))
    # bb2 = dt * (1 - theta) * (f_f - bb2_1.reshape(bb2_1.shape[0]) - bb2_2.reshape(bb2_2.shape[0]))
    #
    # bb2_m = np.loadtxt('bb2_' + str(k + 1) + '.txt')
    # d_bb2 = np.max(np.abs(bb2 - bb2_m))


    #bb += bb2


    tv = np.linalg.solve(M_ff+dt*theta*D_ff,bb)
    tv_m = np.loadtxt('time_vector_'+str(k+1)+'.txt')
    d_tv = np.max(np.abs(tv-tv_m))
    print(k,d_tv)
    u_f[k + 1,:] = tv
    qq = 0

from data_all import data_all_dof
# Data for all dof
dt_k = np.zeros(T.shape[0])
for k in range(T.shape[0]):
    time[k].u=data_all_dof(u_f[k,:].T,u_p[:,k],dof_constrained)
    time_k_u_m = np.loadtxt('time_k_u_'+str(k+1)+'.txt')
    dt_k[k] = np.max(np.abs(time_k_u_m-time[k].u))

# Interpolation of the solution OK
A=afference_matrix(n_el,dof_el)
from interp import interpolation
A_m = np.loadtxt('A_interp.txt')
q = A - A_m +np.ones_like(A)
d_A_m = np.max(q)
interp_nk = np.zeros((n_el,T.shape[0]))
for n in range(n_el):
    for k in range(T.shape[0]):
        el[n].time[k].u=interpolation(n,time[k].u,A,n_e)
        time_k_u_m = np.loadtxt('interp_n_' +str(n+1)+ '_k_' + str(k + 1) + '.txt')
        interp_nk[n][k] = np.max(np.abs(time_k_u_m - el[n].time[k].u))

# Analytical solution
u_anal = np.zeros((n_el,T.shape[0]))

for k in range(T.shape[0]):
    t=T[k]
    alfa=np.sqrt(1+4*v*t/l**2)
    u_anal[:,k]=u_max/alfa*np.exp(-np.power(((x_p-x_0-a*t)/(l*alfa)),2))

qq = 0






