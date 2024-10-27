import numpy as np
import matplotlib.pyplot as plt
from sympy import *

def unsteady_convection_diffusion_reaction(
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
):


    from matrices_and_vectors import get_matrices_and_vectors
    beta, v_arr, a_arr, t, N, dN, W, dW, A, J,w,err = get_matrices_and_vectors(v,a,n_el,n_gauss,L_el,h,dof_el)
    # d_A_m = np.max(np.abs(aff_m - A + np.ones_like(A))) # - A.reshape(A.shape[0]*A.shape[1])))

    from time_integr import time_integration
    el,time,u_f,u_p,d_tv = time_integration(dof_el,n_el,dof,n_gauss, N, W, w, J,a_arr,dN,v_arr,dW,x_i,L_el,x_e,A,sigma,
                     dof_constrained,bound_cond_fun,T,u_0_fun,x,dt,theta,s)

    from interpolation_module import interpolation
    el,d_interp_nk = interpolation(T, time, u_f, u_p, dof_constrained, n_el, dof_el, n_e, el)




    # Analytical solution
    from analytics import get_analytical_solution
    u_anal, d_u_anal = get_analytical_solution(n_el,T,u_max,l,x_p,x_0,a,v)

    qq = 0

    # %% Plot of solutions
    # to Sat morning:
    # 1. find analogue of plot(x_p,u_anal(:,end),'b:','LineWidth',2)
    # 2. check values
    # 3. reproduce plot
    # 4. organize as function
    # 5. if possible check with some external example (e.g. Lorena Barba's)
    # 6. solve Lavrentiev's problem.
    from plot_solution import compare_plot
    xt, yt, d_yt = compare_plot(x_p,u_anal,n_el,el,x_i,x_f)

    return xt,yt,d_yt
