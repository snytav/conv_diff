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
    d_interp_nk = interpolation(T, time, u_f, u_p, dof_constrained, n_el, dof_el, n_e, el)




    # Analytical solution
    u_anal = np.zeros((n_el,T.shape[0]))

    d_u_anal_m = np.zeros(T.shape[0])
    for k in range(T.shape[0]):
        t=T[k]
        alfa=np.sqrt(1+4*v*t/l**2)
        u_anal[:,k]=u_max/alfa*np.exp(-np.power(((x_p-x_0-a*t)/(l*alfa)),2))
        u_anal_m = np.loadtxt('u_anal_k_'+str(k+1)+'.txt')
        d_u_anal_m[k] = np.max(np.abs(u_anal_m[:u_anal[:,k].shape[0]]-u_anal[:,k]))
        qq = 0
    d_u_anal = np.max(d_u_anal_m)

    qq = 0

    # %% Plot of solutions
    # to Sat morning:
    # 1. find analogue of plot(x_p,u_anal(:,end),'b:','LineWidth',2)
    # 2. check values
    # 3. reproduce plot
    # 4. organize as function
    # 5. if possible check with some external example (e.g. Lorena Barba's)
    # 6. solve Lavrentiev's problem.
    plt.figure()
    # axes('FontSize',14)
    plt.plot(x_p,u_anal[:,0],label='Initial condition',color='green')



    xt = np.zeros(n_el)
    yt = np.zeros(n_el)
    for n in range(n_el):
        xt[n] = el[n].x[0]
        yt[n] = el[n].time[-1].u[0]
    plt.plot(xt,yt,'o', label='Numerical solution', color='blue')
    plt.plot(x_p,u_anal[:,-1],label='Analytical solution',color='red')
    plt.title('Analytical and numerical solution')
    plt.legend()
    plt.xlim(x_i,x_f)
    plt.show(block=True)
    yt_m = np.loadtxt('yt_final.txt')
    d_yt = np.max(np.abs(yt-yt_m))

    return xt,yt,d_yt
