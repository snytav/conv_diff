import numpy as np
from sympy import *



def get_matrices_and_vectors(v,a,n_el,n_gauss,L_el,h,dof_el):
    beta = 0
    v_arr = v * np.ones(n_el)
    a_arr = a * np.ones(n_el)
    t = symbols('t')

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

    return beta,v_arr,a_arr,t,N,dN,W,dW,A,J,w,d_A_m
