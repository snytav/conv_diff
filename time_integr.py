import numpy as np
import torch
from sympy import *

def time_integration(dof_el,n_el,dof,n_gauss, N, W, w, J,a_arr,dN,v_arr,dW,x_i,L_el,x_e,A,sigma,
                     dof_constrained,bc,T,u_0,x,dt,theta,s):

    from element import Element, TimeMoment

    t = symbols('t')

    el = [Element(dof_el, T.shape[0]) for n in range(n_el)]
    time = [TimeMoment(dof) for n in range(T.shape[0])]

    # Element mass matrix
    from matrix import element_mass_matrix, element_convection_matrix, element_diffusion_matrix

    for n in range(n_el):
        el[n].M = element_mass_matrix(dof_el, n_gauss, N, W, w, J)

    # Element convection matrix
    for n in range(n_el):
        el[n].C = element_convection_matrix(a_arr[n], dof_el, n_gauss, dN, W, w, J)

    # Element diffusion matrix
    for n in range(n_el):
        el[n].K = element_diffusion_matrix(v_arr[n], dof_el, n_gauss, dN, dW, w, J)

    from matrix import element_load_vector

    # Element load vector
    for n in range(n_el):
        mlab_n = n + 1
        mlab_first = (mlab_n - 1) * (dof_el - 1) + 1
        mlab_last = mlab_n * (dof_el - 1) + 1
        el[n].s = s[mlab_first - 1:mlab_last]
        el[n].f = element_load_vector(el[n].s, dof_el, n_gauss, N, W, w, J)
        qq = 0

    # Element abscissae
    for n in range(n_el):
        el[n].x = x_i + n * L_el + x_e

    # Assemblate matrices and vectors

    # Assemblage of mass matrix
    from mass_matrix import assemble_mass_matrix

    M = assemble_mass_matrix(el, dof, n_el, dof_el, A)

    # Assemblage of convection matrix
    from convection_matrix import assemble_convection_matrix

    C = assemble_convection_matrix(el, dof, n_el, dof_el, A)

    # Assemblage of diffusion matrix
    from diffusion_matrix import assemble_diffusion_matrix

    K = assemble_diffusion_matrix(el, dof, n_el, dof_el, A)

    # Convection+Diffusion+Reaction matrix
    D = C + K  #+ sigma * M

    # Assemblage of load vector
    from load_vector import assemble_load_vector

    f = assemble_load_vector(el, dof, n_el, dof_el, A)

    # Definition of the constrained DOFs
    dof_free = dof - len(dof_constrained)
    n_dof_constrained = len(dof_constrained)

    constrain_der_fun = bc
    for n in range(n_dof_constrained):
        g = bc[n]
        constrain_der_fun.append(g)

    # Evaluation of boundary conditions over time
    u_p = np.zeros((n_dof_constrained, T.shape[0] + 1))
    u_der_p = np.zeros((n_dof_constrained, T.shape[0] + 1))

    constrain = np.zeros(n_dof_constrained)
    constrain_der = np.zeros(n_dof_constrained)
    t = symbols('t')

    for k, ti in enumerate(T):
        for n in range(n_dof_constrained):
            constrain[n] = bc[n]
            constrain_der[n] = constrain_der_fun[n]
        u_p[:, k] = constrain.T
        u_der_p[:, k] = constrain_der.T

    u_p = np.array(u_p)

    # Mass matrix
    from constrain import constrain_matrix

    [M_ff, M_fp, M_pf, M_pp] = constrain_matrix(M, dof_constrained)

    # Convection matrix
    [C_ff, C_fp, C_pf, C_pp] = constrain_matrix(C, dof_constrained)

    # Diffusion matrix
    [K_ff, K_fp, K_pf, K_pp] = constrain_matrix(K, dof_constrained)

    # Convection+Diffusion matrix
    [D_ff, D_fp, D_pf, D_pp] = constrain_matrix(D, dof_constrained)
    # Load vector
    from constrain import constrain_vector

    [f_f, f_p] = constrain_vector(f, dof_constrained);

    u_0 = u_0.T
    from constrain import constrain_vector

    u_0_f, _ = constrain_vector(u_0, dof_constrained)

    # Unsteady convectio-diffusion-reaction solution
    u_f = torch.zeros((T.shape[0] + 1, u_0_f.shape[0]))
    # Time integration
    u_f[0, :] = torch.from_numpy(u_0_f)
    u_f = u_f.double()
    for k, t in enumerate(T):
        MM = (torch.from_numpy(M_ff) + dt * theta * D_ff)
       # MM_m = np.loadtxt('TimeMatrix_m.txt')
       #  MM_m = MM_m.reshape(MM.shape)
       #  d_MM = np.max(np.abs(MM - MM_m))
       #  u_p_m = np.loadtxt('u_p' + '_' + str(k + 1) + '.txt')
       #  d_u_p = np.max(np.abs(u_p[:, k + 1] - u_p_m))
       #  D_fp_m = np.loadtxt('D_fp' + '_' + str(k + 1) + '.txt')
       #  d_D_fp = np.max(np.abs(D_fp - D_fp_m.reshape(D_fp.shape)))
       #  u_der_p_m = np.loadtxt('u_der_p' + '_' + str(k + 1) + '.txt')
       #  d_u_der_p = np.max(np.abs(u_der_p_m - u_der_p[:, k + 1]))
       #  M_fp_m = np.loadtxt('M_fp' + '_' + str(k + 1) + '.txt')
       #  d_M_fp = np.max(np.abs(M_fp_m.reshape(M_fp.shape) - M_fp))
        # M_fp_m = np.loadtxt('M_fp' +'_'+  str(k+1) + '.txt')
        # d_M_fp = np.max(np.abs(M_fp_m - M_fp))
        # M_fp_m = np.loadtxt('M_fp' +'_'+  str(k+1) + '.txt')
        # d_M_fp = np.max(np.abs(M_fp_m - M_fp))
        # f_f_m = np.loadtxt('f_f_' + str(k + 1) + '.txt')
        # d_f_f = np.max(np.abs(f_f_m - f_f))

        # the operation M_fp*u_der_p[:,k+1] probably needs matrix multiplication
        A = M_fp
        b = u_der_p[:, k + 1]
        x = np.matmul(A, b)
        br = f_f - x
        upk = u_p[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1)
        res = torch.matmul(D_fp.double(), torch.from_numpy(upk))
        br = torch.from_numpy(br)
        br -= res.reshape(res.shape[0])  # *u_p[:,k+1]
        # matlab dimensionality is (149,2) X( 2,1) resulting in 149,1
        bb = torch.matmul(torch.from_numpy(M_ff) - dt * (1 - theta) * D_ff.double() , u_f[k, :])
        # bb_m = np.loadtxt('bb_' + str(k + 1) + '.txt')
        # uf_init_m = np.loadtxt('uf_init_m_' + str(k + 1) + '.txt')  # uf_init_m_
        # d_uf_init = np.max(np.abs(u_f[k, :] - uf_init_m))
        # d_bb too big at k == 2
        # d_bb_init = np.max(np.abs(bb - bb_m))
        # Matlab     dt*theta*(f_f-M_fp*u_der_p(:,k+1)-D_fp*u_p(:,k+1))
        bb_1 = dt * theta * (torch.matmul(torch.from_numpy(M_fp), torch.from_numpy(u_der_p[:, k + 1]).reshape(u_der_p[:, k + 1].shape[0], 1))
                             + torch.matmul(D_fp.double(),
                                            torch.from_numpy(u_p[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1))
                                            )
                             )
        # bb1 = dt*theta*(f_f- bb_1.reshape(bb_1.shape[0],) - bb_2.reshape(bb_2.shape[0],))

        # bb1_m = np.loadtxt('bb_u_f_1_' + str(k + 1) + '.txt')
        # d_bb1 = np.max(np.abs(bb_1 - bb1_m))

        # matlab bb_2 = dt*(1-theta)*(f_f-M_fp*u_der_p(:,k)-D_fp*u_p(:,k))
        bb_2 = dt * (1.0 - theta) * (torch.matmul(torch.from_numpy(M_fp), torch.from_numpy(u_der_p[:, k]).reshape(u_der_p[:, k].shape[0], 1))
                                     + torch.matmul(D_fp.double(), torch.from_numpy(u_p[:, k]).reshape(u_p[:, k].shape[0], 1)))
        # bb_2_m = np.loadtxt('bb_u_f_2_' + str(k + 1) + '.txt')
        # d_bb2 = np.max(np.abs(bb_2 - bb_2_m))

        # bb_final_m = np.loadtxt('bb_final_' + str(k + 1) + '.txt')
        # d_bb_final = np.max(np.abs(bb_final_m))
        bb += bb_1.reshape(bb_1.shape[0]) + bb_2.reshape(bb_2.shape[0])
        # d_bb_final = np.max(np.abs(bb - bb_final_m))

        # bb += dt*(1-theta)*(f_f-np.matmul(M_fp,u_der_p[:,k].reshape(u_der_p[:,k].shape[0],1))
        #                     -np.matmul(D_fp,u_p[:,k].reshape(u_p[:,k].shape[0],1)))

        bb2_1 = np.matmul(M_fp, u_der_p[:, k].reshape(u_der_p[:, k].shape[0], 1))
        bb2_2 = torch.matmul(D_fp.double(), torch.from_numpy(u_der_p)[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1))
        bb2 = dt * (1 - theta) * (torch.from_numpy(f_f) - bb2_1.reshape(torch.from_numpy(bb2_1).shape[0]) - bb2_2.reshape(bb2_2.shape[0]))
        #
        # bb2_m = np.loadtxt('bb2_' + str(k + 1) + '.txt')
        # d_bb2 = np.max(np.abs(bb2 - bb2_m))

        bb += bb2

        tv = torch.linalg.solve(torch.from_numpy(M_ff) + dt * theta * D_ff, bb)
        # tv_m = np.loadtxt('time_vector_' + str(k + 1) + '.txt')
        # d_tv = np.max(np.abs(tv - tv_m))

        # print(k, d_tv)
        u_f[k + 1, :] = tv

    return el,time,u_f,u_p