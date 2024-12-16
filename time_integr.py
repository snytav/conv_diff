import numpy as np
import torch
from sympy import *

def get_bb_1(M_ff,dt,theta,D_ff,u_f,m_fp,u_der_p,u_p,M_fp,D_fp,k):
    M_ff = torch.from_numpy(M_ff)
    M_ff.requires_grad=True
    M2   = - dt * (1 - theta) * D_ff.double()
    column = u_f[k, :]
    #column.requires_grad=True
    bb = torch.matmul(M_ff+M2, column)
    lf = torch.max(torch.abs(M_ff))
    lf.backward(retain_graph=True)
    lf2 = torch.max(torch.abs(M2))
    lf2.backward(retain_graph=True)

    lf3 = torch.max(torch.abs(column))
    lf3.backward(retain_graph=True)

    bb_1 = dt * theta * (torch.matmul(torch.from_numpy(M_fp),
                                      torch.from_numpy(u_der_p[:, k + 1]).reshape(u_der_p[:, k + 1].shape[0], 1))
                         + torch.matmul(D_fp.double(),
                                        torch.from_numpy(u_p[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1))
                                        )
                         )
    bb_2 = dt * (1.0 - theta) * (
                torch.matmul(torch.from_numpy(M_fp), torch.from_numpy(u_der_p[:, k]).reshape(u_der_p[:, k].shape[0], 1))
                + torch.matmul(D_fp.double(), torch.from_numpy(u_p[:, k]).reshape(u_p[:, k].shape[0], 1)))
    bb += bb_1.reshape(bb_1.shape[0]) + bb_2.reshape(bb_2.shape[0])
    return bb

def prepare_bb(M_fp,u_der_p,f_f,u_p,D_fp,k):
    A = M_fp
    b = u_der_p[:, k + 1]
    x = np.matmul(A, b)
    br = f_f - x
    upk = u_p[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1)
    res = torch.matmul(D_fp.double(), torch.from_numpy(upk))
    br = torch.from_numpy(br)
    br -= res.reshape(res.shape[0])  # *u_p[:,k+1]
    return


def get_bb2(M_fp,u_der_p,k,D_fp,u_p,theta,f_f,dt):
        bb2_1 = torch.matmul(M_fp, u_der_p[:, k].reshape(u_der_p[:, k].shape[0], 1))
        bb2_2 = torch.matmul(D_fp.double(), u_der_p[:, k + 1].reshape(u_p[:, k + 1].shape[0], 1))
        bb2 = dt * (1 - theta) * (torch.from_numpy(f_f) - bb2_1.reshape(bb2_1.shape[0]) - bb2_2.reshape(bb2_2.shape[0]))
        return bb2


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

    # Unsteady convection-diffusion-reaction solution
    u_f = torch.zeros((T.shape[0] + 1, u_0_f.shape[0]))
    # Time integration
    u_f[0, :] = torch.from_numpy(u_0_f)
    u_f.requires_grad = True
    u_f = u_f.double()
    for k, t in enumerate(T):
        MM = (torch.from_numpy(M_ff) + dt * theta * D_ff)

        # the operation M_fp*u_der_p[:,k+1] probably needs matrix multiplication
        #prepare_bb(M_fp, u_der_p, f_f, u_p, D_fp, k)
        # matlab dimensionality is (149,2) X( 2,1) resulting in 149,1
        bb = get_bb_1(M_ff, dt, theta, D_ff, u_f, M_fp, u_der_p, u_p, M_fp, D_fp,k)
        lf = torch.max(torch.abs(bb))
        lf.backward(retain_graph=True)
        print(k,v_arr.grad)




        #!!!!!!!!!!!!!!!!!

        u_der_p = torch.from_numpy(u_der_p)
        u_der_p.requires_grad = True
        M_fp = torch.from_numpy(M_fp)
        M_fp.requires_grad = True
        bb += get_bb2(M_fp,u_der_p,k,D_fp,u_p,theta,f_f,dt)


        tv = torch.linalg.solve(torch.from_numpy(M_ff) + dt * theta * D_ff, bb)
        u_f[k + 1, :] = tv
        lb = torch.max(torch.abs(u_der_p)) #+torch.max(torch.abs(D_fp))+torch.max(torch.abs(u_p))
        lb.backward(retain_graph=True)
        lf = torch.max(torch.abs(tv))
        lf.backward(retain_graph=True)
        print(k,v_arr.grad)

    return el,time,u_f,u_p