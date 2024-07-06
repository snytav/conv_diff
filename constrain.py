#function [A_ff,A_fp,A_pf,A_pp]
import numpy as np


def constrain_matrix(A,dof_constrained):
    # Constrain a matrix
    N=A.shape[0]
    p=dof_constrained
    f_aus=np.arange(0,N,1)
    p_aus=np.zeros(N)
    p_aus[p] = p
    f = f_aus - p_aus
    f = np.where(f != 0)
    f = f[0]

    A_ff_m = np.loadtxt('A_ff_cons_m.txt')
    A_ff_m = A_ff_m.reshape(f.shape[0], f.shape[0])
    from numpy import ix_
    A_ff = A[ix_(f,f)]
    d = np.max(np.abs(A_ff-A_ff_m))
    #A_ff = np.diag(A[f,f])

    p = np.array(p)
    # A_fp_m = np.loadtxt('A_fp_cons_m.txt')
    # A_fp_m = A_fp_m.reshape(f.shape[0],p.shape[0])

    A_fp = A[ix_(f, p)]
    # d_fp = np.max(np.abs(A_fp - A_fp_m))
    A_pf = A[ix_(p,f)]
    # A_pf_m = np.loadtxt('A_pf_cons_m.txt')
    # A_pf_m = A_pf_m.reshape(p.shape[0],f.shape[0])
    # d_pf = np.max(np.abs(A_pf - A_pf_m))
    A_pp = A[ix_(p,p)]
    # A_pp_m = np.loadtxt('A_pp_cons_m.txt')
    # A_pp_m = A_pp_m.reshape(p.shape[0],p.shape[0])
    # d_pp = np.max(np.abs(A_pp - A_pp_m))
    qq = 0

    return A_ff,A_fp,A_pf,A_pp