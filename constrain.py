#function [A_ff,A_fp,A_pf,A_pp]
import numpy as np

def matrix_by_indices(A,f,p):
    row, col = np.indices((f.shape[0], p.shape[0]))  # indices for rows and columns
    c = np.indices(p.shape)  # indices of all columns in row array
    row[:, c] = f  # assigning f to all the columns of 'row'
    col = row.T  # we want all the columns and rows with indices from 'f'
    A_ff = A[row, col]  # getting all the specified elements from 'A'
    return A_ff

def constrain_matrix(A,dof_constrained):
    from numpy import ix_
    # Constrain a matrix
    N=A.shape[0]
    p=dof_constrained
    f_aus=np.arange(0,N,1)
    p_aus=np.zeros(N)
    p_aus[p] = p
    f = f_aus - p_aus
    f = np.where(f != 0)
    f = f[0]
    p = np.array(p)


    A_ff = A[ ix_( f, f ) ]
    #A_ff = np.diag(A[f,f])
    A_fp = A[ ix_( f, p ) ]

    A_pf = A[ ix_( p, f ) ]
    A_pp = A[ ix_( p, p ) ]
    return A_ff,A_fp,A_pf,A_pp


import numpy as np


def constrain_vector(v, dof_constrained):
    from numpy import ix_
    # Constrain a vector
    N = len(v)
    p = dof_constrained
    f_aus = np.arange(0, N)
    p_aus = np.zeros(N, dtype=int)
    p_aus[ix_(p)] = p
    f = f_aus - p_aus
    f = np.nonzero(f)[0]

    v_f = v[ix_(f)]
    v_p = v[ix_(p)]

    return v_f, v_p