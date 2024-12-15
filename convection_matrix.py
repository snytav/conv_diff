import numpy as np
import torch

def assemble_convection_matrix(el,dof,n_el,dof_el,A):

    # Assemblage of convection matrix
    C=torch.zeros((dof,dof))
    for n in range(n_el):
        for i in range(dof_el):
            for j in range(dof_el):
                C[int(A[n][i])][int(A[n][j])] += el[n].C[i][j]

    return C