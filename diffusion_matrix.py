import numpy as np
import torch

def assemble_diffusion_matrix(el,dof,n_el,dof_el,A):

    # Assemblage of diffusion matrix
    K = torch.zeros((dof,dof))
    for n in range(n_el):
        for i in range(dof_el):
            for j in range(dof_el):
                K[int(A[n][i])][int(A[n][j])] += el[n].K[i][j]
    return K
