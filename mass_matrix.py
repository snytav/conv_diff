import numpy as np


def assemble_mass_matrix(el,dof,n_el,dof_el,A):

# Assemblage of mass matrix
    M=np.zeros((dof,dof))
    for n in range(n_el):
        for i in range(dof_el):
            for j in range(dof_el):
                M[int(A[n][i])][int(A[n][j])] += el[n].M[i][j]

    return M
            
        
    



