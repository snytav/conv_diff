import numpy as np


def element_mass_matrix(dof_el,n_gauss,N,W,w,J):
    # Element mass matrix
    M=np.zeros((dof_el,dof_el))
    for i in range(dof_el):
        for j in range(dof_el):
            for nn in range(n_gauss):
                M[i][j]=M[i][j]+(W[i][nn]*N[j][nn])*w[nn]
                qq = 0
                #M(i, j) = M(i, j) + (W(i, nn) * N(j, nn)) * w(nn);

            M[i][j]=M[i][j]*J
    return M


