import numpy as np
import torch


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

def element_convection_matrix(a,dof_el,n_gauss,dN,W,w,J):
    # Element convection matrix
    C=np.zeros((dof_el,dof_el))

    for i in range(dof_el):
        for j in range(dof_el):
            for nn in range(n_gauss):
                C[i][j]=C[i][j]+(W[i,nn]*dN[j,nn])*w[nn]
          #  C[i][j]=a*C[i][j]
    #C.requires_grad = True
    C = a*torch.from_numpy(C)
    return C

def assemble_diffusion_matrix(el,dof,n_el,dof_el,A):

    # Assemblage of diffusion matrix
    K=np.zeros((dof,dof))

    for n in range(n_el):
        for i in range(dof_el):
            for j in range(dof_el):
                K[A[n][i]][A[n][j]]=K[A[n][i]][A[n][j]]+el[n].K[i,j]

    return K

def element_diffusion_matrix(v,dof_el,n_gauss,dN,dW,w,J):
# Element diffusion matrix
    K=np.zeros((dof_el,dof_el))
    for i in range(dof_el):
        for j in range(dof_el):
            for nn in range(n_gauss):
                K[i][j]=K[i][j]+dW[i][nn]*dN[j,nn]*w[nn]
            #K[i][j]=v*K[i][j]/J
    K = v*torch.from_numpy(K/J)
    return K

def element_load_vector(s,dof_el,n_gauss,N,W,w,J):
    # Elementload vector
    f = np.zeros(dof_el)
    for i in range(dof_el):
        for j in range(dof_el):
            for nn in range(n_gauss):
                f[i] = f[i] + (W[i][nn] * W[j][nn] * s[j] )* w[nn]


        f[i] = f[i] * J
    return f

