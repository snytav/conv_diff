import numpy as np

def assemble_load_vector(el,dof,n_el,dof_el,A):

    # Assemblage of load vector
    f=np.zeros(dof)
    for n in range(n_el):
        for i in range(dof_el):
            f[int(A[n,i])] += el[n].f[i]
    return f
