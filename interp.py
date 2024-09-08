import numpy as np
from gauss import f_N

def interpolation(n,u,A,n_e):
    # Interpolation of scalar variable

    u_e = np.zeros(n_e)
    csi=np.linspace(-1,+1,n_e)
    for i in range(n_e):
        Ni=f_N(csi[i])
        un=u[A[n,:].astype(int)]
        u_e[i]=np.inner(Ni,un)
    return u_e


