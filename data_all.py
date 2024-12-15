import numpy as np
import torch

def data_all_dof(u_f, u_p, dof_constrained):
    #u_f = u_f.detach().numpy()
    # Data for all DOF
    N = len(u_f) + len(u_p)
    u = np.zeros(N)

    p = dof_constrained
    f_aus = np.arange(N)
    p_aus = np.zeros(N)
    p_aus[p] = p
    f = f_aus - p_aus
    f = np.where(f != 0)
    f=  f[0]

    u[f.astype(int)] = u_f.detach().numpy()
    u[p] = u_p
    upt = torch.from_numpy(u_p)
    ut = torch.cat((upt[0].reshape(1), u_f, upt[1].reshape(1)))
    return ut
