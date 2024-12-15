import numpy as np

def data_all_dof(u_f, u_p, dof_constrained):
    # Data for all DOF
    u_f = u_f.detach().numpy()
    N = len(u_f) + len(u_p)
    u = np.zeros(N)

    p = dof_constrained
    f_aus = np.arange(N)
    p_aus = np.zeros(N)
    p_aus[p] = p
    f = f_aus - p_aus
    f = np.where(f != 0)
    f=  f[0]

    u[f.astype(int)] = u_f
    u[p] = u_p
    return u
