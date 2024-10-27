import numpy as np

def interpolation(T,time,u_f,u_p,dof_constrained,n_el,dof_el,n_e,el):
    from data_all import data_all_dof
    # Data for all dof
    dt_k = np.zeros(T.shape[0])
    for k in range(T.shape[0]):
        time[k].u = data_all_dof(u_f[k, :].T, u_p[:, k], dof_constrained)
        time_k_u_m = np.loadtxt('time_k_u_' + str(k + 1) + '.txt')
        dt_k[k] = np.max(np.abs(time_k_u_m - time[k].u))

    # Interpolation of the solution OK
    from afference import afference_matrix
    A = afference_matrix(n_el, dof_el)
    from interp import interpolation
    A_m = np.loadtxt('A_interp.txt')
    q = A - A_m + np.ones_like(A)
    d_A_m = np.max(q)
    interp_nk = np.zeros((n_el, T.shape[0]))
    for n in range(n_el):
        for k in range(T.shape[0]):
            el[n].time[k].u = interpolation(n, time[k].u, A, n_e)
            time_k_u_m = np.loadtxt('interp_n_' + str(n + 1) + '_k_' + str(k + 1) + '.txt')
            interp_nk[n][k] = np.max(np.abs(time_k_u_m - el[n].time[k].u))

    d_interp_nk = np.max(interp_nk)
    return d_interp_nk