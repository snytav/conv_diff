import numpy as np

def get_analytical_solution(n_el,T,u_max,l,x_p,x_0,a,v):
    u_anal = np.zeros((n_el, T.shape[0]))

    d_u_anal_m = np.zeros(T.shape[0])
    for k in range(T.shape[0]):
        t = T[k]
        alfa = np.sqrt(1 + 4 * v * t / l ** 2)
        u_anal[:, k] = u_max / alfa * np.exp(-np.power(((x_p - x_0 - a * t) / (l * alfa)), 2))
        # u_anal_m = np.loadtxt('u_anal_k_' + str(k + 1) + '.txt')
        # d_u_anal_m[k] = np.max(np.abs(u_anal_m[:u_anal[:, k].shape[0]] - u_anal[:, k]))
        qq = 0
    # d_u_anal = np.max(d_u_anal_m)
    return u_anal #,d_u_anal
