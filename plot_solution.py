import numpy as np
import matplotlib.pyplot as plt

def compare_plot(x_p,u_anal,n_el,el,x_i,x_f):
    plt.figure()
    # axes('FontSize',14)
    plt.plot(x_p ,u_anal[: ,0] ,label='Initial condition' ,color='green')



    xt = np.zeros(n_el)
    yt = np.zeros(n_el)
    for n in range(n_el):
        xt[n] = el[n].x[0]
        yt[n] = el[n].time[-1].u[0]
    plt.plot(xt ,yt ,'o', label='Numerical solution', color='blue')
    plt.plot(x_p ,u_anal[: ,-1] ,label='Analytical solution' ,color='red')
    plt.title('Analytical and numerical solution')
    plt.legend()
    plt.xlim(x_i ,x_f)
    plt.show(block=True)
    yt_m = np.loadtxt('yt_final.txt')
    d_yt = np.max(np.abs(yt -yt_m))
    return xt,yt,d_yt
