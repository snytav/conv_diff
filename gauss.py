import numpy as np

def Gauss_parameters(n):
    # Parameters

    if n == 1: # Gauss with 1 nodes
       csi = np.zeros(1)
       w   = 2*np.ones(1)

    if n == 2:  # Gauss with 2 nodes
       csi = np.array([-1 / np.sqrt(3), +1 / np.sqrt(3)])
       w   = np.ones(2)

    if n == 3: # Gauss with 3 nodes
       csi = np.array([np.sqrt(3.0 / 5.0), 0,np.sqrt(3.0 / 5.0)])
       w = np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    if n == 4: # Gauss with 4 nodes
       csi = np.array([-1.0 / 35.0 * np.sqrt(525 + 70 * np.sqrt(30)),
                       -1.0 / 35.0 * np.sqrt(525 - 70 * np.sqrt(30)),
                       + 1.0 / 35.0 * np.sqrt(525 - 70 * np.sqrt(30)),
                       +1.0 / 35.0 * np.sqrt(525 + 70 * np.sqrt(30))])

       w = np.array([1.0 / 36.0 * (18 - np.sqrt(30)),
                     1.0 / 36.0 * (18 + np.sqrt(30)),
                     1.0 / 36.0 * (18 + np.sqrt(30)),
                     1.0 / 36.0 * (18 - np.sqrt(30))])

    return csi,w

def f_N(csi):
#   Shape functions
    N1=0.5*(np.ones_like(csi)-csi)
    N2=0.5*(np.ones_like(csi)+csi)
    N=np.array([N1.T,N2.T])
    return N

def f_dN(csi):
# 1st derivatives of shape functions
   dN1 = -0.5
   dN2 =  0.5
   dN=np.array([dN1,dN2])
   return dN


#function [N,dN]=\
def shape_functions_Gauss_points(csi):
# Computation of shape functions (and derivatives) at Gauss points
   n_gauss= csi.shape[0]
   N  = np.zeros((n_gauss,n_gauss))
   dN = np.zeros((n_gauss,n_gauss))
   for n in range(n_gauss):
       N[:,n]  = f_N(csi[n])
       dN[:,n] = f_dN(csi[n])
   return N,dN

def f_W(csi,beta):
  # Test functions
  W1=0.5*(np.ones_like(csi)-csi)-3.0/4.0*beta*(np.ones_like(csi)-np.power(csi,2))
  W2=0.5*(np.ones_like(csi)+csi)+3.0/4.0*beta*(np.ones_like(csi)-np.power(csi,2))
  W=np.array([W1.T,W2.T])
  return W

def f_dW(csi,beta):
# 1st derivatives of test functions
    dW1=-0.5+1.5*beta*csi
    dW2=+0.5-1.5*beta*csi
    dW=np.array([dW1,dW2])
    return dW


def test_functions_Gauss_points(csi,beta):
# Computation of shape functions (and derivatives) at Gauss points
   n_gauss= csi.shape[0]
   N  = np.zeros((n_gauss,n_gauss))
   dN = np.zeros((n_gauss,n_gauss))
   for n in range(n_gauss):
       N[:,n]  = f_W(csi[n],beta)
       dN[:,n] = f_dW(csi[n],beta)
   return N,dN



