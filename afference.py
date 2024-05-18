import numpy as np

def afference_matrix(n_el,dof_el):
# Afference matrix
   A=np.zeros((n_el,dof_el))
   for i in range(n_el):
       for j in range(dof_el):
           A[i][j] = i*(dof_el-1)+j

   return A


