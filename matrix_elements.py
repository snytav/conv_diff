import numpy as np


def matrix_elements_by_list(A,f1,f2):
    row, col = np.indices((f1.shape[0], f2.shape[0]))  # indices for rows and columns
    c = np.indices(f1.shape)  # indices of all columns in row array
    row[:, c] = f1  # assigning f to all the columns of 'row'
    col = row.T  # we want all the columns and rows with indices from 'f'
    A_ff = A[row, col]  # getting all the specified elements from 'A'
    return A_ff