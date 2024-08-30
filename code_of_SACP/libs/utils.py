import os
import numpy as np
import torch
from sklearn.decomposition import PCA


def set_seed(seed):
    if seed != 0:
        np.random.seed(2)
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
    return seed

def indexToAssignment(index_, pad_length, Row, Col):
    new_assign = {}
    for counter, value in enumerate(index_):
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign

def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index

def selectNeighboringPatch(matrix, ex_len, pos_row, pos_col):
    selected_rows = matrix[:,range(pos_row-ex_len,pos_row+ex_len+1), :]
    selected_patch = selected_rows[:, :, range(pos_col-ex_len, pos_col+ex_len+1)]
    return selected_patch

def zeroPadding_3D(old_matrix, pad_length, pad_depth = 0):
    new_matrix = np.lib.pad(old_matrix, ((pad_depth, pad_depth), (pad_length, pad_length), (pad_length, pad_length)), 'constant', constant_values=0)
    return new_matrix

def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def get_device(model=None):
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device



