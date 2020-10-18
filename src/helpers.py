import numpy as np
import implementations
from proj1_helpers import *

def batch_iter(y, tx, batch_size=1, num_batches=1, shuffle=True):
    data_size = len(y)
    if shuffle: 
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def compute_gradient(y, tx, w):
    error = y - (tx @ w)
    gradient = (- 1/y.shape[0]) * tx.T @ (error)
    return gradient, error

def compute_mse(e):
    return 1/2 * np.mean(e ** 2)

def sigmoid(x):
    return 1/(1  + np.exp(-x))

def logistic_loss(y, tx, w):
    sig = sigmoid(tx @ w)
    loss = (y.T @ np.log(sig)) + ((1 - y)).T @ (np.log(1 - sig))
    return -loss

def logistic_gradient(y, tx, w):
    sig = sigmoid(tx @ w)
    grad = tx.T @ (sig - y)
    return grad

def regression(y, tx, w, gamma):
    grad = logistic_gradient(y, tx, w)
    loss = logistic_loss(y, tx, w)
    w = w - gamma*grad
    return w, loss

def regularized_regression(y, tx, w, gamma, lambda_):
    grad = logistic_gradient(y, tx, w) + lambda_ * 2 * w
    loss = logistic_loss(y, tx, w) + lambda_ * np.linalg.norm(w)
    w = w - gamma*grad
    return w, loss  

# ==============================================
# Data cleaning part
# ==============================================

def standardize(x):
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    return x / std_x

def remove_NA(tX, threshold):
    rows, cols = tX.shape
    bool_arr = (tX == -999)
    percent_missing = np.sum(bool_arr, axis=0)/rows
    return tX[:,(percent_missing <= threshold)]

def build_data(y, tX, threshold):
    pri_jet_num = tX[:,22]
    prn0 = (pri_jet_num == 0)
    prn1 = (pri_jet_num == 1)
    prn2 = (pri_jet_num == 2)
    prn3 = (pri_jet_num == 3)
    
    X0 = tX[prn0,:]
    X1 = tX[prn1,:]
    X2 = tX[prn2,:]
    X3 = tX[prn3,:]
    
    Y0 = y[prn0]
    Y1 = y[prn1]
    Y2 = y[prn2]
    Y3 = y[prn3]
    
    X0 = np.delete(X0, 22, 1)
    X0 = np.delete(X0, 28, 1)
    X1 = np.delete(X1, 22, 1)
    X2 = np.delete(X2, 22, 1)
    X3 = np.delete(X3, 22, 1)
    
    cX0 = remove_NA(X0, threshold)
    cX1 = remove_NA(X1, threshold)
    cX2 = remove_NA(X2, threshold)
    cX3 = remove_NA(X3, threshold)

    sX0 = standardize(cX0)
    sX1 = standardize(cX1)
    sX2 = standardize(cX2)
    sX3 = standardize(cX3)

    tx0 = np.c_[np.ones(sX0.shape[0]), sX0]
    tx1 = np.c_[np.ones(sX1.shape[0]), sX1]
    tx2 = np.c_[np.ones(sX2.shape[0]), sX2]
    tx3 = np.c_[np.ones(sX3.shape[0]), sX3]

    return ((tx0, tx1, tx2, tx3), (Y0, Y1, Y2, Y3))

# Basic data cleaning

def clean_tx(tx, threshold):
    row, col = tx.shape[0], tx.shape[1]
    empty = np.zeros(col)
    for i in range(col):
        count = np.count_nonzero(tx[:,i] == -999)
        empty[i] = count/row    
    
    return tx[:,(empty <= threshold)]

# ==============================================
# Cross Validation
# ==============================================

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, initial_w, max_iters, gamma, lambda_, k_indices, k):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    x_te = x[te_indice]
    x_tr = x[tr_indice]
    # ridge regression
    w_final, _ = implementations.reg_logistic_regression(y_tr, x_tr, initial_w, max_iters, gamma, lambda_)
    # calculate the loss for train and test data
    pred_tr = np.count_nonzero((y_tr == predict_labels(w_final, x_tr)) == 1)/len(y_tr)
    pred_te = np.count_nonzero((y_te == predict_labels(w_final, x_te)) == 1)/len(y_te)
    return pred_tr, pred_te

# ==============================================
# Cross Validation
# ==============================================

def pred_labels(all_tx_te, all_id_te, weights):
    pred_res = np.array([], dtype=int)
    all_ids = np.array([], dtype=int)

    for tx_te, weight, id_te in zip(all_tx_te, weights, all_id_te):
        pred = np.dot(tx_te, weight)
        pred[np.where(pred <= 0)] = -1
        pred[np.where(pred > 0)] = 1
        pred_res = np.append(pred_res, pred)
        all_ids = np.append(all_ids, id_te)

    rescale_factor = min(all_ids)
    all_ids = all_ids - rescale_factor

    result = np.zeros(len(all_ids))
    result[all_ids] = pred_res
    return result