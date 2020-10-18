import numpy as np
import helpers

def least_square_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        gradient, error = helpers.compute_gradient(y, tx, w)
        loss = helpers.compute_mse(error)
        w = w - gamma*gradient
    return w, loss

def least_square_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            gradient,error = helpers.compute_gradient(batch_y, batch_x, w)
            loss = helpers.compute_mse(error)
            w = w - gamma*gradient
    return w, loss

def least_square(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = helpers.compute_mse(y - (tx @ w))
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    ridge = (2*y.shape[0]*lambda_)*np.identity(tx.shape[1])
    gram = tx.T @ tx
    w = np.linalg.solve(gram + ridge, tx.T @ y)
    loss = helpers.compute_mse(y - (tx @ w))
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            w, loss = helpers.regression(batch_y, batch_x, w, gamma)
    return w, loss


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_,):
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            w, loss = helpers.regularized_regression(batch_y, batch_x, w, gamma, lambda_)
    return w, loss

