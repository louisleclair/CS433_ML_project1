import numpy as np
import helpers

def least_square_GD(y, tx, initial_w, max_iters, gamma):
    '''
    Compute the least square gradiend descend.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
        initial_w (Mx1 np.array of weights): weight vector of 0 for the initialization.
        max_iters: The number of iterations the gradient descent is going to be executed.
        gamma: Hyperparameter needed in the gradient descent.
    
    Returns:
        w: the last weight vector after a number of iterations.
        loss: the last loss after a number of iterations.
    '''
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        gradient, error = helpers.compute_gradient(y, tx, w)
        loss = helpers.compute_mse(error)
        w = w - gamma*gradient
    return w, loss

def least_square_SGD(y, tx, initial_w, max_iters, gamma):
    '''
    Compute the least square stochastic gradiend descend.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
        initial_w (Mx1 np.array of weights): weight vector of 0 for the initialization.
        max_iters: The number of iterations the gradient descent is going to be executed.
        gamma: Hyperparameter needed in the stochastic gradient descent.
    
    Returns:
        w: the last weight vector after a number of iterations.
        loss: the last loss after a number of iterations.
    '''
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            gradient,error = helpers.compute_gradient(batch_y, batch_x, w)
            loss = helpers.compute_mse(error)
            w = w - gamma*gradient
    return w, loss

def least_square(y, tx):
    '''
    Compute the least square.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
    
    Returns:
        w: compute with the close formula.
        loss: compute with the mean square error.
    '''
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = helpers.compute_mse(y - (tx @ w))
    return w, loss
    
def ridge_regression(y, tx, lambda_):
    '''
    Compute the ridge regression which the least square adding to it a regularized parameter lambda.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
        lambda_: Hyperparameter add to the closed formula to avoid overfitting.
    
    Returns:
        w: the last weight vector after a number of iterations.
        loss: the last loss after a number of iterations.
    '''
    ridge = (2*y.shape[0]*lambda_)*np.identity(tx.shape[1])
    gram = tx.T @ tx
    w = np.linalg.solve(gram + ridge, tx.T @ y)
    loss = helpers.compute_mse(y - (tx @ w))
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    '''
    Compute the logistic regression using stochastic gradient descent.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
        initial_w (Mx1 np.array of weights): weight vector of 0 for the initialization.
        max_iters: The number of iterations the gradient descent is going to be executed.
        gamma: Hyperparameter needed in the stochastic gradient descent part.
    
    Returns:
        w: the last weight vector after a number of iterations.
        loss: the last loss after a number of iterations.
    '''
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            grad = helpers.logistic_gradient(batch_y, batch_x, w)
            loss = helpers.logistic_loss(batch_y, batch_x, w)
            w = w - gamma*grad
    return w, loss

def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_,):
    '''
    Compute the regularized logistic regression using stochastic gradient descent.
    
    Parameters:
        y (Nx1 np.array of labels): known labels.
        tx (NxM np.array of data): training set of N datas with M features.
        initial_w (Mx1 np.array of weights): weight vector of 0 for the initialization.
        max_iters: The number of iterations the gradient descent is going to be executed.
        gamma: Hyperparameter needed in the gradient descent.
        lamda_: Hyperparameter needed to regularized the logistic regression.

    
    Returns:
        w: the last weight vector after a number of iterations.
        loss: the last loss after a number of iterations.
    '''
    w = initial_w
    for _ in range(max_iters):
        for batch_y, batch_x in helpers.batch_iter(y, tx):
            grad = helpers.logistic_gradient(batch_y, batch_x, w) + lambda_ * 2 * w
            loss = helpers.logistic_loss(batch_y, batch_x, w) + lambda_ * np.linalg.norm(w)
            w = w - gamma*grad
    return w, loss

