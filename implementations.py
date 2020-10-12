import numpy as np

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    error = y - (tx @ w)
    gradient = - 1/y.shape[0] * tx.T @ (error)
    return gradient, error

def compute_mse(e):
    return 1/2 * np.mean(e ** 2)

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



def least_square_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        gradient, error = compute_gradient(y, tx, w)
        loss = compute_mse(error)
        w = w - gamma*gradient
    return w, loss

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    loss = 0
    for _ in range(max_iters):
        for batch_y, batch_x in batch_iter(y, tx):
            gradient,_ = compute_gradient(batch_y, batch_x, w)
            w = w - gamma*gradient
            loss = compute_mse(y - (batch_x @ w))
    return w, loss

def least_square(y, tx):
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    e = y - (tx @ w)
    return w, e
    
def ridge_regression(y, tx, lambda_):
    ridge = (2*y.shape[0]*lambda_)*np.identity(tx.shape[1])
    gram = tx.T @ tx
    w = np.linalg.solve(gram + ridge, tx.T @ y)
    loss = y - (tx @ w)
    return w, loss