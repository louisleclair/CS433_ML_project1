import numpy as np
import implementations
import helpers
import proj1_helpers
import os

# Loading the data

DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
y, tX, ids = proj1_helpers.load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

# Preprocessing of the data: cleaning and standarization 

threshold = 0.15
all_tx, all_y = helpers.build_data(y, tX, threshold)

# Processing of the data with our model
# initialisation of the hyperparameter, best values found.
gamma = 1e-5
max_iters = 300

# Try to find the best values for w

all_w = []
all_pred = []
for tx, y_ in zip(all_tx, all_y):
    initial_w = np.zeros(tx.shape[1])
    w_final, _ = implementations.logistic_regression(y_, tx, initial_w, max_iters, gamma)
    all_pred.append(np.count_nonzero((y_ == proj1_helpers.predict_labels(w_final, tx)) == 1)/len(y_))
    all_w.append(w_final)
    
weights = all_w

# After found the best w, we use the test set to predict the y

DATA_TEST_PATH = '../data/test.csv'  
_, tX_test, ids_test = proj1_helpers.load_csv_data(DATA_TEST_PATH)
all_tx_te, all_id_te = helpers.build_data(ids_test, tX_test, 0.15)
y_pred = helpers.pred_labels(all_tx_te, all_id_te, weights)

# Output our prediction 

dir = os.path.join("..", "result")
if not os.path.exists(dir):
    os.mkdir(dir)
OUTPUT_PATH = '../result/submission.csv' 
proj1_helpers.create_csv_submission(ids_test, y_pred, OUTPUT_PATH)