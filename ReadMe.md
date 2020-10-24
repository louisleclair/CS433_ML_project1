# EPFL Machine Learning Higgs 
__`Team`__: arbitrageurs

__`Team Members`__: Qais Humeid, Kim Ki Beom, Louis Leclair

To reproduce our results you must folllow these instructions step by step:

1. Be sure to have `Numpy` installed. This is the only external library we used.
2. Download the `train.csv` and `test.csv` and put them in the /data folder.
3. Be sure to have all auxiliary modules with the `run.py` file.
4. Once you download all the files, you need to go to the directory `src` where the script is. 
5. Run python script - `run.py`

## The structure of the project
```
├── ReadMe.md
├── data
│   ├── sample-submission.csv
│   ├── test.csv
│   └── train.csv
├── result
│   └── submission.csv
└── src
    ├── __init__.py
    ├── helpers.py
    ├── implementations.py
    ├── proj1_helpers.py
    ├── project1.ipynb
    └── run.py
```

## Mandatory modules

###`Implementations.py`

Contain the mandatory implementations of regression models for the project.

- __`least_square_GD`__: Linear regression using gradient descent
- __`least_square_SGD`__: Linear regression using stochastic gradient descent
- __`least_square`__: Least squares regression using normal equations
- __`ridge_regression`__: Ridge regression using normal equations
- __`logistic_regression`__: Logistic regression using SGD
- __`reg_logistic_regression`__: Regularized logistic regression using SGD

### `run.py` 
Script that generate the `submission.csv` file containing our predictions.

## Auxiliary modules

###`proj1_helpers.py`
Helper file given by the TAs. It contains:

- __`load_csv_data`__: help us to load the data from the csv file with the distinction between the outputs, the inputs and the ids of the data.
- __`predict_labels`__: given the training weights and the data give the predictions of our model.
- __`create_csv_submission`__: create the output file in csv format for the submission.

###`helpers.py`
File where we put all our auxiliary methods which help us during the project. We have different category inside this file. We have the methods for data processing, for the costs, for the cross validation and some tool functions.

#### `Data Processing Functions`
We have 2 auxiliary functions and 1 function which create our data in the wanted format.

- __`standardize`__: normalize the data to have a mean equals to 0 and a standard deviation equals to 0.
- __`remove_NA`__: remove features when the percentage of -999 is greater than a certain threshold.
- __`build_data`__: Create the data we are going to used in our model.

#### `Cost and gradient functions`
Contain 1 auxiliary function, 2 cost functions and 2 gradient functions.

- __`sigmoid`__: auxiliary function to help compute the logistic loss
- __`compute_mse`__: Compute mean square error.
- __`logistic_loss`__: Compute the loss for logistic regression.
- __`compute_gradient`__: Compute the gradient of linear regression and return it with the error.
- __`logistic_gradient`__: Compute the gradient for logistic regression.

#### `Cross Validation functions`
Helpers functions  __`build_k_indices`__, __`cross_validation`__. Which help us to find the best hyperparameter gamma and lambda and avoid overfitting.

#### `Tool functions`
We have __`batch_iter`__ for the creation of batch, the implementation come from the labs and __`pred_labels`__ which is our implementation to predict the labels.

### `Data File`
File where we have the __`test.csv`__, __`train.csv`__ and __`sample-submission.csv`__ which is the data we used for the training of our model and the testing.

### `Result File`
File created at the end of the __`run.py`__ where we put __`submission.csv`__ which is our prediction model.


