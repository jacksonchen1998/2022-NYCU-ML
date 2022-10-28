# %%
import numpy as np
import pandas as pd

# %% [markdown]
# <h2> Basic function </h2>

# %%
# Split data into train_data and valid_data
def split_data(x, y, vaild_ratio=0.2, random_state=None):
    data_size = x.shape[0]
    split_id = int(data_size * vaild_ratio)

    index = np.arange(data_size)
    np.random.shuffle(index)
    x = x[index]
    y = y[index]

    if random_state is not None:
        np.random.seed(random_state)

    train_x, valid_x = x[split_id:], x[:split_id]
    train_y, valid_y = y[split_id:], y[:split_id]

    return train_x, valid_x, train_y, valid_y

def rmse(true_y, pred_y):
        return np.sqrt(np.mean((true_y - pred_y)**2))

def fac(n):
    if (n == 0 or n == 1):
        return np.array([1])
    else:
        return np.array([n * fac(n-1)])
        
def product(*args, repeat=1): # Cartesian product
        pools = [tuple(pool) for pool in args] * repeat
        result = [[]]
        for pool in pools:
            result = [x+[y] for x in result for y in pool]
        for prod in result:
            yield tuple(prod)

def replacement(iterable, r): # when n > 0, return the number of (n + r - 1)! / r! / (n - 1)!
        pool = tuple(iterable)
        n = len(pool)
        for indices in product(range(n), repeat=r):
            if sorted(indices) == list(indices):
                yield tuple(pool[i] for i in indices)

def standard_norm(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    return (x - mean) / std

# %% [markdown]
# <h1>2 Linear Regression</h1>
# <h3>2.1 Feature Selection</h3>
# In real-world applications, the dimension of data is usually more than one. In the training
# stage, please fit the data by applying a polynomial function of the form
# 
# $$
#     y = (\textbf{x}, \textbf{w}) = w_0 + \sum_{i=1}^D w_i x_i + \sum_{i=1}^D\sum_{j=1}^D w_ix_ix_j \ (M = 2)
# $$
# 
# and minimizing the error function
# 
# $$
#     E(\textbf{w}) = \sqrt{\frac{1}{N} \sum_{n=1}^N \{y(x_n, \textbf{w}) - t_n\}^2}
# $$

# %% [markdown]
# (a) In the feature selection stage, please apply polynomials of order $M = 1$ and $M = 2$
# 
# over the input data with dimension $D = 11$. Please evaluate the corresponding RMS error on the training set and valid set.

# %% [markdown]
# Root Mean Square Error
# <p align="center">
#     <img src="image/rmse.png"/>
# </p>
# Least Square Solution
# <p align="center">
#     <img src="image/lsq.jpg"/>
# </p>

# %%
# Implement details from sklearn LinearRegression model
class LinearRegression:
    # hyper is for regularization
    def __init__(self, learning_rate=0.0001, iter=1000, hyper=0):
        self.hyper = hyper
        self.learning_rate = learning_rate
        self.iter = iter
        self.random_rate = 1e-7
        self.weight = None
        self.bias = None
        
    def fit(self, x, y):
        self.data_size, self.feature_number = x.shape

        # Using least square solution to find the best weight and bias
        # We need to add a matrix to prevent generate singular matrix, making it can be inversed correctly
        # If we use hyper parameter, it will turn into be MAP, otherwise is MLE
        bias_inputs = np.ones((self.data_size, 1))
        phi = np.concatenate((bias_inputs, x), axis=1)
        noise = self.hyper * np.eye(self.feature_number+1)
        normal = np.dot(phi.T, phi) + self.random_rate * np.random.randn(self.feature_number+1)
        inverse_equ = np.linalg.inv(normal + noise)
        theta = np.dot(inverse_equ, np.dot(phi.T, y))
        self.bias, self.weight = theta[0], theta[1:]

    def predict(self, x):
        return np.dot(x, self.weight) + self.bias

# %%
data_x_df = pd.read_csv('X.csv')
data_t_df = pd.read_csv('T.csv')

train_x, valid_x, train_y, valid_y = split_data(data_x_df.values, data_t_df.values, vaild_ratio=0.2, random_state=1)

# %%
model = LinearRegression()
model.fit(train_x, train_y)

# RMS error
print("M = 1")
predict_y = model.predict(train_x)
print('Train RMS error: {}'.format(rmse(train_y, predict_y)))
baseline = rmse(train_y, predict_y)
predict_y2 = model.predict(valid_x)
print('Test RMS error: {}'.format(rmse(valid_y, predict_y2)))

# %% [markdown]
# <p align="center">
#     <img src="image/Output_5.png"/>
# </p>

# %%
# We can use polynomial features to define polynomial function
# It's implemented from sklearn PolynomialFeatures
class PolynomialFeatures:

    def __init__(self, x, degree=2):
        self.data = x
        self.degree = degree
        self.data_size = x.shape[0]
        self.feature_num = x.shape[1]
        self.numerator = fac(self.feature_num + self.degree) # (11 + 2)!
        self.denominator = fac(self.degree) * fac(self.feature_num) # 11! * 2!

    def fit(self):
        # calculate number of output feature size
        self.n_output_features = int(self.numerator / self.denominator) - 1 # (13 * 12 / 2) - 1 = 77
    
    def transform(self):
        # Transform data to polynomial features.
        # feature_tuple is list of tuples indices to calculate polynomial features
        # Create list of tuples containing feature index combinations.
        # to store new array from transformation
        feature_tuple = [replacement(range(self.feature_num), idx)
                for idx in range(1, self.degree+1)]
        combinations = [item for sublist in feature_tuple for item in sublist]
        x_new = np.empty((self.data_size, self.n_output_features))

        for i, index_feature_tuple in enumerate(combinations):
            x_new[:, i] = np.prod(self.data[:, index_feature_tuple], axis=1)

        return x_new
    
    
    def predict(self, x):
        return np.dot(x, self.weight) + self.bias

# %%
transform = PolynomialFeatures(data_x_df.values, degree=2)
transform.fit()
x_2 = transform.transform()
train_x, valid_x, train_y, valid_y = split_data(x_2, data_t_df.values, vaild_ratio=0.2, random_state=1)

model2 = LinearRegression()
model2.fit(train_x, train_y)

# RMS error
print("M = 2")
y_pred = model2.predict(train_x)
print('Train RMS error: {}'.format(rmse(train_y, y_pred)))
y_pred = model2.predict(valid_x)
print('Test RMS error: {}'.format(rmse(valid_y, y_pred)))

# %% [markdown]
# <p align="center">
#     <img src="image/Output_7.png"/>
# </p>

# %% [markdown]
# (b) How will you analysis the weights of polynomial model $M = 1$ and select the most contributive feature?

# %%
row = 0
for i in data_x_df.columns:
    print(i, end=": ")
    for j in model.weight[row:row+1]:
        print(j)
    row += 1

# %% [markdown]
# <p align="center">
#     <img src="image/Output_8.png"/>
# </p>

# %% [markdown]
# Correlation for each feature in x data

# %%
arr_x = np.array(data_x_df)
arr_y = np.array(data_t_df)
for i in range(11):
    print(data_x_df.columns[i])
    print(np.corrcoef(arr_x[:, i], arr_y[:,0]))

# %% [markdown]
# <p align="center">
#     <img src="image/Output_9.png"/>
# </p>

# %% [markdown]
# <style>
# 
# .red{
#     color: red;
# };
# 
# </style>
# 
# When M = 1, the most positive value of the coefficient is sulphates.
# 
# But as we use corrcoef function, the most contributive is **alcohol**.
# 
# <span class="red">So I think, **alcohol** is the most contributive feature.</span>

# %% [markdown]
# <h3>2.2 Maximum likelihood approach</h3>
# 
# (a) Which basis function will you use to further improve your regression model, polynomial, Gaussian, Sigmoid, or hybrid?

# %% [markdown]
# There are three choices.
# 
# Firstly, **polynomial basis** $y(x,w) = \sum_{j=0}^{M-1} w_j \phi_j(x) = w^T \phi(x)$
# 
# If we choose it, we may need to face some problems
# 
# 1. It's diccicult to formulate
# 2. It needs to use different polynomials in each region
# 3. Polynomials are global basis functions, each affecting the prediction over the whole input space
# 
# For the **Gaussian Radial Basis Functions** $\phi_j (x) = \rm{exp}(\frac{(x - \mu)^2}{2 \sigma^2})$
# 
# - $\mu$ govern the locations of the basis functions
# - $\sigma$ governs the spatial scale
# - [Demo graph from desmos](https://www.desmos.com/calculator/uc1hqmb09u?lang=zh-TW)
# 
# The last one, Sigmoidal Basis Function $\phi_j (x) = \sigma(\frac{x -\mu_j}{s})$ where $\sigma(a) = \frac{1}{1 + \rm{exp}(-a)}$

# %% [markdown]
# <style>
# 
# .red{
#     color: red;
# };
# 
# </style>
# 
# After above discussion, <span class="red">I will choose to use sigmoid basis function.</style>
# 
# Since it can be combined to create a model called **Artifical Neueal Network (ANN)**
# 
# In this situation, we will normalize input data, and than transfer it with sigmoid basis function.
# 
# Such that we can use the preprocessing data to analysis with Linear Regression model.

# %% [markdown]
# (b) Introduce the basis function you just decided in (a) to linear regression model and analyze the result you get.
# 
# $$
#     \phi(x) = [\phi_1(x), \phi_2(x), ..., \phi_N(x), \phi_{bias}(x)]
# $$
# <p align="center">
#     <img src="image/2.2.a.png"/>
# </p>
# 
# <p align="center">
#     <img src="image/standard.jpg"/>   
# </p>
# 

# %%
def sigmoid(x):
    return 1 / (1 + np.exp(- (x - np.mean(x, axis=0)) / np.std(x, axis=0)))

norm_x = standard_norm(data_x_df.values)

basis_x = sigmoid(norm_x)

train_x, valid_x, train_y, valid_y = split_data(basis_x, data_t_df.values, vaild_ratio=0.2, random_state=1)

model_M = LinearRegression()
model_M.fit(train_x, train_y)

# RMS error
train_pred = model_M.predict(train_x)
test_pred = model_M.predict(valid_x)

print('Train RMS error: {}, Test RMS error: {}'.format(rmse(train_y, train_pred), rmse(valid_y, test_pred)))

# %% [markdown]
# <p align="center">
#     <img src="image/Output_10.png"/>
# </p>

# %% [markdown]
# (c) Apply N -fold cross-validation in your training stage to select at least one hyperparameter (order, parameter number, . . .) 
# 
# for model and do some discussion (underfitting, overfitting).

# %%
# Implement details from sklearn model KFold
class KFold:
    def __init__(self, n_splits=3):
        self.now = 0
        self.one = 1
        
        if n_splits <= 1:
            raise ValueError(
                "k-fold cross-validation requires at least one"
                " train/test split by setting n_splits=2 or more,"
                " got n_splits={0}.".format(n_splits)
            )

        self.n_splits = n_splits

    def split(self, X):
        self.indices = np.arange(len(X))
        train_list = []
        test_list = []

        np.random.RandomState(self.one).shuffle(self.indices)

        for id in self.mask(X):
            train_list.append(self.indices[np.logical_not(id)])
            test_list.append(self.indices[(id)])

        return train_list, test_list
        
    def mask(self, X): # Generates boolean masks corresponding to test sets.
        mask_list = []
        for id in self.indice(X): 
            mask = np.zeros(len(X), dtype=bool)
            mask[id] = True
            mask_list.append(mask)

        return mask_list
    
    def indice(self, X): # Generates integer indices corresponding to test sets.
        n_splits = self.n_splits
        fold_sizes = np.full(n_splits, len(X)//n_splits, dtype=int)
        fold_sizes[:len(X) % n_splits] += 1
        index = []
        for fold_size in fold_sizes:
            start, stop = self.now, self.now + fold_size
            index.append(self.indices[start:stop])
            self.now = stop
        
        return index

kf = KFold(n_splits=3)
training_indices, indice = kf.split(basis_x)

train_rmse = np.zeros(len(data_t_df))
test_rmse = np.zeros(len(data_t_df))

# %%
for m in range(1, 7):
    train_rmse = np.zeros(len(data_t_df))
    test_rmse = np.zeros(len(data_t_df))
    
    for train_idx, index in zip(training_indices, indice):
        transform = PolynomialFeatures(basis_x, degree=m)
        transform.fit()
        polynomial_x = transform.transform()
        train_x, test_x = polynomial_x[train_idx], polynomial_x[index]
        train_y, test_y = data_t_df.values[train_idx], data_t_df.values[index]

        model_polynomial = LinearRegression()
        model_polynomial.fit(train_x, train_y)

        # RMS error
        train_pred = np.array(model_polynomial.predict(train_x))
        test_pred = np.array(model_polynomial.predict(test_x))
        train_rmse = np.append(train_rmse, rmse(train_y, train_pred))
        test_rmse = np.append(test_rmse, rmse(test_y, test_pred))
    
    average_train_rmse = sum(train_rmse) / len(training_indices)
    average_test_rmse = sum(test_rmse) / len(training_indices)
    print('M = {}, Train RMS error: {}, Test RMS error: {}'.format(m, average_train_rmse, average_test_rmse))

# %% [markdown]
# <p align="center">
#     <img src="image/Output_11.png"/>
# </p>

# %% [markdown]
# <h3> 2.3 Maximum a posteriori approach </h3>
# 
# <p align="center">
#     <img src="image/2.3.png"/>
# </p>
# 
# (a) What is the key difference between maximum likelihood approach and maximum a posteriori approach?

# %% [markdown]
# Maximum posteriori approach (MAP) affect from priori proability $p (\theta \ | \ m)$, 
# 
# but maximum likelihood approach (MLE) doesn't affect by it.
# 
# Priori proability, which means the probability based on past experience and analysis.
# 
# MLE, comparing with MAP. It's much easier to get overfit. 
# 
# Because MAP will use priori proability's and experimential data's info. to predict the testing data.
# 
# The influence determines by their weight, which is set by model designer.
# 
# ---
# 
# MAP: 
# 
# $$ \ p(\theta \ | \ \mathcal{D}, m) = \frac{p(\mathcal{D} \ | \ m, \theta) p(\theta \ | \ m)}{p(\mathcal{D} \ | \ m)} $$
# 
# MLE:
# 
# $$p(\mathcal{D} \ | \ m, \theta)$$

# %% [markdown]
# (b) Use maximum a posteriori approach method to retest the model in 2.2 you designed.
# 
# You could choose Gaussian distribution as a prior.

# %%
for m in range(1, 7):
    transform = PolynomialFeatures(basis_x, degree=m)
    transform.fit()
    polynomial_x = transform.transform()
    train_x, test_x = polynomial_x[train_idx], polynomial_x[index]
    train_y, test_y = data_t_df.values[train_idx], data_t_df.values[index]

    model = LinearRegression()
    model.fit(train_x, train_y)
    model_regularization = LinearRegression(hyper=0.01)
    model_regularization.fit(train_x, train_y)

    # RMS error
    train_pred = model.predict(train_x)
    test_pred = model.predict(test_x)
    trainx_regular = model_regularization.predict(train_x)
    testx_regular = model_regularization.predict(test_x)
    
    print("If we don't use priori probability: ")
    print('M = {}, Train RMS error: {}, Test RMS error: {}'.format(m, rmse(train_y, train_pred), rmse(test_y, test_pred)))
    print("If we use priori probability: ")
    print('M = {}, Train RMS error: {}, Test RMS error: {}'.format(m, rmse(train_y, trainx_regular), rmse(test_y, testx_regular)))
    print()

# %% [markdown]
# <p align="center">
#     <img src="image/Output_12.png"/>
# </p>

# %%
row = 0
for i in data_x_df.columns:
    print(i, end=": ")
    for j in model2.weight[row:row+1]:
        print(j)
    row += 1

# %% [markdown]
# <p align="center">
#     <img src="image/Output_13.png"/>
# </p>

# %% [markdown]
# (c) Compare the result between maximum likelihood approach and maximum a posteriori approach. 
# 
# Is it consistent with your conclusion in (a)?

# %% [markdown]
# Yes, it's consistent.
# 
# As we can see the result from previous code's output.
# 
# **Using MAP can effictively prevent overfitting.** 


