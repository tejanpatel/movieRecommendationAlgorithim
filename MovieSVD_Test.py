import numpy as np
import pandas as pd

##### Movie Recommendations - Remove 10% of data to test accuracy of the method

# First, load in the data
data = pd.read_csv('cleaned_data.csv')

# Save the original data. 
data_orig = np.copy(data.to_numpy()) # Save as a numpy array because we will
                                    # use that later.
data_orig = np.array(data_orig[:, 1:], dtype=float) # remove student numbers

# Find all entries which are NOT nan, those that people rated
rating_ind = np.where(~np.isnan(data_orig.ravel(order='F')))[0]

# Create a new data array, which we will use for the algorithm
data_py = np.copy(data_orig)

## Set a random sample of these rated entries to NaN: we will try to predict
## those values
# First determine how many we want to set equal to Nan. We will do 10%
n_test = int(np.floor(len(rating_ind)/10))

# Set the seed for the RNG
np.random.seed(12)

index_nan = np.random.choice(rating_ind, n_test, replace=False)
# Set those random entries to nan
data_py.ravel(order='F')[index_nan] = np.nan

## Shift the data so each row has zero mean and fill in missing with zeros
avg_user_ratings = np.nanmean(data_py, axis=1)
data_py -= avg_user_ratings.reshape(-1, 1)

# Now we will run the algorithm 
r = 1 # Set the rank to 1, we will do rank-1 updates
tol = 1e-8 # We will stop when our successive approximations are this close

Ak = np.nan_to_num(data_py) # Initial guess
for k in range(100000):
    U, S, Vt = np.linalg.svd(Ak, full_matrices = False)
    S = np.diag(S)
    Akplus1 = (U[:, :r]@S[:r, :r])@Vt[:r, :] # Redefine Akplus1
    # Replace the values in the low-rank approximation with the values we know
    Akplus1[~np.isnan(data_py)] = data_py[~np.isnan(data_py)]
    if np.linalg.norm(Ak - Akplus1)<tol:
         break
    Ak = Akplus1


A_final = Akplus1 + avg_user_ratings.reshape(-1,1) # Add back in the row averages

## Check the error
rmse = np.sqrt(np.sum(A_final.ravel(order='F')[index_nan] - data_orig.ravel(order='F')[index_nan])**2/len(index_nan))
print(rmse)

## Print some of the approximations versus the truth
print(np.vstack((A_final.ravel(order='F')[index_nan], data_orig.ravel(order='F')[index_nan])))
