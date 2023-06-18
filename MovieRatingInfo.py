import numpy as np
### This script shows information for the movie rankings for AMATH 301, Winter 2023

# Load in the data, save it to the variable "table". We will do this using "pandas"
import pandas as pd
data = pd.read_csv('movie_ratings.csv')
print(data)

#import pdb
#pdb.set_trace()
# Change the index to student numbers, instead of rows
data = data.set_index('StudentNo') # Index by student number# Print the data to see what we have
print(data)
print('Shape before any cleaning = ', data.shape)

## We need to clean the data a little bit. 
# Remove those from the list that answered anything but "4" 
# for the first quality control test, and remove those who 
# answered anything but "2" for the second quality control test.
qc1 = 'Answer "4" to show that you are still paying attention (2016)'
qc2 = 'Answer "2" to show that you are still paying attention (2020)'
data = data[data[qc1]==4.0]
data = data[data[qc2]==2.0]
print('Shape after checking quality control tests ', data.shape)

# Remove the two quality control "movies" from the dataset now
data = data.drop(columns = [qc1, qc2])

## Also remove those people that rated every movie. This is 
## a choice: I think it is unlikely to have occured.
number_nans = data.isna().sum(axis=1) # This is the number of nans for each person
# Remove those that have 0 nans: that means they watched everything.
data = data[number_nans != 0]
# Print the size of the data set again to see how many people were removed
print('Shape after removing those who answered every question= ', data.shape)


### Now that we are happy with our data let's look at some statistics
# pandas does this for us easily with "describe"
stats = data.describe() 
# Print what we get
print(stats)

# First let's see what the most watched movie is.
pd.set_option('display.max_rows', None) # Allows us to print everything
print(stats.sort_values('count', ascending=False, axis=1).iloc[0])

# Now let's see what the highest rated movie is.
print(stats.sort_values('mean', ascending=False, axis=1).iloc[1])


### Now that we know a little bit about our data, let's save the cleaned up
### data. 
data.to_csv('cleaned_data.csv', index=True)
