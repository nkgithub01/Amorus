import tensorflow as tf
import pandas as pd
r = pd.read_csv

# read training and testing data and pop the information under survived
train = 'https://storage.googleapis.com/tf-datasets/titanic/train.csv'
eval = 'https://storage.googleapis.com/tf-datasets/titanic/eval.csv'

d_train = r(train) # features that we will use to train the linear regression
d_eval = r(eval) # features that we will input into our linear regression

y_train = d_train.pop('survived') # boolean column of whether the person survived or not
y_eval = d_eval.pop('survived') # boolean column of survival to compared with the output of our lin reg based on the person's features



# .head() first 5 elements of some data converted from csv
print(d_train.head())

# .describe() statistical analysis of the data converted from csv
print(d_train.describe())

# the dimensions of our training data here it's ( # training examples, # features)
print(d_train.shape)

# shows that it's a m x 2 matrix where the entry [i][1] is if person i survived
print(d_eval.head())



# putting strings into a linear regression seems like a very bad idea
# (you would be doing theta*string instead of theta*numerical_value for that part of the line)
# so let's convert all categorical data (here stuff with strings) into something numerical
feature_columns = [] #list of feature columns
for feature in d_train:
    if isinstance(d_train[feature][0],str):
        # the feature is categorical/ describe by a string so let's convert it via mapping each category with
        # a number via a vocabulary list
        vocabulary = d_train[feature].unique() #unique categories in this feature column
        feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))
    else:
        # the datatype is a float
        feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.float32))


# returns an input function which creates a dataset everytime it's called
# which can be used later with linear_est and feature columns
def make_input_function(training_examples, survival_outcomes, num_epochs=10, shuffle=True, batch_size= 20):
    def input_function():
        # convert from pandas representation of csv data to tensors
        ds = tf.data.Dataset.from_tensor_slices((dict(training_examples), survival_outcomes)) # takes a tuple

        if shuffle:
            ds = ds.shuffle(buffer_size=1000) #randomize order of dataset here 1000 is greater than the
                                  # # of training examples, so the buffer does nothing
        ds = ds.batch(batch_size).repeat(num_epochs)
        '''num_epochs times split dataset into batches of size 32
        The batch size is the number of samples processed before the model is updated and the number of epochs
        is the number of total passes through the training dataset (of course it randomizes the order again)
        Quicker since it only updates num_epochs/batch_size times as much as if you updated at every element but ran 1 time'''

        return ds
    return input_function

train_function = make_input_function(d_train, y_train)
'''10 iterations through data (epochs) since we want to train it well
huffle because we want to make sure data is randomized so no bias, and batch size of 32 to update the linear function
that we are fitting after every 32 elements'''
eval_function = make_input_function(d_eval, y_eval, 1, False,1000)
'''1 pass through data, cuz we aren't training anything
definitely don't want to shuffle because then the order of the data would be messed up'''

linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)
#uses the feature columns (where we converted categorical to numerical) to train the linear estimator
linear_est.train(train_function)
#uses the feature columns (where we converted categorical to numerical) to evaluate the linear estimator
result = linear_est.evaluate(eval_function)

print(result['accuracy'])