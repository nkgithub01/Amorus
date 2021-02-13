def imp():
    import pandas as pd
    import tensorflow as tf

population = pd.lo
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

features = ['age', 'status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
                       'ethnicity', 'height', 'income', 'job', 'offspring', 'pets', 'religion', 'smokes', 'speaks']
class User:
    def __init__(self, make_linear_classifier = True):
        self.features = dict(zip(features, range(len(features))))
        self.linear_classifier = None

        # setup linear_classifier
        if make_linear_classifier:


def main():
    pass


main()