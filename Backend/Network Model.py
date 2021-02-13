import pandas as pd
import tensorflow as tf

#read in saved population data
#first 2 elements are name and neighbors, rest are the trainable features
population = pd.read_csv("population.csv")

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

# vocabulary list for categorical data
# the order matters for some features (categories listed in ascending order of drinking level for example)
Vocabulary = { 'sex' : ['male', 'female'],
     'status' : ['single', 'seeing someone'],
    'sexual orientation' : ['straight', 'gay'],
     'body_type' : ['skinny', 'thin', 'average', 'fit', 'athletic', 'curvy', 'a little extra', 'full figured', 'jacked'],
    'diet' : ['strictly other', 'mostly other', 'other', 'strictly vegetarian', 'mostly vegetarian', 'vegetarian',
              'strictly anything', 'mostly anything', 'anything'],
     'drinks' : ['not at all', 'rarely', 'socially', 'often', 'desperately'],
    'drugs' : ['never', 'sometimes'],
     'education' : ['high school', 'dropped out', 'working', 'graduated'],
    'ethnicity' : ['white', 'native', 'middle', 'hispanic', 'black', 'asian', 'pacific', 'indian'],
     'job title' : ['transportation', 'hospitality', 'student', 'artistic', 'computer', 'science', 'banking', 'sales',
                    'medicine', 'executive', 'clerical', 'construction', 'political', 'law', 'education', 'military'],
    'offspring' : ["doesn't want", "might want", "doesn't have kids", "has", "wants"],
     'pets' : ["likes dogs and cats", "likes dogs", "likes cats", "has dog", "has cat"],
    'religion' : ['agnostic', 'atheism', 'buddhism', 'catholicism', 'very serious', 'somewhat serious',
                  'not too serious', 'laughing'],
     'smokes' : ["no","trying to quit", "sometimes", "when drinking", "yes"],
    'speaks' : ["english", "spanish", "french", "german", "sign language", "italian", "japanese", "russian", "gujarati",
                "hindi", "chinese", "sanskrit", "portuguese"]}

class User:
    def __init__(self, make_linear_classifier = True):
        self.features = dict(zip(features, range(len(features))))
        self.neighbors = []
        self.linear_classifier = None

        # setup person's features and
        # setup linear_classifier

        # user needs to input info to create this linear classifier
        if make_linear_classifier:



def main():
    pass


main()