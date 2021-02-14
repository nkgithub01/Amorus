import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import pandas as pd
import tensorflow as tf

# the id of a user is just their index / row-1 in the dataframe population
# first 2 elements are name and neighbors, rest are the trainable features
population = pd.read_csv("population.csv", na_values=['nan'])

curr_id = 0

names_to_id = dict()
for id in range(population.shape[0]):
    name = population.loc[id, 'name']
    if name in names_to_id:
        names_to_id[name].append(id)
    else:
        names_to_id[name] = [id]

# id_to_user[id] points to the object of type User associated with this id
id_to_user = [None]*population.shape[0]

# adjacency list for the matchfinding function
adj = [population.loc[i]['neighbors'] for i in range(population.shape[0])]

# features that a person will have
features = list(population.columns)

# category/ vocabulary list for features that have categorical data
# the order matters for some features (categories listed in ascending order of drinking level for example)
categories = { 'sex' : ['m', 'f'],
     'status' : ['single', 'seeing someone'],
    'orientation' : ['straight', 'gay'],
     'body_type' : ['skinny', 'thin', 'average', 'fit', 'athletic', 'curvy', 'a little extra', 'full figured', 'jacked'],
    'diet' : ['strictly other', 'mostly other', 'other', 'strictly vegetarian', 'mostly vegetarian', 'vegetarian',
              'strictly anything', 'mostly anything', 'anything'],
     'drinks' : ['not at all', 'rarely', 'socially', 'often', 'desperately'],
    'drugs' : ['never', 'sometimes'],
     'education' : ['high school', 'graduated from high school','working on college/university',
                    'graduated from college/university', 'working on masters program', 'graduated from masters program',
                    'working on ph.d program', 'graduated from ph.d program', 'dropped out of space camp',
                    'working on space camp', 'graduated from space camp'],
    'ethnicity' : ['white', 'native', 'middle', 'hispanic', 'black', 'asian', 'pacific', 'indian'],
     'job' : ['transportation', 'hospitality', 'student', 'artistic', 'computer', 'science', 'banking', 'sales',
                    'medicine', 'executive', 'clerical', 'construction', 'political', 'law', 'education', 'military'],
    'offspring' : ["doesn't want", "might want", "doesn't have kids", "has", "wants"],
     'pets' : ["likes dogs and cats", "likes dogs", "likes cats", "has dog", "has cat"],
    'religion' : ['agnosticism and laughing about it', 'agnosticism but not too serious about it',
                  'agnosticism and somewhat serious about it', 'agnosticism and very serious about it', 'agnosticism',
                  'atheism and laughing about it', 'atheism but not too serious about it',
                  'atheism and somewhat serious about it', 'atheism and very serious about it', 'atheism',
                  'christianity and laughing about it', 'christianity but not too serious about it',
                  'christianity and somewhat serious about it', 'christianity and very serious about it', 'christianity',
                  'catholicism and laughing about it', 'catholicism but not too serious about it',
                  'catholicism and somewhat serious about it', 'catholicism and very serious about it', 'catholicism',
                  'buddhism and laughing about it', 'buddhism but not too serious about it',
                  'buddhism and somewhat serious about it', 'buddhism and very serious about it', 'buddhism',
                  'judaism and laughing about it', 'judaism but not too serious about it',
                  'judaism and somewhat serious about it', 'judaism and very serious about it', 'judaism',
                  'hinduism and laughing about it', 'hinduism but not too serious about it',
                  'hinduism and somewhat serious about it', 'hinduism and very serious about it', 'hinduism'
                  'other and laughing about it', 'other but not too serious about it',
                  'other and somewhat serious about it', 'other and very serious about it', 'other'],
     'smokes' : ["no","trying to quit", "sometimes", "when drinking", "yes"],
    'speaks' : ["english", "spanish", "french", "german", "sign language", "italian", "japanese", "russian", "gujarati",
                "hindi", "chinese", "sanskrit", "portuguese"]}

# make sure the nan values are filled in or you will get errors with the .train function (expects some type)
for feature in features:
    if feature in categories:
        population[feature].fillna('', inplace=True)
    else:
        population[feature].fillna(0, inplace=True)


#function that makes an input function for training/predicting
def make_input_function(training_examples, love_interests, num_epochs=10, shuffle=True, batch_size=5):
    def input_function():
        # convert from pandas representation of csv data to tensors
        ds = tf.data.Dataset.from_tensor_slices((dict(training_examples), love_interests))  # takes a tuple

        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size).repeat(num_epochs)
        return ds

    return input_function


class User:
    def __init__(self, user_features = 0, linear_classifier = 0):
        # dict of features that have user_features as values
        self.features = dict(zip(features, [None]*len(features)))
        # user features list
        self.features_list = []
        # linear regression model
        self.linear_classifier = None


        # setup person's features
        if user_features == 0:
            # user needs to input their personal features

            print("\nEnter your name")
            inp = input().strip()
            self.features['name'] = inp

            for feature in features[2:]:
                if feature in categories:
                    # categorical feature
                    print("\nEnter the number of the option that best describes you")
                    print(feature + ':')
                    for a, b in enumerate(categories[feature]):
                        print(str(a) + '. ' + b)
                    inp = int(input())
                    self.features[feature] = categories[feature][inp]

                else:
                    # numerical feature
                    print('\nEnter your ' + feature + ':')
                    inp = int(input())
                    self.features[feature] = inp

            self.features['neighbors'] = []
            print("\nNow let's connect you to your friends :)")
            while True:
                print('\nEnter the name of the person who you are friends with or enter \"amorus\" to stop')
                inp = input().strip()

                if inp == 'amorus':
                    break
                else:
                    if inp not in names_to_id:
                        print("\nSorry! It seems your friend is not a user :(")
                    else:
                        print("\nEnter the id number of the person who is most similar to your friend")
                        for id in names_to_id[inp]:
                            print("\nID:", str(id))
                            print("Description:\n")
                            for feature in features:
                                print(feature + ':', population.loc[id, feature])

                        inp = int(input())
                        self.features['neighbors'].append(inp)

            for feature in features:
                self.features_list.append(self.features[feature])
        else:
            # just create this object of type User using the previously stored user_features
            self.features_list = user_features
            self.feature_list = {features[i]: user_features[i] for i in range(len(features))}

        # setup linear_classifier
        if linear_classifier == 0:
            # make new linear classifier
            # user needs to input who they like to create this linear classifier

            # user input determines the user's love interest in some random users
            training_examples = population.sample(min(population.shape[0], 20)).iloc[:, 2:]  # first 2 features not trainable
            love_interests = [0]*training_examples.shape[0]
            print('''\nLet's find out who you like!
For each person, enter a number from 0 to 100:
0 means that this person is not attractive at all
100 means they are as attractive as a person can possibly be''')
            for i in range(training_examples.shape[0]):
                example = list(training_examples.iloc[i])
                print("\nDescription:\n")
                for j in range(2, len(features)):
                    feature = features[j]
                    print(feature + ':', example[j-2])

                while True:
                    print("\nPlease enter an integer between 0 and 100:")
                    inp = input().strip()
                    try:
                        inp = int(inp)
                        if not (0 <= inp <= 100):
                            pass
                        else:
                            love_interests[i] = inp / 100
                            break
                    except ValueError:
                        pass

            feature_columns = []  # list of feature columns
            for feature in features[2:]:
                if feature in categories:
                    vocabulary = categories[feature]  # categories in this feature column where order is sometimes intentional
                    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature, vocabulary))
                else:
                    # the datatype is a int
                    feature_columns.append(tf.feature_column.numeric_column(feature, dtype=tf.int32))

            # returns an input function which creates a dataset everytime it's called
            # which can be used later with linear_est and feature columns

            # train the linearclassifier
            train_function = make_input_function(training_examples, love_interests)

            self.linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns)
            self.linear_classifier.train(train_function)

            '''
            # testing:
            preds = list(self.linear_classifier.predict(make_input_function(population.sample(min(population.shape[0], 20)).iloc[:, 2:], love_interests, 1, False, 1)))
            preds2 = list(self.linear_classifier.predict(make_input_function(training_examples, love_interests, 1, False, 1)))

            print("Predictions for the 20 people you entered")
            for i in [pred['probabilities'][1] for pred in preds2]:
                print(i)
            print("Average predicted percentage that you are attracted to the 20 people you entered:",
                  sum([pred['probabilities'][1] for pred in preds2]) / 20)

            print("\nPredictions for 20 random people:")
            for i in [pred['probabilities'][1] for pred in preds]:
                print(i)
            print("Average predicted percentage that you are attracted to 20 random people:",
                  sum([pred['probabilities'][1] for pred in preds])/20)
            '''
        else:
            # just create the linear classifier using the previously stored linear classifier
            pass

        # edit some global variables to incorporate this User into the network of existing Users
        if user_features == 0:
            global curr_id
            adj.append(self.features['neighbors'])
            id_to_user[curr_id] = self
            curr_id += 1
            name = self.features['name']
            if name in names_to_id:
                names_to_id[name].append(id)
            else:
                names_to_id[name] = [id]

        print("\nYou've been Added!")


# ONLY ADD NEW USERS AFTER LOADING PREEXISTING DATA (will mess up indexing/ids if you don't)
def add_new_user():
    User()


def add_random_user():
    pass


def add_preexisting_user():
    pass

# add cracked bfs nitin C^(length of the path) * product of 1/(all compatablilites(both directions))
def matchmake():
    pass


def main():
    add_new_user()

main()