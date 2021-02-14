import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
import json
from joblib import dump, load
import random


# the id of a user is just their index / row-1 in the dataframe population
# first 2 elements are name and neighbors, rest are the trainable features
population = pd.read_csv("population.csv", na_values=['nan'])
#need to convert string of a dict to an actual dict
new_neighbors_col = []
for i in range(population.shape[0]):
    new_neighbors_col.append(json.loads(population.loc[i, 'neighbors']))
population['neighbors'] = new_neighbors_col
#print(isinstance(population.loc[0, 'neighbors'], dict))

# this is to ensure we keep the indexing right
added_preexisting_data = False

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
adj = [population.loc[i, 'neighbors'] for i in range(population.shape[0])]

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

categories = {key: dict(zip(categories[key], range(1,len(categories[key])+1))) for key in categories.keys()}
categories_rev = {key: dict(enumerate(categories[key],start =1)) for key in categories.keys()}
print(categories, categories_rev)
# make sure the nan values are filled in or you will get errors with the .train function (expects some type)
for feature in features:
    if feature in categories:
        population[feature].fillna('', inplace=True)
    else:
        population[feature].fillna(0, inplace=True)


# linear regression section
###################################################################################################################
#alters user_features so that categorical features are numerical
def make_features(user_features):
    new_features = []
    for i in range(2, len(features)-1):
        if features[i] in categories:
            # categories in this feature column where order is sometimes intentional
            if user_features[i] in categories[features[i]]:
                # included in some ordering
                new_features.append(float(categories[features[i]][user_features[i]]))
            else:
                # belongs to other which we default to 0
                new_features.append(float(0))
        else:
            # the datatype is numerical
            new_features.append(float(user_features[i]))

    return new_features

def make_features_all(user_features_all):
    new_features_all = []
    for user_features in user_features_all:
        new_features_all.append(make_features(user_features))

    return new_features_all
###################################################################################################################


class User:
    def __init__(self, user_features = None, linear_classifier = None, new_user = False):
        # dict of features that have user_features as values
        self.features = dict(zip(features, [None]*len(features)))
        # user features list
        self.features_list = []
        # linear regression model
        self.linear_classifier = None

        '''
        # setup person's features
        if user_features is None:
            # user needs to input their personal features

            print("\nEnter your name")
            inp = input().strip()
            self.features['name'] = inp

            for feature in features[2:-1]:
                if feature in categories:
                    # categorical feature
                    print("\nEnter the number of the option that best describes you")
                    print(feature + ':')
                    for a, b in enumerate(categories[feature]):
                        print(str(a) + '. ' + b)
                    inp = int(input())
                    self.features[feature] = categories_rev[feature][inp]

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
        '''
        # setup linear_classifier
        if linear_classifier is None:
            # make new linear classifier
            # user needs to input who they like to create this linear classifier

            # user input determines the user's love interest in some random users
            training_examples = population.sample(min(population.shape[0], 20))
            love_interests = [0]*training_examples.shape[0]
            print('''\nLet's find out who you like!
For each person, enter a number from 0 to 100:
0 means that this person is not attractive at all
100 means they are as attractive as a person can possibly be''')
            for i in range(training_examples.shape[0]):
                example = list(training_examples.iloc[i])
                print("\nDescription:\n")
                for j in range(2, len(features)-1):
                    feature = features[j]
                    print(feature + ':', example[j])

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

            # convert dataframe to list of lists and then convert the categorical data to numerical
            training_examples = make_features_all([list(training_examples.iloc[i]) for i in range(training_examples.shape[0])])
            self.linear_classifier = LinearRegression()
            self.linear_classifier.fit(training_examples, love_interests)


            # testing:
            preds = self.linear_classifier.predict(training_examples)
            random_sample = population.sample(20)
            random_sample = make_features_all([list(random_sample.iloc[i]) for i in range(20)])
            preds2 = self.linear_classifier.predict(random_sample)


            print("Predictions for the 20 people you entered")
            for i in preds:
                print(i)
            print("Average predicted percentage that you are attracted to the 20 people you entered:",
                  sum(preds) / 20)

            print("\nPredictions for 20 random people:")
            for i in preds2:
                print(i)
            print("Average predicted percentage that you are attracted to 20 random people:",
                  sum(preds2)/20)
            return
        else:
            # just create the linear classifier using the previously stored linear classifier
            self.linear_classifier = linear_classifier
            pass

        # edit some global variables to incorporate this User into the network of existing Users
        # Also save the new user to the csv file population
        if new_user:
            global curr_id
            adj.append(self.features['neighbors'])
            id_to_user[curr_id] = self
            curr_id += 1
            name = self.features['name']
            if name in names_to_id:
                names_to_id[name].append(curr_id)
            else:
                names_to_id[name] = [curr_id]

            population.loc[population.shape[0]] = self.features_list

            print("\nYou've been Added!")
        else:
            # we were just loading a preexisting user so there's nothing to update
            # curr_id, adj, id_to_user, and names_to_id have already been created from the preexisting users
            pass


# takes in a dataframe of users without neighbors or linear regression models and a number of distinct linear classifiers
# randomly assigns neighbors and a linear classifier to each user and updates the csv file populations
# then it creates all the classes of the users
# population = dataset to train, num_distinct = number of distinct lin classifiers, num_examples = num training examples
def add_random_neighbors_and_lin_class_users(population, max_friends=25, num_distinct=500, num_examples=20):
    #give each neighbor
    for id in range(population.shape[0]):
        num_neighbors = random.randint(1, max_friends)
        population.loc[id, 'neighbors'].clear()
        for j in range(num_neighbors):
            neighbor = random.randint(0, population.shape[0]-1)
            population.loc[id, 'neighbors'][neighbor] = 0

    linear_classifiers = []
    ids = list(range(population.shape[0]))  # randomly ordered ids
    random.shuffle(ids)
    for i in range(num_distinct):
        # for getting a random sample of x users
        training_examples = [population.iloc[ids[(i*num_examples+j) % len(ids)], 2:-1] for j in range(num_examples)]
        print(training_examples[0])
        love_interests = [random.randint(0, 100)/100 for j in range(num_examples)]
        linear_classifiers.append(create_lin_classifier(pd.DataFrame(training_examples), love_interests))
        print(predict(linear_classifiers[-1], population.iloc[10]))
        linear_classifiers[i].export_saved_model(f'Linear Classifiers/{i}', serving_input_receiver_fn=(
            tf.estimator.export.build_parsing_serving_input_receiver_fn(
                tf.feature_column.make_parse_example_spec(feature_columns))))

    id_to_classifier = [random.randint(0, num_distinct-1) for i in range(population.shape[0])]
    for id in range(population.shape[0]):
        population.loc[id, 'linear classifier'] = f'Linear Classifiers/{id_to_classifier[id]}'
        a = tf.estimator.LinearClassifier(feature_columns=feature_columns, warm_start_from=population.loc[id, 'linear classifier'])
        print(predict(a, population.iloc[10]))



def add_preexisting_users():

    global added_preexisting_data
    added_preexisting_data = True
    pass

# ONLY ADD NEW USERS AFTER LOADING PREEXISTING DATA (will mess up indexing/ids if you don't)
def add_new_user():
    if not added_preexisting_data:
        print("Add the preexisting users first!")
    else:
        User()




# add cracked bfs nitin C^(length of the path) * product of 1/(all compatibilities(both directions))
def matchmake(user_id):
    pass


def main():
    User()
    add_random_neighbors_and_lin_class_users(population, 1, 1, 5)
    #add_preexisting_users()
    #add_new_user()

main()