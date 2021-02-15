import pandas as pd
#import tensorflow as tf
from sklearn.linear_model import LinearRegression
import json
from joblib import dump, load
import random

def string_to_dict(d):

# the id of a user is just their index / row-1 in the dataframe population
# first 2 elements are name and neighbors, rest are the trainable features
population = pd.read_csv("populations.csv", na_values=['nan'])

#need to convert string of a dict to an actual dict
new_neighbors_col = []
for i in range(population.shape[0]):
    dct = population.loc[i, 'neighbors'].replace("\'", '\"')
    print(dct)
    new_neighbors_col.append(json.loads(dct))
population['neighbors'] = new_neighbors_col
print(isinstance(population.loc[0, 'neighbors'], dict))


# this is to ensure we keep the indexing right
added_preexisting_data = False

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

# load linear classifiers that have been saved
print("loading linear classifiers")
linear_classifiers = [load(population.loc[id, 'linear classifier']) for id in range(population.shape[0])]
#linear_classifiers = [0]*population.shape[0]
print("finished loading linear classifiers")

# features that a person will have
features = list(population.columns)

# category/ vocabulary list for features that have categorical data
# the order matters for some features (categories listed in ascending order of drinking level for example)
categories = { 'sex' : ['other', 'm', 'f'],
     'status' : ['other', 'single', 'seeing someone'],
    'orientation' : ['other', 'straight', 'gay'],
     'body_type' : ['other', 'skinny', 'thin', 'average', 'fit', 'athletic', 'curvy', 'a little extra', 'full figured', 'jacked'],
    'diet' : ['other', 'strictly other', 'mostly other', 'other', 'strictly vegetarian', 'mostly vegetarian', 'vegetarian',
              'strictly anything', 'mostly anything', 'anything'],
     'drinks' : ['other', 'not at all', 'rarely', 'socially', 'often', 'desperately'],
    'drugs' : ['other', 'never', 'sometimes'],
     'education' : ['other', 'high school', 'graduated from high school','working on college/university',
                    'graduated from college/university', 'working on masters program', 'graduated from masters program',
                    'working on ph.d program', 'graduated from ph.d program', 'dropped out of space camp',
                    'working on space camp', 'graduated from space camp'],
    'ethnicity' : ['other', 'white', 'native', 'middle', 'hispanic', 'black', 'asian', 'pacific', 'indian'],
     'job' : ['other', 'transportation', 'hospitality', 'student', 'artistic', 'computer', 'science', 'banking', 'sales',
                    'medicine', 'executive', 'clerical', 'construction', 'political', 'law', 'education', 'military'],
    'offspring' : ['other', "doesn't want", "might want", "doesn't have kids", "has", "wants"],
     'pets' : ['other', "likes dogs and cats", "likes dogs", "likes cats", "has dog", "has cat"],
    'religion' : ['other'],
     'smokes' : ['other', "no","trying to quit", "sometimes", "when drinking", "yes"],
    'speaks' : ['other', "english", "spanish", "french", "german", "sign language", "italian", "japanese", "russian", "gujarati",
                "hindi", "chinese", "sanskrit", "portuguese"]}

categories = {key: dict(zip(categories[key], range(1,len(categories[key])+1))) for key in categories.keys()}
categories_rev = {key: dict(enumerate(categories[key],start =1)) for key in categories.keys()}

# make sure the nan values are filled in or you will get errors with the .train function (expects some type)
for feature in features:
    if feature in categories:
        population[feature].fillna('', inplace=True)
    else:
        population[feature].fillna(0, inplace=True)


# linear regression section
###################################################################################################################

# alters user_features so that categorical features are numerical
# converts a single user's list of user features
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

# converts a list where each element is a list of user features
def make_features_all(user_features_all):
    new_features_all = []
    for user_features in user_features_all:
        new_features_all.append(make_features(user_features))

    return new_features_all

# adds an edge from u to v via adding a key value pair in u's neighbor dictionary
# with key = v and value = predicted compatibility and vice versa for v
def make_edge(u,v):
    prediction = linear_classifiers[u].predict([make_features(population.loc[v])])[0]
    # make sure it's between 0 and 1
    prediction = max(0, prediction)
    prediction = min(1, prediction)
    population.loc[u, 'neighbors'][v] = prediction

    # print(population.loc[id, 'neighbors'][neighbor])
    prediction = linear_classifiers[v].predict([make_features(population.loc[u])])[0]
    # make sure it's between 0 and 1
    prediction = max(0, prediction)
    prediction = min(1, prediction)
    population.loc[v, 'neighbors'][u] = prediction

    # print(population.loc[id, 'neighbors'][neighbor])

###################################################################################################################


class User:
    def __init__(self, user_features = None, linear_classifier = None, user_id = population.shape[0]):
        # dict of features that have user_features as values
        self.features = dict(zip(features, [None]*len(features)))
        # user features list
        self.features_list = []
        # linear regression model
        self.linear_classifier = None


        # setup person's features -> order matters create feature_list then create features later
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
            self.features = {features[i]: user_features[i] for i in range(len(features))}

        # setup linear_classifier
        if linear_classifier is None:
            # make new linear classifier
            # user needs to input who they like to create this linear classifier

            # user input determines the user's love interest in some random users
            training_examples = population.sample(min(population.shape[0], 5))
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

        else:
            # just create the linear classifier using the previously stored linear classifier
            self.linear_classifier = linear_classifier
            pass

        # edit some global variables to incorporate this User into the network of existing Users
        # Also save the new user to the csv file population
        if user_id == population.shape[0]:
            # add weights for neighbors
            for neighbor in self.features['neighbors']:
                prediction = linear_classifiers[id].predict([make_features(population.loc[neighbor])])[0]
                # make sure it's between 0 and 1
                prediction = max(0, prediction)
                prediction = min(1, prediction)
                self.features['neighbors'][neighbor] = [id, 'neighbors'][neighbor] = prediction

            # update global vars
            id_to_user.append(self)
            name = self.features['name']
            if name in names_to_id:
                names_to_id[name].append(population.shape[0])
            else:
                names_to_id[name] = [population.shape[0]]
            linear_classifiers.append(self.features['linear classifier'])
            adj.append(self.features['neighbors'])

            # add to data frame and update csv
            population.loc[population.shape[0]] = self.features_list
            pd.DataFrame(population.loc[population.shape[0] - 1]).T.to_csv('populations.csv', mode='a', header=False)
        elif user_id >= 0:
            id_to_user[user_id] = self
        # print("\nYou've been Added!")


# takes in a dataframe of users without neighbors or linear regression models and a number of distinct linear classifiers
# randomly assigns neighbors and a linear classifier to each user and updates the csv file populations
# then it creates all the classes of the users
# population = dataset to train, num_distinct = number of distinct lin classifiers, num_examples = num training examples
######## if you have time make sure that after running this it's the same as running add_preexisting_users + setting up global vars
def add_random_neighbors_and_lin_class_users(population, max_friends=25, num_distinct=60000, num_examples=20):
    # give each user 0 - max friends friends/neighbors
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        num_neighbors = random.randint(1, max_friends)
        population.loc[id, 'neighbors'].clear()
        for j in range(num_neighbors):
            neighbor = random.randint(0, population.shape[0]-1)
            population.loc[id, 'neighbors'][neighbor] = 0

    linear_classifiers2 = []
    ids = list(range(population.shape[0]))  # randomly ordered ids
    random.shuffle(ids)
    shuffles = 1
    for i in range(num_distinct):
        if i % 1000 == 0:
            print(i)

        # reshuffle data so that it's randomized
        if shuffles*len(ids) < i*num_examples:
            random.shuffle(ids)
            shuffles += 1

        # for getting a random sample of x users
        training_examples = make_features_all([list(population.iloc[ids[(i*num_examples+j) % len(ids)]]) for j in range(num_examples)])
        love_interests = [random.randint(0, 100)/100 for j in range(num_examples)]
        # create linear classifier
        linear_classifiers2.append(LinearRegression().fit(training_examples, love_interests))
        # download linear classifier to directory
        dump(linear_classifiers2[i], f'Linear Classifiers/{i}.joblib')
        # test to make sure you can reload it
        a = load(f'Linear Classifiers/{i}.joblib')
        '''
        for i in a.predict(training_examples):
            print(i)
        '''

    # assign each user a random classifier
    id_to_classifier = [random.randint(0, num_distinct-1) for i in range(population.shape[0])]
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        # add the linear classifiers to the dataframe and make them accessible via linear_classifiers array
        linear_classifiers[id] = linear_classifiers2[id_to_classifier[id]]
        population.loc[id, 'linear classifier'] = f'Linear Classifiers/{id_to_classifier[id]}.joblib'

        # find the percent that the current user is attracted to their neighbors and save it (like an edge weight)
        for neighbor in population.loc[id, 'neighbors']:


        # print(list(population.loc[id]))
        User(list(population.loc[id]), linear_classifiers[id], id)

    # rewrite to the csv file
    population.to_csv("populations2.csv", index=False)


def add_preexisting_users():
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        User(list(population.loc[id]), linear_classifiers[id], id)

    global added_preexisting_data
    added_preexisting_data = True


# ONLY ADD NEW USERS AFTER LOADING PREEXISTING DATA (will mess up indexing/ids if you don't)
def add_new_user():
    if not added_preexisting_data:
        print("Add the preexisting users first!")
    else:
        User()


# add cracked bfs nitin C^(length of the path) * product of 1/(all compatibilities(both directions))
def matchmake(user_id):
    print("Best match:\n")
    for i in range(population.shape[1]):
        print(population.iloc[100, i])


def main():
    add_random_neighbors_and_lin_class_users(population)
    #add_preexisting_users()
    #add_new_user()
    #matchmake(10)

main()