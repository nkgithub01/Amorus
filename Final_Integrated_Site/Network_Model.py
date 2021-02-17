import pandas as pd
# import tensorflow as tf
from sklearn.linear_model import LinearRegression
from joblib import dump, load
import random
from collections import deque

# input and global variables
###################################################################################################################

# global variables


# the id of a user is just their index / row-1 in the dataframe population
# first 2 elements are name and neighbors, rest are the trainable features
saved_database_of_users_in_csv = "population 1k.csv"

saved_folder_of_linear_classifiers = "Linear Classifiers 1k"

population = None

# this is to ensure we keep the indexing right
added_preexisting_data = False

names_to_id = None

# id_to_user[id] points to the object of type User associated with this id
id_to_user = None

# adjacency list for the matchfinding function, list of dictionaries with neighbor : compatibility key value pairs
adj = None

linear_classifiers = None

# features that a person will have
features = None

# category/ vocabulary list for features that have categorical data
# the order matters for some features (categories listed in ascending order of drinking level for example)
categories = { 'sex' : ['other', 'm', 'f'],
     'status' : ['other', 'single', 'seeing someone'],
    'orientation' : ['other', 'straight', 'gay'],
     'body_type' : ['other', 'skinny', 'thin', 'average', 'fit', 'athletic', 'curvy', 'a little extra', 'full figured', 'jacked'],
    'diet' : ['other', 'strictly vegetarian', 'mostly vegetarian', 'vegetarian',
              'strictly anything', 'mostly anything', 'anything'],
     'drinks' : ['other', 'not at all', 'rarely', 'socially', 'often', 'desperately'],
    'drugs' : ['other', 'never', 'sometimes'],
     'education' : ['other', 'high school', 'graduated from high school','working on college/university',
                    'graduated from college/university', 'working on masters program', 'graduated from masters program',
                    'working on ph.d program', 'graduated from ph.d program', 'dropped out of space camp',
                    'working on space camp', 'graduated from space camp'],
    'ethnicity' : ['other', 'white', 'native american', 'middle eastern', 'hispanic', 'black', 'asian', 'pacific islander', 'indian'],
     'job' : ['other', 'transportation', 'hospitality', 'student', 'artistic', 'computer science', 'science', 'banking', 'sales',
                    'medicine', 'executive', 'clerical', 'construction', 'political', 'law', 'education', 'military'],
    'offspring' : ['other', "doesn't want", "might want", "has", "wants"],
     'pets' : ['other', "likes dogs and cats", "likes dogs", "likes cats", "has dog", "has cat"],
    'religion' : ['other', 'atheism', 'agnosticism', 'hindiusm', 'buddhism', 'judaism', 'christianity', 'catholicism'],
     'smokes' : ['other', "no","trying to quit", "sometimes", "when drinking", "yes"],
    'speaks' : ['other', "english", "spanish", "french", "german", "sign language", "italian", "japanese", "russian", "gujarati",
                "hindi", "chinese", "sanskrit", "portuguese"]}

categories = {key: dict(zip(categories[key], range(1, len(categories[key])+1))) for key in categories.keys()}
categories_rev = {key: dict(enumerate(categories[key], start=1)) for key in categories.keys()}


# functions


# converts string of a dict of int : float to an actual python dict
def string_to_dict(d):
    d = d[1:-1]
    d = d.split(",")
    d = [''.join(pair.split()).split(":") for pair in d]

    # no elements in the dict will throw error for last conversion
    if len(d[0]) == 1:
        return {}

    d = [[int(key), float(value)] for key, value in d]
    new_d = dict(d)

    return new_d


# creates a panda dataframe from the csv file "saved database of users"
def csv_to_database():
    global population
    global features

    # the id of a user is just their index / row-1 in the dataframe population
    # first 2 elements are name and neighbors, rest are the trainable features
    population = pd.read_csv(saved_database_of_users_in_csv, na_values=['nan'])

    # convert from string dict to actual dict
    new_neighbors_col = []
    for i in range(population.shape[0]):
        new_neighbors_col.append(string_to_dict(population.loc[i, 'neighbors']))
    population['neighbors'] = new_neighbors_col

    features = list(population.columns)

    # make sure the nan values are filled in or you will get errors with the .train function (expects some type)
    for feature in features:
        if feature in categories:
            population[feature].fillna('', inplace=True)
        else:
            population[feature].fillna(0, inplace=True)


# checks that all global variables and user objects have been created correctly from the csv file
def check_loaded_correctly():
    print("\nadd_preexisting_users_run:\n")
    print(added_preexisting_data)
    if (added_preexisting_data == False):
        raise Exception("You did not load the preexisting data and this is very bad!")
    print("\nFeatures:\n")
    print(features)
    print("\nDict setup properly:\n")
    print(isinstance(population.loc[0, 'neighbors'], dict))
    print("\n10 random users:\n")
    for i in range(10):
        print(list(population.iloc[i]))
    print("\nId to User:\n")
    print(id_to_user[:10])
    print("\nAdj:\n")
    print(adj[:10])
    print("\nNames to id:\n")
    print(str(names_to_id)[:100])
    print("\nRandom Predictions/ check linear_classifiers setup:\n")
    print(linear_classifiers[0].predict([make_features(list(population.iloc[0]))])[0],
    linear_classifiers[0].predict([make_features(list(population.iloc[population.shape[0]-1]))])[0])


###################################################################################################################


# linear regression section
###################################################################################################################

# alters user_features so that categorical features are numerical
# converts a single user's list of user features
# input is essentially list(population[user_id])
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
def make_edge(u, v):
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


# adding Users/ loading preexisting ones from the database and matchmake function
###################################################################################################################
class User:
    def __init__(self, type, user_features, linear_classifier, user_id = -1):
        # dict of features that have user_features as values
        self.features = dict(zip(features, [None]*len(features)))
        # user features list
        self.features_list = []
        # linear regression model
        self.linear_classifier = None

        # setup person's features -> order matters create feature_list then create features later
        if type == "new_user":
            # we use inputted the list of user features from the website
            self.features['name'] = user_features['name']
            digits_and_space = " 0123456789"
            user_features['neighbors'] = ''.join([i for i in user_features['neighbors'] if i in digits_and_space])
            self.features['neighbors'] = \
                dict(zip(map(int, user_features['neighbors'].split()), [0]*len(user_features['neighbors'].split())))
            self.features['neighbors'] = {id: 0 for id in self.features['neighbors'] if id < population.shape[0]}
            for feature in features[2:-1]:
                if feature in categories:
                    self.features[feature] = user_features[feature]
                else:
                    self.features[feature] = float(user_features[feature])

            for feature in features[:-1]:
                self.features_list.append(self.features[feature])
        else:
            # just create this object of type User using the previously stored user_features
            self.features_list = user_features
            self.features = {features[i]: user_features[i] for i in range(len(features))}

        # setup linear_classifier
        if type == "new_user":
            # we use the inputted list of id, percent like pairs from the website
            training_examples = make_features_all([list(population.iloc[int(id)]) for id, percent in linear_classifier])
            love_interests = [float(percent)/100 for id, percent in linear_classifier]

            # create and train linear classifier and save it as a .joblib file in directory Linear Classifiers
            self.linear_classifier = LinearRegression()
            self.linear_classifier.fit(training_examples, love_interests)
            dump(self.linear_classifier, saved_folder_of_linear_classifiers+f'/{population.shape[0]}.joblib')
            self.features['linear classifier'] = saved_folder_of_linear_classifiers+f'/{population.shape[0]}.joblib'
            self.features_list.append(saved_folder_of_linear_classifiers+f'/{population.shape[0]}.joblib')
        else:
            # just create the linear classifier using the previously stored linear classifier
            self.linear_classifier = linear_classifier

        # edit some global variables to incorporate this User into the network of existing Users
        # Also save the new user to the csv file population
        if type == "new_user":
            # update global vars
            id_to_user.append(self)

            name = self.features['name']
            if name in names_to_id:
                names_to_id[name].append(population.shape[0])
            else:
                names_to_id[name] = [population.shape[0]]

            linear_classifiers.append(self.linear_classifier)

            # add to data frame
            population.loc[population.shape[0]] = self.features_list

            # add weights for neighbors
            for neighbor in self.features['neighbors']:
                make_edge(population.shape[0]-1, neighbor)

            adj.append(self.features['neighbors'])

            # update instance attributes
            self.features_list[1] = population.loc[population.shape[0] - 1, 'neighbors']
            self.features['neighbors'] = population.loc[population.shape[0] - 1, 'neighbors']

            # update csv
            population.to_csv(saved_database_of_users_in_csv, index=False)
        elif user_id >= 0:
            id_to_user[user_id] = self
        # print("\nYou've been Added!")


# training 50k users takes 10 minutes
# edits dataframe without neighbors or linear classifiers and adds this to the dataframe and csv file
# also adds linear_classifiers
# doesn't set up the rest of the global variables, so just call add_preexisting_users
# randomly assigns neighbors and a linear classifier to each user and updates the csv file populations
# the only global variable it updates is pandas
# population = dataset to train, num_distinct = number of distinct lin classifiers, num_examples = num training examples
def add_random_neighbors_and_lin_class_users(max_friends=25, num_distinct=60000, num_examples=20):
    global population
    global features
    global linear_classifiers

    csv_to_database()

    linear_classifiers = [0] * population.shape[0]

    # every training example will get a unique linear classifier anyway if num_distinct > # training examples
    num_distinct = min(num_distinct, population.shape[0])

    # give each user 0 - max friends friends/neighbors
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        num_neighbors = random.randint(1, max_friends)
        population.loc[id, 'neighbors'].clear()
        for j in range(num_neighbors):
            neighbor = random.randint(0, population.shape[0]-1)
            population.loc[id, 'neighbors'][neighbor] = None

    # create num_distinct linear classifiers
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
        dump(linear_classifiers2[i], saved_folder_of_linear_classifiers+f'/{i}.joblib')

    # assign each user a random classifier the output is the idx of the classifier in linear_classifiers2
    # this classifier has been saved at 'Linear Classifiers/idx.joblib'
    id_to_classifier = [random.randint(0, num_distinct-1) for i in range(population.shape[0])]
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        # add the linear classifiers to the dataframe and add the linear classifier to linear_classifiers
        population.loc[id, 'linear classifier'] = saved_folder_of_linear_classifiers+f'/{id_to_classifier[id]}.joblib'
        linear_classifiers[id] = linear_classifiers2[id_to_classifier[id]]

    # uses the linear classifiers to find the percent that the current user is attracted to their neighbors + vice versa
    # + save that percent (like an edge weight)
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)
        # find the percent that the current user is attracted to their neighbors and vice versa
        # + save that percent (like an edge weight)
        for neighbor in population.loc[id, 'neighbors']:
            make_edge(id, neighbor)

    # rewrite to the csv file
    population.to_csv(saved_database_of_users_in_csv, index=False)


# adds preexisting users from the csv database via updating global variables and creating User objects for them
def add_preexisting_users():
    global population
    global added_preexisting_data
    global names_to_id
    global id_to_user
    global adj
    global linear_classifiers
    global features

    csv_to_database()

    names_to_id = dict()
    for id in range(population.shape[0]):
        name = population.loc[id, 'name']
        if name in names_to_id:
            names_to_id[name].append(id)
        else:
            names_to_id[name] = [id]

    # adjacency list for the matchfinding function
    adj = [population.loc[i, 'neighbors'] for i in range(population.shape[0])]

    # load linear classifiers that have been saved (takes ~20-30 seconds for 60k)
    print("loading linear classifiers")
    linear_classifiers = [load(population.loc[id, 'linear classifier']) for id in range(population.shape[0])]
    print("finished loading linear classifiers")

    # id_to_user[id] points to the object of type User associated with this id
    id_to_user = [None] * population.shape[0]
    # create the User objects and add them to id_to_user
    for id in range(population.shape[0]):
        if id % 1000 == 0:
            print(id)

        User("existing_user", list(population.loc[id]), linear_classifiers[id], id)

    added_preexisting_data = True


# ONLY ADD NEW USERS AFTER LOADING PREEXISTING DATA (will mess up indexing/ids if you don't)
def add_new_user(user_features, training_labels):
    if not added_preexisting_data:
        print("Add the preexisting users first!")
    else:
        User("new_user", user_features, training_labels)


# for 1000 people with 1-25 neighbors:
# the the average total connected users = 126.213, the min num = 10, and the mx = 235
# the the average percent compatibility of the top 10 most compatible connected users = 86.66%,
# the min max percent compatibility = 16.034%, and the max max percent compatibility = 99.99999999994054%
# keep in mind that anything above 100% compatibility is ignored as overfitting (for connected users + size then)
# cracked bfs c^(length of the path) * product of 1/(all compatibilities(both directions))
def find_connected_users(root_user_id):
    # adds constraint that all connected users must be <= 3 users away
    # and only 3 in the case that percent compatibilities are nearly maxed out >= 0.95% for every person
    max_length = (1/0.95)**4*10**3 + 1
    c = 10
    # queue where each entry is id, path length
    q = deque([[root_user_id, 1]])
    visited = {root_user_id}

    while q:
        curr_user, path_length = q.popleft()

        for neighbor in adj[curr_user]:
            if neighbor not in visited:
                new_path_length = c*path_length
                if adj[curr_user][neighbor] == 0:
                    new_path_length = max_length + 1
                else:
                    # unrealistic for 100% compatibility
                    new_path_length *= 1/min(adj[curr_user][neighbor], 0.95)
                if adj[neighbor][curr_user] == 0:
                    new_path_length = max_length + 1
                else:
                    # unrealistic for 100% compatibility
                    new_path_length *= 1/min(adj[neighbor][curr_user], 0.95)

                if new_path_length <= max_length:
                    visited.add(neighbor)
                    q.append([neighbor, new_path_length])

    visited.remove(root_user_id)
    visited_connected_users = \
        [[(id_to_user[root_user_id].linear_classifier.predict([make_features(id_to_user[user].features_list)])[0])*100, user]
                               for user in visited]
    # unrealistic for 100% compatibility
    visited_connected_users = [[max(0, compatibility), user] for compatibility, user in visited_connected_users if compatibility < 100]
    visited_connected_users.sort(reverse=True)
    # padding so that even if there are less than 10 connected users there will be "nonexistent" users to prevent errors
    len_before_pad = len(visited_connected_users)
    if len(visited_connected_users) < 10:
        visited_connected_users.extend([[0, -1] for i in range(10-len(visited_connected_users))])

    return [len_before_pad, visited_connected_users[:10]]

##############################################################################################################


def main():
    #add_random_neighbors_and_lin_class_users()
    add_preexisting_users()
    check_loaded_correctly()

main()