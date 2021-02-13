import pandas as pd
all = pd.read_csv("okcupid_profiles.csv") # the entire dataset

features_to_include = {'name', 'age', 'status', 'sex', 'orientation', 'body_type', 'diet', 'drinks', 'drugs', 'education',
                       'ethnicity', 'height', 'income', 'job', 'offspring', 'pets', 'religion', 'smokes', 'speaks'}

features_to_not_include = []
for feature in all:
    if feature not in features_to_include:
        features_to_not_include.append(feature)

for feature in features_to_not_include:
    del all[feature]

#create the list of names
names_file = pd.read_csv("60knames.csv")
names = [names_file.loc[i][0] + ' ' + names_file.loc[i][1] for i in range(names_file.shape[0])]

#insert the names as the first column in the dataframe
all.insert(0,'name',names[:all.shape[0]])

#check that features we want are left
print(set(all.keys()) == features_to_include, '\n')
print(all.head())
print(all.shape)

#create the population output file
all.to_csv("population.csv")



