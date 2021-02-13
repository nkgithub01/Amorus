import pandas as pd
all = pd.read_csv("okcupid_profiles.csv") # the entire dataset

features_to_include = set(["age","status",'sex','orientation','body_type','diet','drinks','drugs','education',\
           'ethnicity','height','income','job','offspring','pets','religion','smokes','speaks'])

features_to_not_include = []
for feature in all:
    if feature not in features_to_include:
        features_to_not_include.append(feature)

for feature in features_to_not_include:
    all.pop(feature)

#check that features we want are left
print(set(all.keys()) == features_to_include, '\n')
print(all.head())

all.to_csv("population.csv")



