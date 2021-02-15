from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from os import path
import random

app = Flask(__name__)

isSignedIn = False
userFeatures = [None]*20

profilesShown = 0
training_labels = []

dataSet = pd.read_csv("Backend/population.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/matchmaker')
def matchmaker():
    global isSignedIn
    if isSignedIn:
        return render_template('matchmake.html')
    else:
        return render_template("login.html")

@app.route("/signIn", methods=['GET','POST'])
def signIn():
    return render_template("login.html")


@app.route("/newAccount")
def newAccount():
    global dataSet
    return render_template("userDataInput.html", id=dataSet.shape[0]+2) 


@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    global isSignedIn
    isSignedIn = True

    #check if user wants new account
    #load in all the user data
    return render_template("homeLoggedIn.html", id= request.form["id"])
         


def returnRandomProfile():
    global dataSet

    randInt = random.randrange(1, 59947, 1)
    features = []
    
    for i in dataSet.iloc[randInt]:
        features.append(i)
    #for element in row, append to list
    return features

@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    global userFeatures

    #oh god
    userFeatures[0] = request.form['name']
    userFeatures[1] = request.form['friends']
    userFeatures[2] = request.form['age']
    userFeatures[3] = request.form['status']
    userFeatures[4] = request.form['sex']
    userFeatures[5] = request.form['orientation']
    userFeatures[6] = request.form['bodyType']
    userFeatures[7] = request.form['diet']
    userFeatures[8] = request.form['drinks']
    userFeatures[9] = request.form['drugs']
    userFeatures[10] = request.form['education']
    userFeatures[11] = request.form['ethnicity']
    userFeatures[12] = request.form['height']
    userFeatures[13] = request.form['income']
    userFeatures[14] = request.form['job']
    userFeatures[15] = request.form['offspring']
    userFeatures[16] = request.form['pets']
    userFeatures[17] = request.form['religion']
    userFeatures[18] = request.form['smokes']
    userFeatures[19] = request.form['speaks']

    return render_template("homeLoggedIn.html", handle=userFeatures[0], pog = userFeatures)

@app.route("/beginSearch")
def beginSearch():
    global profilesShown
    profilesShown+=1
    
    features = returnRandomProfile()

    name = features[0]
    age = features[2]
    status= features[3]
    sex= features[4]
    orientation= features[5]
    bodyType= features[6]
    diet = features[7]
    drinks= features[8]
    drugs= features[9]
    education= features[10]
    ethnicity= features[11]
    height= features[12]
    income= features[13]
    job= features[14]
    offspring= features[15]
    pets= features[16]
    religion= features[17]
    smokes= features[18]
    speaks= features[19]
    return render_template("displayProfile.html", \
        name = name, \
        age = age, \
        status= status, \
        sex= sex, \
        orientation= orientation, \
        bodyType= bodyType, \
        diet = diet, \
        drinks= drinks, \
        drugs= drugs, \
        education= education, \
        ethnicity= ethnicity, \
        height= height, \
        income= income, \
        job= job, \
        offspring= offspring, \
        pets= pets,\
        religion= religion, \
        smokes= smokes, \
        speaks= speaks )


@app.route("/continueSearch", methods=['GET','POST'])
def continueSearch():
    global profilesShown
    global training_labels

    training_labels.append(request.form['profileRating'])

    if profilesShown >= 20:
        return render_template("searchResults.html")
    else:
        profilesShown+=1
        features = returnRandomProfile()

        #the features are labeled incorrectly
        name = features[0]
        age = features[1]
        status= features[2]
        sex= features[3]
        orientation= features[4]
        bodyType= features[5]
        diet = features[6]
        drinks= features[7]
        drugs= features[8]
        education= features[9]
        ethnicity= features[10]
        height= features[11]
        income= features[12]
        job= features[13]
        offspring= features[14]
        pets= features[15]
        religion= features[16]
        smokes= features[17]
        speaks= features[18]
        return render_template("displayProfile.html", \
            name = name, \
            age = age, \
            status= status, \
            sex= sex, \
            orientation= orientation, \
            bodyType= bodyType, \
            diet = diet, \
            drinks= drinks, \
            drugs= drugs, \
            education= education, \
            ethnicity= ethnicity, \
            height= height, \
            income= income, \
            job= job, \
            offspring= offspring, \
            pets= pets,\
            religion= religion, \
            smokes= smokes, \
            speaks= speaks )