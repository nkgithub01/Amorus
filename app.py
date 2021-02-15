from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from os import path
import random

app = Flask(__name__)

isSignedIn = False
userFeatures = []

profilesShown = 0
training_labels = []

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
    global userFeatures
    userFeatures.append(request.form['id'])

    return render_template("userDataInput.html", id=userFeatures[0]) 


@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    global isSignedIn
    isSignedIn = True

    #check if user wants new account
    #load in all the user data
    return render_template("homeLoggedIn.html", id= request.form["id"])
         


def returnRandomProfile():
    #get data from csv of random user and return in list (in order)
    randInt = random.randrange(1, 59947, 1)
    dataSet = pd.read_csv("Backend/population.csv")
    
    features = []
    
    for i in dataSet.iloc[randInt]:
        features.append(i)
    #for element in row, append to list
    return features

@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    #addData to Python class
    return render_template("homeLoggedIn.html")

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