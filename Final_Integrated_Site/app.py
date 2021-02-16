from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from os import path
import random
import Network_Model as nm

#testt
app = Flask(__name__)

isSignedIn = False
userFeatures = {}

profilesShown = 0
training_labels = [[0,0]]*20

dataSet = nm.population

@app.route('/')
def home():
    print("hi")
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
    return render_template("userDataInput.html", id=dataSet.shape[0]+1) 


@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    global isSignedIn
    isSignedIn = True

    #check if user wants new account
    #load in all the user data
    return render_template("homeLoggedIn.html", id= request.form["id"])
         


def returnRandomProfile():
    global dataSet

    randInt = random.randrange(1, dataSet.shape[0], 1)
    features = []
    
    for i in dataSet.iloc[randInt]:
        features.append(i)
    #for element in row, append to list
    features.append(randInt)
    return features

@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    global isSignedIn
    global userFeatures
    try:
        isSignedIn = True

        #oh god
        userFeatures['name'] = request.form['name']
        userFeatures['neighbors'] = request.form['friends']
        userFeatures['age'] = request.form['age']
        userFeatures['status'] = request.form['status']
        userFeatures['sex'] = request.form['sex']
        userFeatures['orientation'] = request.form['orientation']
        userFeatures['body_type'] = request.form['bodyType']
        userFeatures['diet'] = request.form['diet']
        userFeatures['drinks'] = request.form['drinks']
        userFeatures['drugs'] = request.form['drugs']
        userFeatures['education'] = request.form['education']
        userFeatures['ethnicity'] = request.form['ethnicity']
        userFeatures['height'] = request.form['height']
        userFeatures['income'] = request.form['income']
        userFeatures['job'] = request.form['job']
        userFeatures['offspring'] = request.form['offspring']
        userFeatures['pets'] = request.form['pets']
        userFeatures['religion'] = request.form['religion']
        userFeatures['smokes'] = request.form['smokes']
        userFeatures['speaks'] = request.form['speaks']

        return beginSearch()

    except Exception as e:
        return "bro u dummy thicc: go back and fill out *ALL* the forms >:C"

@app.route("/beginSearch")
def beginSearch():
    global profilesShown
    profilesShown+=1
    
    features = returnRandomProfile()

    training_labels[profilesShown-1] = [features[-1], 0]
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

    training_labels[profilesShown-1][1] = request.form['profileRating']

    if profilesShown >= 2:
        #call function which returns matrix of candidates and their attributes
        nm.add_new_user(userFeatures, training_labels)
        return render_template("searchResults.html")
    else:
        profilesShown+=1
        features = returnRandomProfile()

        training_labels[profilesShown-1] = [features[-1], 0]
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

