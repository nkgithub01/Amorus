from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from os import path
import random

app = Flask(__name__)
isSignedIn = False

profilesShown = 0
training_results = []

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

@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    global isSignedIn
    isSignedIn = True

    #check if user wants new account
    if path.exists("loginData.csv"):
        #request username from the csv and add it to the handle
        #load in all the user data
        return render_template("homeLoggedIn.html")
    else:
        d = {'AmorusID': [request.form['id']]}
        df = pd.DataFrame(data=d)
        df.to_csv('loginData.csv')
        return render_template("userDataInput.html", handle=d['AmorusID'])  

    #do bunch of sign in shit   

def returnRandomProfile():
    #get data from csv of random user and return in list (in order)
    randInt = random.randrange(1, 59947, 1)
    dataSet = pd.read_csv("Backend/population.csv")
    
    features = []
    
    for i in dataSet[randInt]:
        features.append(i)
    #for element in row, append to list
    return features

@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    #addData to CSV file
    return render_template("homeLoggedIn.html")

@app.route("/beginSearch", methods=['GET', 'POST'])
def beginSearch():
    #global profilesShown
    #if len(training_results) != 0:
       # training_results.append(request.form['profileRating'])

    #if profilesShown >= 20:
     # return render_template("searchResults.html")
    #else:
        #profilesShown+=1
       # features = returnRandomProfile()

       # name = features[0]
       # age = features[1]
       # status= features[2]
       # sex= features[3]
       # orientation= features[4]
       # bodyType= features[5]
       # diet = features[6]
       # drinks= features[7]
      #  drugs= features[8]
       # education= features[9]
       # ethnicity= features[10]
       # height= features[11]
      #  income= features[12]
       # job= features[13]
      #  offspring= features[14]
      #  pets= features[15]
      #  religion= features[16]
       # smokes= features[17]
       # speaks= features[18]

        return "ran out of time, check code for linear regression py file"