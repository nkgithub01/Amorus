from flask import Flask, render_template, request
import pandas as pd
from os import path

app = Flask(__name__)
isSignedIn = False

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/matchmaker')
def matchmaker():
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
        #df.to_csv('loginData.csv')
        return render_template("userDataInput.html", handle=d['AmorusID'])  

    #do bunch of sign in shit   

def returnRandomProfile():
    #get data from csv of random user and return in list (in order)
    return

@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    #addData to CSV file

    #call returnRandomProfile, open display page and put in vals from list

    return "hi"

@app.route("/beginSearch", methods=['GET', 'POST'])
def beginSearch():
    #start search algorithm
    return render_template("searchResults.html")
