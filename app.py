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

    if path.exists("/loginData.csv"):
        #request username from the csv and add it to the handle
        #load in all the user data
        return render_template("homeLoggedIn.html")
    else:
        d = {'AmorusHandle': [request.form['handle']], 'Password': [request.form['password']]}
        df = pd.DataFrame(data=d)
        df.to_csv('loginData.csv')
        return render_template("userDataInput.html", handle=request.form['handle'])  

    #do bunch of sign in shit   

@app.route("/beginSearch", methods=['GET', 'POST'])
def beginSearch():
    #start search algorithm
    return render_template("searchResults.html")
