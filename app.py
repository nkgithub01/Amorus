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
        return "hello"
    else:
        d = {'AmorusHandle': [], 'Password': []}
        df = pd.DataFrame(data=d)
        df.to_csv('loginData.csv')
    #do bunch of sign in shit   
    return render_template("homeLoggedIn.html", handle=request.form['handle'])  

@app.route("/beginSearch", methods=['GET', 'POST'])
def beginSearch():
    #start search algorithm
    return render_template("searchResults.html")