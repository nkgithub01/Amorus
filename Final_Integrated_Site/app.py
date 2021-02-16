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

userID = ""

dataSet = nm.population

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
    global userID

    userID = dataSet.shape[0]
    return render_template("userDataInput.html", id=dataSet.shape[0]+1) 


@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    global isSignedIn
    global userID
    
    if int(request.form['id']) >= dataSet.shape[0]:
        return "u dummy thic fake ID"
    #check if user wants new account
    #load in all the user data
    userID = request.form['id']
    isSignedIn = True
    return render_template("homeLoggedIn.html", id= request.form["id"])
         


def returnRandomProfile():
    global dataSet

    randInt = random.randrange(0, dataSet.shape[0])
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
    global userFeatures

    training_labels[profilesShown-1][1] = request.form['profileRating']

    if profilesShown >= 20:
        #call function which returns matrix of candidates and their attributes
        nm.add_new_user(userFeatures, training_labels)
        compatibleUsers = nm.find_connected_users(int(userID))

        score1 = compatibleUsers[1][0][0]
        name1 = dataSet["name"][compatibleUsers[1][0][1]]

        score2 = compatibleUsers[1][1][0]
        name2 = dataSet["name"][compatibleUsers[1][1][1]]

        score3 = compatibleUsers[1][2][0]
        name3 = dataSet["name"][compatibleUsers[1][2][1]]

        score4 = compatibleUsers[1][3][0]
        name4 = dataSet["name"][compatibleUsers[1][3][1]]

        score5 = compatibleUsers[1][4][0]
        name5 = dataSet["name"][compatibleUsers[1][4][1]]

        score6 = compatibleUsers[1][5][0]
        name6 = dataSet["name"][compatibleUsers[1][5][1]]

        score7 = compatibleUsers[1][6][0]
        name7 = dataSet["name"][compatibleUsers[1][6][1]]

        score8 = compatibleUsers[1][7][0]
        name8 = dataSet["name"][compatibleUsers[1][7][1]]

        score9 = compatibleUsers[1][8][0]
        name9 = dataSet["name"][compatibleUsers[1][8][1]]

        score10 = compatibleUsers[1][9][0]
        name10 = dataSet["name"][compatibleUsers[1][9][1]]



        return render_template("searchResults.html", totalCompatible=compatibleUsers[0], \
            name1=name1, score1=score1, \
            name2=name2, score2=score2, \
            name3=name3, score3=score3, \
            name4=name4, score4=score4, \
            name5=name5, score5=score5, \
            name6=name6, score6=score6, \
            name7=name7, score7=score7, \
            name8=name8, score8=score8, \
            name9=name9, score9=score9, \
            name10=name10, score10=score10, )
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


@app.route("/doMatchmake")
def doMatchmake():
    global userID
    global dataSet

    compatibleUsers = nm.find_connected_users(int(userID))

    score1 = compatibleUsers[1][0][0]
    if compatibleUsers[1][0][1] != -1:
        name1 = dataSet.loc[compatibleUsers[1][0][1], "name"]
    else:
        name1 = "No User Found"

    score2 = compatibleUsers[1][1][0]
    if compatibleUsers[1][1][1] != -1:
        name2 = dataSet.loc[compatibleUsers[1][1][1], "name"]
    else:
        name2 = "No User Found"

    score3 = compatibleUsers[1][2][0]
    if compatibleUsers[1][2][1] != -1:
        name3 = dataSet.loc[compatibleUsers[1][2][1], "name"]
    else:
        name3 = "No User Found"

    score4 = compatibleUsers[1][3][0]
    if compatibleUsers[1][3][1] != -1:
        name4 = dataSet.loc[compatibleUsers[1][3][1], "name"]
    else:
        name4 = "No User Found"

    score5 = compatibleUsers[1][4][0]
    if compatibleUsers[1][4][1] != -1:
        name5 = dataSet.loc[compatibleUsers[1][4][1], "name"]
    else:
        name5 = "No User Found"

    score6 = compatibleUsers[1][5][0]
    if compatibleUsers[1][5][1] != -1:
        name6 = dataSet.loc[compatibleUsers[1][5][1], "name"]
    else:
        name6 = "No User Found"

    score7 = compatibleUsers[1][6][0]
    if compatibleUsers[1][6][1] != -1:
        name7 = dataSet.loc[compatibleUsers[1][6][1], "name"]
    else:
        name7 = "No User Found"

    score8 = compatibleUsers[1][7][0]
    if compatibleUsers[1][7][1] != -1:
        name8 = dataSet.loc[compatibleUsers[1][7][1], "name"]
    else:
        name8 = "No User Found"

    score9 = compatibleUsers[1][8][0]
    if compatibleUsers[1][8][1] != -1:
        name9 = dataSet.loc[compatibleUsers[1][8][1], "name"]
    else:
        name9 = "No User Found"

    score10 = compatibleUsers[1][9][0]
    if compatibleUsers[1][9][1] != -1:
        name10 = dataSet.loc[compatibleUsers[1][9][1], "name"]
    else:
        name10 = "No User Found"



    return render_template("searchResults.html", totalCompatible=compatibleUsers[0], \
        name1=name1, score1=score1, \
        name2=name2, score2=score2, \
        name3=name3, score3=score3, \
        name4=name4, score4=score4, \
        name5=name5, score5=score5, \
        name6=name6, score6=score6, \
        name7=name7, score7=score7, \
        name8=name8, score8=score8, \
        name9=name9, score9=score9, \
        name10=name10, score10=score10, id=dataSet["name"][int(userID)] )
