from flask import Flask, render_template, request, session
import secrets
import random
import Network_Model as nm

#testt
app = Flask(__name__)

app.secret_key = secrets.token_bytes(32)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/matchmaker', methods=['GET', 'POST'])
def matchmaker():
    if 'id' in session:
        return render_template('matchmake.html', id=nm.population.loc[session['id'], "name"])
    else:
        return render_template("login.html")

@app.route("/signIn", methods=['GET','POST'])
def signIn():
    return render_template("login.html")


@app.route("/newAccount")
def newAccount():
    return render_template("userDataInput.html")


@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    # user is just going back to the home logged in page we should already have their id
    if 'id' not in request.form:
        return render_template("homeLoggedIn.html", id=nm.population.loc[session['id'], "name"])

    if int(request.form['id']) >= nm.population.shape[0]:
        return "This id does not exist in the database, consider making a new account"
    # check if user wants new account
    # load in all the user data
    else:
        session['id'] = int(request.form['id'])
        return render_template("homeLoggedIn.html", id=nm.population.loc[session['id'], "name"])


@app.route("/dologout")
def dologout():
    session.clear()
    return render_template("index.html")


def returnRandomProfile():
    randInt = random.randrange(0, nm.population.shape[0])
    features = []
    
    for i in nm.population.iloc[randInt]:
        features.append(i)
    #for element in row, append to list
    features.append(randInt)
    return features


@app.route("/uploadData", methods=['GET', 'POST'])
def uploadData():
    try:
        # create user features key to be used later for creating user object and feeding to backend
        session['user features'] = {}

        session['user features']['name'] = request.form['name']
        session['user features']['neighbors'] = request.form['friends']
        session['user features']['age'] = float(request.form['age'])
        session['user features']['status'] = request.form['status']
        session['user features']['sex'] = request.form['sex']
        session['user features']['orientation'] = request.form['orientation']
        session['user features']['body_type'] = request.form['bodyType']
        session['user features']['diet'] = request.form['diet']
        session['user features']['drinks'] = request.form['drinks']
        session['user features']['drugs'] = request.form['drugs']
        session['user features']['education'] = request.form['education']
        session['user features']['ethnicity'] = request.form['ethnicity']
        session['user features']['height'] = float(request.form['height'])
        session['user features']['income'] = float(request.form['income'])
        session['user features']['job'] = request.form['job']
        session['user features']['offspring'] = request.form['offspring']
        session['user features']['pets'] = request.form['pets']
        session['user features']['religion'] = request.form['religion']
        session['user features']['smokes'] = request.form['smokes']
        session['user features']['speaks'] = request.form['speaks']

        # create training labels key to be used later for creating user object and feeding to backend
        # the empty list is because there is no input so we will error and then pop this empty list
        # and begin normal training
        session['training labels'] = [[]]

        return continueSearch()

    except Exception as e:
        return "Please go back and fill out all parts of the form"


@app.route("/continueSearch", methods=['GET','POST'])
def continueSearch():
    needs_to_refill = False
    print("sup")
    try:
        print("trying")
        session['training labels'][-1][1] = float(request.form['profileRating'])
        print("worked")
        session.modified = True
        print(session['training labels'])
    except Exception as e:
        needs_to_refill = True

    if not needs_to_refill and len(session['training labels']) >= 20:
        #call function which returns matrix of candidates and their attributes
        nm.add_new_user(session['user features'], session['training labels'])
        # only after we finally create the user and guarantee the csv has been updated do we do this
        # the reason for not creating the id before is that
        # if person A gets an id before person B but then person B makes a new account/user before
        # person A then person B actually has person a's id in the population database
        session['id'] = nm.population.shape[0]-1

        # after user makes their account they are redirected to the page where they can call match make
        # before we auto did this, but that adds 80 lines of needless copy paste
        return render_template('matchmake.html', id=nm.population.loc[session['id'], "name"])

    else:
        features = returnRandomProfile()

        # remove old id for incomplete training label
        if needs_to_refill:
            session['training labels'].pop()
            session.modified = True

        # create new training label
        session['training labels'].append([features[-1], 0])
        session.modified = True

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


    compatibleUsers = nm.find_connected_users(session['id'])

    score1 = compatibleUsers[1][0][0]
    if compatibleUsers[1][0][1] != -1:
        name1 = nm.population.loc[compatibleUsers[1][0][1], "name"]
    else:
        name1 = "No User Found"

    score2 = compatibleUsers[1][1][0]
    if compatibleUsers[1][1][1] != -1:
        name2 = nm.population.loc[compatibleUsers[1][1][1], "name"]
    else:
        name2 = "No User Found"

    score3 = compatibleUsers[1][2][0]
    if compatibleUsers[1][2][1] != -1:
        name3 = nm.population.loc[compatibleUsers[1][2][1], "name"]
    else:
        name3 = "No User Found"

    score4 = compatibleUsers[1][3][0]
    if compatibleUsers[1][3][1] != -1:
        name4 = nm.population.loc[compatibleUsers[1][3][1], "name"]
    else:
        name4 = "No User Found"

    score5 = compatibleUsers[1][4][0]
    if compatibleUsers[1][4][1] != -1:
        name5 = nm.population.loc[compatibleUsers[1][4][1], "name"]
    else:
        name5 = "No User Found"

    score6 = compatibleUsers[1][5][0]
    if compatibleUsers[1][5][1] != -1:
        name6 = nm.population.loc[compatibleUsers[1][5][1], "name"]
    else:
        name6 = "No User Found"

    score7 = compatibleUsers[1][6][0]
    if compatibleUsers[1][6][1] != -1:
        name7 = nm.population.loc[compatibleUsers[1][6][1], "name"]
    else:
        name7 = "No User Found"

    score8 = compatibleUsers[1][7][0]
    if compatibleUsers[1][7][1] != -1:
        name8 = nm.population.loc[compatibleUsers[1][7][1], "name"]
    else:
        name8 = "No User Found"

    score9 = compatibleUsers[1][8][0]
    if compatibleUsers[1][8][1] != -1:
        name9 = nm.population.loc[compatibleUsers[1][8][1], "name"]
    else:
        name9 = "No User Found"

    score10 = compatibleUsers[1][9][0]
    if compatibleUsers[1][9][1] != -1:
        name10 = nm.population.loc[compatibleUsers[1][9][1], "name"]
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
        name10=name10, score10=score10,
        id=nm.population.loc[session['id'], "name"])

if __name__ == '__main__':
    app.run(debug=False)