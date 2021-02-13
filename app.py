from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template("index.html")

@app.route("/matchmake", methods=['GET', 'POST'])
def matchmake():
    return render_template("matchmake.html", messageForEli=request.form['mensaje'])

@app.route("/login", methods=['GET','POST'])
def loginPage():
    return render_template("login.html")

@app.route("/doLogin", methods=['GET', 'POST'])
def doLogin():
    #do bunch of sign in shit   
    return render_template("homeLoggedIn", handle=request.form['handle'])