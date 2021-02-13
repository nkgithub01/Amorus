from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template("index.html")

@app.route("/matchmake", methods=['GET', 'POST'])
def matchmake():
    return render_template("matchmake.html", messageForEli=request.form['mensaje'])

