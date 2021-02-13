from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def homePage():
    return render_template("index.html")

@app.route("matchmake")