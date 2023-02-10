from flask import Flask, render_template
from flask import request
from flask import redirect, url_for
from flask import session
from flask import jsonify
from relation_extraction.process_data import RE_DataEncoder_sp_based
import relation_extraction.predict as re_model

app = Flask(__name__)
app.secret_key = b"abc5b2db26e723e327dd0b30c377fb6ec709973aa3184b750e4e3e32adc22f9e"


@app.route("/", methods=["GET", "POST"])
def index():
    global sentence
    sentence_token = None
    if request.method == "POST":
        s = request.form["sentence"]
        if s:
            sentence = s
            sentence_token = re_model.sentence_token(sentence)
    return render_template("home.html", sentence_token=sentence_token)


@app.route("/predict")
def predict():
    e1pos = request.args.getlist("e1pos[]")
    e2pos = request.args.getlist("e2pos[]")
    result = re_model.predict(sentence, e1pos, e2pos)
    return jsonify(
        {"e1": result[0], "type": result[1], "e2": result[2], "acc": result[3]}
    )
    # return redirect(url_for('index'))


if __name__ == "__main__":
    sentence = ""
    app.run(debug=True)
