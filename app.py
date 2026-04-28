from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["text"]
    data = vectorizer.transform([text])
    prediction = model.predict(data)[0]
    prob = model.predict_proba(data).max()
    return render_template("index.html", prediction=prediction, prob=round(prob,2))

if __name__ == "__main__":
    app.run(debug=True)