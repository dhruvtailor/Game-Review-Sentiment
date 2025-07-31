from flask import Flask, request, render_template
import pickle
from scipy.sparse import hstack

# Load model and vectorizer
model = pickle.load(open("sgd_classifier_reviews.pkl", "rb"))
vectorizer_word = pickle.load(open("vectorizer_word.pkl", "rb"))
vectorizer_char = pickle.load(open("vectorizer_char.pkl","rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['text_input']

        # Apply both vectorizers to the new review
        features_word = vectorizer_word.transform([input_text])
        features_char = vectorizer_char.transform([input_text])

        # Combine both feature sets
        features_combined = hstack([features_char, features_word])

        prediction = model.predict(features_combined)[0]
        
        return render_template('index.html', prediction = prediction, input_text = input_text)

if __name__ == '__main__':
    #app.run(debug=True)
    app.run(debug=False, host='0.0.0.0', port=5000)
