from flask import Flask, render_template, request, jsonify
app = Flask(__name__)
import pickle


# Load the saved vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

def predict_spam(text):
    # Transform the input text using the loaded vectorizer
    vector_input = tfidf.transform([text])
    # Predict using the loaded model
    result = model.predict(vector_input)[0]
    return "Spam" if result == 1 else "Not Spam"

# Example usage
message = "Congratulations! You've won a free prize. Click here to claim."
print(predict_spam(message))
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    msg = request.json['message']
    print("Prediction: ",msg)
    
    # TODO: Add your spam classification logic here
    # For now, returning a placeholder response
    return jsonify({
        'prediction': predict_spam(msg),  # Replace with actual prediction
        'message': msg
    })


if __name__ == '__main__':
    app.run(debug=True)
