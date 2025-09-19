from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("loan_model.pkl", "rb"))

@app.route("/")
def home():
    return "Loan Prediction API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Example JSON: {"Gender":1,"Married":1,"ApplicantIncome":5000,"CoapplicantIncome":2000,"LoanAmount":150,"Credit_History":1}
        features = np.array([[ 
            float(data["Gender"]),
            float(data["Married"]),
            float(data["ApplicantIncome"]),
            float(data["CoapplicantIncome"]),
            float(data["LoanAmount"]),
            float(data["Credit_History"])
        ]])

        prediction = model.predict(features)[0]

        result = "Approved" if prediction == 1 else "Rejected"
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
