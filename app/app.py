"""Flask application placeholder for the future prediction API."""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.get("/")
def index():
    return jsonify(
        {
            "project": "Credit Card Fraud Detection System",
            "status": "API scaffold ready. No model has been trained yet.",
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
