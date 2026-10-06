from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load("final_microfinance_repayment_model.pkl")


# ============================================================
# HOME / TEST ROUTE
# ============================================================

@app.route("/")
def home():
    return jsonify({
        "message": "Microfinance Loan Repayment Prediction API is running",
        "status": "success"
    })


# ============================================================
# PREDICTION ROUTE
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    try:
        # Get data sent from website
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No input data received"
            }), 400

        # ----------------------------------------------------
        # Create DataFrame from user input
        # ----------------------------------------------------

        df = pd.DataFrame([data])

        # ----------------------------------------------------
        # Convert pdate into datetime
        # Same processing as main.py
        # ----------------------------------------------------

        df["pdate"] = pd.to_datetime(
            df["pdate"],
            format="%d-%m-%Y"
        )

        # ----------------------------------------------------
        # FEATURE ENGINEERING
        # Same features used during model training
        # ----------------------------------------------------

        # 1. Loan amount growth
        df["loan_amount_growth"] = (
            df["amnt_loans90"] - df["amnt_loans30"]
        )

        # 2. Loan count growth
        df["loan_count_growth"] = (
            df["cnt_loans90"] - df["cnt_loans30"]
        )

        # 3. Recharge amount growth
        df["recharge_amount_growth"] = (
            df["sumamnt_ma_rech90"] -
            df["sumamnt_ma_rech30"]
        )

        # 4. Recharge count growth
        df["recharge_count_growth"] = (
            df["cnt_ma_rech90"] -
            df["cnt_ma_rech30"]
        )

        # 5. Average loan amount - 30 days
        df["avg_loan_amount30"] = (
            df["amnt_loans30"] /
            (df["cnt_loans30"] + 1)
        )

        # 6. Average loan amount - 90 days
        df["avg_loan_amount90"] = (
            df["amnt_loans90"] /
            (df["cnt_loans90"] + 1)
        )

        # 7. Average recharge amount - 30 days
        df["avg_recharge_amount30"] = (
            df["sumamnt_ma_rech30"] /
            (df["cnt_ma_rech30"] + 1)
        )

        # 8. Average recharge amount - 90 days
        df["avg_recharge_amount90"] = (
            df["sumamnt_ma_rech90"] /
            (df["cnt_ma_rech90"] + 1)
        )

        # 9. Loan / recharge ratio
        df["loan_recharge_ratio30"] = (
            df["amnt_loans30"] /
            (df["sumamnt_ma_rech30"] + 1)
        )

        # 10. Payback difference
        df["payback_difference"] = (
            df["payback90"] - df["payback30"]
        )

        # ----------------------------------------------------
        # CREATE DATE FEATURES
        # Same processing as main.py
        # ----------------------------------------------------

        df["pdate_day"] = df["pdate"].dt.day
        df["pdate_month"] = df["pdate"].dt.month
        df["pdate_dayofweek"] = df["pdate"].dt.dayofweek

        # ----------------------------------------------------
        # DROP COLUMNS
        # Same processing as main.py
        # ----------------------------------------------------

        df = df.drop(
            columns=["msisdn", "pdate"],
            errors="ignore"
        )

        # ----------------------------------------------------
        # MAKE PREDICTION
        # ----------------------------------------------------

        probability = model.predict_proba(df)[0][1]

        prediction = 1 if probability >= 0.5 else 0

        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if prediction == 1:
            result = "Likely to Repay"
        else:
            result = "Likely to Default"

        return jsonify({
            "prediction": prediction,
            "repayment_probability": round(
                probability * 100, 2
            ),
            "result": result
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN FLASK SERVER
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=int(os.environ.get("PORT",5000))
    )