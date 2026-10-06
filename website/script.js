const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function (event) {

    event.preventDefault();

    const button = document.getElementById("predictButton");
    const resultBox = document.getElementById("result");
    const predictionText = document.getElementById("predictionText");
    const probabilityText = document.getElementById("probability");

    button.textContent = "Analysing Customer...";
    button.disabled = true;

    const data = {

        msisdn: Number(document.getElementById("msisdn").value),
        aon: Number(document.getElementById("aon").value),

        daily_decr30: Number(document.getElementById("daily_decr30").value),
        daily_decr90: Number(document.getElementById("daily_decr90").value),

        rental30: Number(document.getElementById("rental30").value),
        rental90: Number(document.getElementById("rental90").value),

        last_rech_date_ma: Number(
            document.getElementById("last_rech_date_ma").value
        ),

        last_rech_date_da: Number(
            document.getElementById("last_rech_date_da").value
        ),

        last_rech_amt_ma: Number(
            document.getElementById("last_rech_amt_ma").value
        ),

        cnt_ma_rech30: Number(
            document.getElementById("cnt_ma_rech30").value
        ),

        fr_ma_rech30: Number(
            document.getElementById("fr_ma_rech30").value
        ),

        sumamnt_ma_rech30: Number(
            document.getElementById("sumamnt_ma_rech30").value
        ),

        medianamnt_ma_rech30: Number(
            document.getElementById("medianamnt_ma_rech30").value
        ),

        medianmarechprebal30: Number(
            document.getElementById("medianmarechprebal30").value
        ),

        cnt_ma_rech90: Number(
            document.getElementById("cnt_ma_rech90").value
        ),

        fr_ma_rech90: Number(
            document.getElementById("fr_ma_rech90").value
        ),

        sumamnt_ma_rech90: Number(
            document.getElementById("sumamnt_ma_rech90").value
        ),

        medianamnt_ma_rech90: Number(
            document.getElementById("medianamnt_ma_rech90").value
        ),

        medianmarechprebal90: Number(
            document.getElementById("medianmarechprebal90").value
        ),

        cnt_da_rech30: Number(
            document.getElementById("cnt_da_rech30").value
        ),

        fr_da_rech30: Number(
            document.getElementById("fr_da_rech30").value
        ),

        cnt_da_rech90: Number(
            document.getElementById("cnt_da_rech90").value
        ),

        fr_da_rech90: Number(
            document.getElementById("fr_da_rech90").value
        ),

        cnt_loans30: Number(
            document.getElementById("cnt_loans30").value
        ),

        amnt_loans30: Number(
            document.getElementById("amnt_loans30").value
        ),

        maxamnt_loans30: Number(
            document.getElementById("maxamnt_loans30").value
        ),

        medianamnt_loans30: Number(
            document.getElementById("medianamnt_loans30").value
        ),

        cnt_loans90: Number(
            document.getElementById("cnt_loans90").value
        ),

        amnt_loans90: Number(
            document.getElementById("amnt_loans90").value
        ),

        maxamnt_loans90: Number(
            document.getElementById("maxamnt_loans90").value
        ),

        medianamnt_loans90: Number(
            document.getElementById("medianamnt_loans90").value
        ),

        payback30: Number(
            document.getElementById("payback30").value
        ),

        payback90: Number(
            document.getElementById("payback90").value
        ),

        pdate: document.getElementById("pdate").value
    };


    try {

        const response = await fetch(
            "http://127.0.0.1:5000/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(data)
            }
        );


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.error || "Prediction failed"
            );
        }


        // ====================================================
        // PROFESSIONAL AI RISK ASSESSMENT
        // ====================================================

        const probability =
            Number(result.repayment_probability);

        let riskLevel;
        let assessment;
        let recommendation;

        if (probability >= 80) {

            riskLevel = "LOW RISK";

            assessment =
                "Strong likelihood of repayment within the 5-day period.";

            recommendation =
                "Customer shows a strong repayment profile.";

        } else if (probability >= 50) {

            riskLevel = "MODERATE RISK";

            assessment =
                "Reasonable likelihood of repayment, with some uncertainty.";

            recommendation =
                "Customer may be considered with normal monitoring.";

        } else {

            riskLevel = "HIGH RISK";

            assessment =
                "Lower likelihood of repayment within the 5-day period.";

            recommendation =
                "Additional risk review is recommended before approval.";
        }


        // ====================================================
        // DISPLAY RESULT
        // ====================================================

        resultBox.classList.remove("hidden");

        predictionText.textContent =
            result.result;

        probabilityText.textContent =
            probability.toFixed(2) + "%";
            // Update probability circle



        // Add professional result information
        resultBox.innerHTML = `

            <div class="ai-result-header">
                <span class="result-label">
                    AI RISK ASSESSMENT
                </span>

                <span class="result-status">
                    ANALYSIS COMPLETE
                </span>
            </div>


            <div class="result-main">

                <div class="result-score">

                    <div class="score-value">
                        ${probability.toFixed(2)}%
                    </div>

                    <div class="score-title">
                        Repayment Probability
                    </div>

                </div>


                <div class="result-details">

                    <div class="prediction-result">
                        ${result.result}
                    </div>

                    <div class="risk-level">
                        ${riskLevel}
                    </div>

                    <p class="assessment-text">
                        ${assessment}
                    </p>

                </div>

            </div>


            <div class="probability-track">

                <div
                    class="probability-fill"
                    style="width: ${probability}%">
                </div>

            </div>
<!-- PROFESSIONAL RISK METER -->

            <div class="risk-meter">

                <div class="risk-meter-header">
                    <strong>Risk Assessment Scale</strong>

                    <span>
                        ${probability.toFixed(2)}% repayment probability
                    </span>
                </div>

                <div class="risk-scale">

                    <div
                        class="risk-marker"
                        style="left: ${probability}%">
                    </div>

                </div>

                <div class="risk-labels">

                    <span>LOW RISK</span>
                    <span>MODERATE RISK</span>
                    <span>HIGH RISK</span>

                </div>

            </div>

            <div class="result-recommendation">

                <span>AI Recommendation</span>

                <p>
                    ${recommendation}
                </p>

            </div>


            <div class="model-info">

                <span>MODEL</span>
                <strong>HistGradientBoosting</strong>

                <span>HORIZON</span>
                <strong>5-Day Repayment</strong>

            </div>
        `;


        // Scroll smoothly to result
        resultBox.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });


    } catch (error) {

        resultBox.classList.remove("hidden");

        resultBox.innerHTML = `

            <div class="error-result">

                <div class="error-title">
                    Prediction Error
                </div>

                <p>
                    ${error.message}
                </p>

                <small>
                    Please check the entered information
                    and try again.
                </small>

            </div>
        `;
    }


    button.textContent = "Predict Repayment";
    button.disabled = false;

});