import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

const DEFAULT_FORM = {
  amt: "",
  merchant: "",
  category: "",
  gender: "",
  state: "",
  city_pop: "",
  transaction_date: "2020-06-21",
  hour: "",
  age: "",
};

const MODEL_FEATURE_LABELS = {
  merchant: "Merchant code",
  category: "Category code",
  amt: "Transaction amount",
  gender: "Gender code",
  state: "State code",
  city_pop: "City population",
  hour: "Transaction hour",
  day_of_week: "Day of week",
  month: "Month",
  day: "Day",
  is_weekend: "Weekend flag",
  age: "Age",
  merchant_distance_km: "Merchant distance",
  previous_amount: "Previous amount",
  previous_transaction_count: "Previous transaction count",
  historical_avg_amount: "Historical average amount",
  time_since_previous_hours: "Hours since previous transaction",
  amount_deviation: "Amount deviation",
};

function formatPercent(value) {
  const percent = Number(value) * 100;

  if (percent > 0 && percent < 0.01) {
    return `${percent.toFixed(6)}%`;
  }

  return `${percent.toFixed(2)}%`;
}

function formatNumber(value) {
  const number = Number(value);

  if (!Number.isFinite(number)) {
    return "";
  }

  if (Math.abs(number) >= 1000) {
    return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
  }

  return number.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

function riskClass(risk) {
  return String(risk || "").toLowerCase();
}

function App() {
  const [form, setForm] = useState(DEFAULT_FORM);
  const [options, setOptions] = useState({
    categories: [],
    genders: [],
    states: [],
    merchants: [],
    threshold: 0.25,
  });
  const [sampleFeatures, setSampleFeatures] = useState(null);
  const [sampleLabel, setSampleLabel] = useState("");
  const [actualLabel, setActualLabel] = useState(null);
  const [result, setResult] = useState(null);
  const [apiStatus, setApiStatus] = useState("Checking API");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const isComplete = useMemo(
    () =>
      ["amt", "merchant", "category", "gender", "state", "city_pop", "transaction_date", "hour", "age"].every(
        (key) => String(form[key] ?? "").trim() !== "",
      ),
    [form],
  );

  useEffect(() => {
    fetch(`${API_BASE_URL}/`)
      .then((response) => response.json())
      .then((data) => setApiStatus(`${data.model} ready`))
      .catch(() => setApiStatus("API offline"));

    fetch(`${API_BASE_URL}/input-options`)
      .then((response) => response.json())
      .then((data) => setOptions(data))
      .catch(() => setError("Unable to load input options from the API."));
  }, []);

  function updateValue(key, value) {
    setForm((current) => ({ ...current, [key]: value }));
    setSampleFeatures(null);
    setSampleLabel("");
    setActualLabel(null);
  }

  async function loadSample(endpoint, label) {
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/${endpoint}`);
      if (!response.ok) {
        throw new Error(`Unable to load ${label.toLowerCase()}.`);
      }

      const data = await response.json();
      setForm({
        ...DEFAULT_FORM,
        ...Object.fromEntries(
          Object.entries(data.user_input).map(([key, value]) => [key, String(value)]),
        ),
      });
      setSampleFeatures(data.transaction);
      setSampleLabel(`${label}: ${data.actual_label_text} test dataset row`);
      setActualLabel(data.actual_label_text);
    } catch (err) {
      setError(err.message);
    }
  }

  async function submitPrediction(event) {
    event.preventDefault();
    setError("");
    setLoading(true);
    setResult(null);

    try {
      const endpoint = sampleFeatures ? "predict-demo" : "predict-user";
      const payload = sampleFeatures
        ? sampleFeatures
        : {
            ...form,
            amt: Number(form.amt),
            city_pop: Number(form.city_pop),
            hour: Number(form.hour),
            age: Number(form.age),
          };

      const response = await fetch(`${API_BASE_URL}/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(typeof data.detail === "string" ? data.detail : "Prediction request failed.");
      }

      setResult(await response.json());
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main>
      <header className="topbar">
        <div>
          <p className="eyebrow">XGBoost model risk estimate</p>
          <h1>Credit Card Fraud Detection</h1>
        </div>
        <div className="status">
          <strong>{apiStatus}</strong>
          <span>Classification threshold: {formatPercent(options.threshold)}</span>
        </div>
      </header>

      <section className="workspace">
        <form className="input-panel" onSubmit={submitPrediction}>
          <div className="panel-heading">
            <div>
              <h2>User Input</h2>
              <p>Enter readable transaction details. Dataset samples use real processed test rows for demonstration.</p>
            </div>
          </div>

          <div className="demo-actions">
            <button type="button" className="secondary" onClick={() => loadSample("sample-transaction", "Load Sample")}>
              Load Sample
            </button>
            <button
              type="button"
              className="secondary"
              onClick={() => loadSample("sample-legitimate-transaction", "Legitimate Sample")}
            >
              Load Legitimate Sample
            </button>
            <button
              type="button"
              className="secondary danger"
              onClick={() => loadSample("sample-fraud-transaction", "Fraud Sample")}
            >
              Load Fraud Sample
            </button>
          </div>

          {sampleLabel ? <p className="sample-note">{sampleLabel}</p> : null}

          <div className="user-grid">
            <label>
              <span>Transaction Amount</span>
              <input
                type="number"
                step="any"
                min="0"
                value={form.amt}
                onChange={(event) => updateValue("amt", event.target.value)}
              />
            </label>

            <label>
              <span>Merchant</span>
              <input
                list="merchant-options"
                value={form.merchant}
                onChange={(event) => updateValue("merchant", event.target.value)}
              />
              <datalist id="merchant-options">
                {options.merchants.map((merchant) => (
                  <option key={merchant} value={merchant} />
                ))}
              </datalist>
            </label>

            <label>
              <span>Category</span>
              <select value={form.category} onChange={(event) => updateValue("category", event.target.value)}>
                <option value="">Select category</option>
                {options.categories.map((category) => (
                  <option key={category} value={category}>
                    {category}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>Gender</span>
              <select value={form.gender} onChange={(event) => updateValue("gender", event.target.value)}>
                <option value="">Select gender</option>
                {options.genders.map((gender) => (
                  <option key={gender} value={gender}>
                    {gender}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>State</span>
              <select value={form.state} onChange={(event) => updateValue("state", event.target.value)}>
                <option value="">Select state</option>
                {options.states.map((state) => (
                  <option key={state} value={state}>
                    {state}
                  </option>
                ))}
              </select>
            </label>

            <label>
              <span>City Population</span>
              <input
                type="number"
                min="0"
                value={form.city_pop}
                onChange={(event) => updateValue("city_pop", event.target.value)}
              />
            </label>

            <label>
              <span>Transaction Date</span>
              <input
                type="date"
                value={form.transaction_date}
                onChange={(event) => updateValue("transaction_date", event.target.value)}
              />
            </label>

            <label>
              <span>Transaction Hour</span>
              <input
                type="number"
                min="0"
                max="23"
                value={form.hour}
                onChange={(event) => updateValue("hour", event.target.value)}
              />
            </label>

            <label>
              <span>Age</span>
              <input
                type="number"
                step="any"
                min="18"
                max="100"
                value={form.age}
                onChange={(event) => updateValue("age", event.target.value)}
              />
            </label>
          </div>

          {!sampleFeatures ? (
            <p className="demo-mode">
              Manual entries run in demo/test mode. The backend uses training-set medians for unavailable
              cardholder history and merchant-distance features instead of pretending real history exists.
            </p>
          ) : null}

          <button className="primary" type="submit" disabled={!isComplete || loading}>
            {loading ? "Scoring transaction..." : "Predict Risk"}
          </button>

          {error ? <p className="error">{error}</p> : null}
        </form>

        <section className="results-panel" aria-live="polite">
          {result ? (
            <>
              <div className="result-summary">
                <div>
                  <span>Fraud Probability</span>
                  <strong>{formatPercent(result.fraud_probability)}</strong>
                </div>
                <div className={`risk-card ${riskClass(result.risk_level)}`}>
                  <span>Risk Level</span>
                  <strong>{result.risk_level}</strong>
                </div>
                <div>
                  <span>Prediction</span>
                  <strong>{result.prediction_label}</strong>
                </div>
              </div>

              {actualLabel ? (
                <p className="sample-note">Actual dataset label for this demo sample: {actualLabel}</p>
              ) : null}

              <div className="explanation">
                <h2>Automated ML Explanation Summary</h2>
                <p>{result.explanation}</p>
                <p className="note">
                  This is a model prediction and application-defined risk estimate, not proof that fraud occurred.
                </p>
              </div>

              <div className="factors">
                <h2>Important Contributing Factors</h2>
                <div className="factor-list">
                  {result.top_factors.map((factor) => (
                    <article key={factor.feature} className="factor">
                      <div>
                        <strong>{factor.feature_label}</strong>
                        <span>{factor.impact}</span>
                      </div>
                    </article>
                  ))}
                </div>
              </div>

              <details className="technical">
                <summary>Technical Explanation</summary>
                <p>{result.feature_source}</p>
                <div className="technical-grid">
                  {result.top_factors.map((factor) => (
                    <div key={factor.feature}>
                      <span>{factor.feature}</span>
                      <strong>SHAP {Number(factor.shap_value).toFixed(4)}</strong>
                      <em>Value {formatNumber(factor.feature_value)}</em>
                    </div>
                  ))}
                </div>
              </details>

              <details className="technical">
                <summary>Advanced / Debugging Mode - internal model features</summary>
                <div className="feature-table">
                  {Object.entries(result.model_features).map(([feature, value]) => (
                    <div key={feature}>
                      <span>{MODEL_FEATURE_LABELS[feature] || feature}</span>
                      <strong>{formatNumber(value)}</strong>
                    </div>
                  ))}
                </div>
              </details>
            </>
          ) : (
            <div className="empty-state">
              <h2>No prediction yet</h2>
              <p>Load a demo sample or enter transaction details to view model probability, risk, and explanation.</p>
            </div>
          )}
        </section>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
