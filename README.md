# ✈️ AirLoyalty Intelligence Platform

An end-to-end **airline loyalty analytics and customer retention platform** that combines feature engineering, machine learning, customer segmentation, churn-risk scoring, and actionable retention recommendations in an interactive **Streamlit dashboard**.

The platform is designed around approximately **16,737 Canadian loyalty members**, with flight activity data spanning **2012–2018** and a **September 2017 cutoff** used in the analytics workflow.

## 🚀 Project Overview

Airline loyalty programs generate large volumes of behavioral data, but identifying which customers are at risk of disengaging — and deciding what action to take — can be difficult.

This project turns loyalty and flight-activity data into a decision-support system that answers:

- Which customers are most likely to churn?
- Which high-value customers are at risk?
- How do customers differ behaviorally?
- What retention action should be prioritized?
- What is the recommended timing and communication channel?
- What does an individual customer's loyalty profile look like?

## 🧩 Solution Architecture

```text
Raw Loyalty Data + Flight Activity
                │
                ▼
        Data Loading & Cleaning
                │
                ▼
        Feature Engineering
                │
                ▼
        ┌───────────────────────┐
        │   Churn Prediction    │
        │   Random Forest       │
        └───────────┬───────────┘
                    │
                    ▼
        Churn Probability / Risk Tier
                    │
                    ▼
        ┌───────────────────────┐
        │ Customer Segmentation │
        │       K-Means         │
        └───────────┬───────────┘
                    │
                    ▼
          Behavioral Segments
                    │
                    ▼
        Retention Rule Engine
                    │
                    ▼
     Recommended Retention Actions
                    │
                    ▼
          Streamlit Dashboard
```

The pipeline runs the stages of loading/cleaning, feature engineering, churn modeling, segmentation, retention rules, and final dataset generation before the dashboard consumes the scored data.

## 📊 Dashboard Modules

### 1. 🏠 Executive Overview

Provides a high-level view of the loyalty base with:

- Total members
- Active members
- Churn rate
- Average customer lifetime value (CLV)
- Critical-risk customers
- Revenue at risk
- CLV distribution by loyalty card tier
- Churn-risk distribution
- Customer segment breakdown
- Churn risk by province
- Network-wide monthly flight activity

### 2. ⚠️ Churn Risk Dashboard

Designed for identifying customers who need attention.

Features include:

- Risk-tier filtering
- Segment filtering
- Loyalty-card filtering
- Top-N high-risk customer selection
- Churn probability vs. CLV analysis
- Risk × card-tier heatmap
- High-risk customer table
- Downloadable filtered customer list

The dashboard prioritizes **Critical** and **High** risk customers by default.

### 3. 👥 Customer Segments

Groups customers into five behavioral segments:

| Segment | Interpretation |
|---|---|
| 🏆 Champions | Highest-value / strongest engagement group |
| 💎 Loyalists | Consistently engaged loyalty members |
| ⚠️ At-Risk Actives | Currently engaged but showing churn risk |
| ✈️ Occasional Flyers | Lower-frequency customers |
| 💤 Dormant | Customers with weak or declining engagement |

The dashboard provides:

- Segment-level member counts
- Average and median CLV
- Churn rates
- Flight activity
- Tenure
- PCA 2D segment map
- CLV box plots
- Behavioral radar analysis
- Segment comparison table

### 4. 🎯 Retention Actions

Converts analytical outputs into customer-level actions.

Users can filter by:

- Segment
- Risk tier
- Priority level

Each action card contains:

- Customer ID
- Customer segment
- Risk level
- Recommended action
- Action details
- Timing
- Communication channel
- CLV
- Churn probability

A downloadable **Retention Action Plan CSV** is also provided.

### 5. 🔍 Individual Customer Lookup

Provides a detailed customer profile based on loyalty number.

The profile includes:

- Location
- Loyalty card tier
- Demographics
- Salary, where available
- CLV
- Enrollment information
- Tenure
- Churn probability
- Risk tier
- Total flights
- 12-month flights
- Days since last flight
- Points balance
- Flight-history trend
- Recommended retention action

## 🤖 Machine Learning

### Churn Prediction

The analytics pipeline trains a **Random Forest** churn model and generates a churn probability for every scored customer.

The pipeline reports:

- AUC-ROC
- 5-fold cross-validation AUC
- Feature importance

Customers are subsequently assigned to risk tiers:

- Critical
- High
- Medium
- Low

### Customer Segmentation

The project uses **K-Means clustering** to identify behaviorally distinct customer groups.

The segmentation stage also reports a **silhouette score** and produces segment-level profiles.

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Model Persistence / Utilities | Joblib |
| Output Data | Parquet |
| Environment | Python 3.x |

The project requirements specify Streamlit, Pandas, NumPy, Scikit-learn, Plotly, Matplotlib, Seaborn, and Joblib.

## 📁 Project Structure

```text
.
├── app.py
├── run_pipeline.py
├── requirements.txt
├── Customer Flight Activity.csv
│
├── analytics/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── churn_model.py
│   ├── segmentation.py
│   ├── retention.py
│   └── scored_customers.parquet
│
└── README.md
```

> The dashboard expects `analytics/scored_customers.parquet` to exist after the analytics pipeline has completed.

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**
```bash
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the analytics pipeline

Run the pipeline before opening the dashboard:

```bash
python run_pipeline.py
```

The pipeline:

1. Loads and cleans the source data
2. Engineers analytical features
3. Trains the churn model
4. Scores customers
5. Trains the segmentation model
6. Assigns customer segments
7. Applies retention rules
8. Saves the final scored dataset to:

```text
analytics/scored_customers.parquet
```

### 5. Launch the Streamlit application

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔄 Pipeline Output

The pipeline prints a summary containing:

- Total customers
- Churn rate
- AUC-ROC
- 5-fold CV AUC
- Silhouette score
- Segment summary
- Risk-tier distribution
- Top feature importances

This makes the pipeline useful both as a production-style scoring workflow and as an evaluation layer for the analytical models.

## 📥 Dashboard Exports

The application provides downloadable CSV outputs for:

### High-Risk Customers

```text
churn_risk_customers.csv
```

Contains the currently filtered customer population from the churn-risk dashboard.

### Retention Action Plan

```text
retention_actions.csv
```

Contains prioritized customers and their recommended retention actions.

## 🎯 Business Value

The platform connects machine-learning outputs directly to customer-retention decisions.

Instead of stopping at a churn probability, the workflow moves from:

**Customer Data → Risk Prediction → Behavioral Segment → Business Action**

This supports:

- Proactive churn management
- High-value customer prioritization
- Customer-level retention planning
- Segment-specific engagement strategies
- Revenue-at-risk identification
- Operationally usable action lists

## 📌 Key Design Decisions

### Temporal cutoff

The application identifies the analytical cutoff as **September 2017**, helping structure the behavioral modeling workflow around historical customer activity.

### Action prioritization

Retention recommendations are ranked using action priorities, allowing teams to focus on **Priority 1 and 2** customers for immediate intervention.

### Risk × Value thinking

The dashboard combines churn probability with **Customer Lifetime Value (CLV)** so that retention decisions are not based on churn probability alone.

## ⚠️ Notes & Limitations

- The dashboard depends on the analytics modules and source data being present in the expected project structure.
- `scored_customers.parquet` is generated by the pipeline and should not be treated as the primary raw dataset.
- Model metrics such as AUC-ROC and silhouette score are generated at runtime by the pipeline.
- Retention actions are rule-based outputs applied after churn scoring and segmentation.
- Customer activity and source-data availability determine the quality of the final recommendations.

## 🧪 Example Workflow

```bash
# Install dependencies
pip install -r requirements.txt

# Build analytical dataset and models
python run_pipeline.py

# Launch dashboard
streamlit run app.py
```

## 📈 Future Enhancements

Potential extensions include:

- Automated model retraining
- Experimentation / A-B testing for retention actions
- Cost-sensitive retention optimization
- SHAP-based model explainability
- Campaign-response tracking
- Real-time or scheduled scoring
- Integration with CRM systems
- Monitoring of model drift and segment stability

## 👨‍💻 Author

**AirLoyalty Intelligence Platform**

Built as an end-to-end analytics project combining **Python, machine learning, customer segmentation, and Streamlit-based business intelligence**.
