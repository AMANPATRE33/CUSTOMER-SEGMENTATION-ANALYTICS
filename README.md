# CUSTOMER-SEGMENTATION-ANALYTICS

# CustomEdge: Enterprise ML Customer Segmentation & Predictive Analytics Engine

An end-to-end unsupervised machine learning pipeline and interactive production intelligence dashboard designed to group consumer behavior profiles into distinct corporate marketing personas. This platform enables marketing managers and product teams to transition from broad blast-marketing to hyper-targeted, ROI-optimized customer retention and operations strategies.

🎯 **Live Production Dashboard:** [Insert Your Streamlit App Link Here]

---

## 📊 Business Problem & Analytics Strategy

Mass marketing campaigns suffer from low conversion rates and inefficient ad spend because they treat highly diverse consumer bases as a single entity. 

By leveraging a dataset of over 2,200 active customers containing historical purchase histories, channel engagement patterns, and basic demographic dimensions, this project implements a mathematical clustering pipeline. The core engineering goal is to isolate low-frequency window shoppers from high-margin physical and digital VIP asset classes, providing automated, actionable operational strategies for each group.

---

## 🧠 Technical Architecture & Pipeline Engineering

The platform's data engine is built upon a rigid execution pipeline designed for enterprise consistency and zero-latency inference:

1. **Data Audit & Cleansing:** Handled missing structural elements (such as `Income` gaps) through clean row deletion rather than statistical imputation to prevent unnatural density spikes that bias unsupervised cluster distance centroids.
2. **Behavioral Feature Engineering:** Extracted dynamic, temporal metrics including customer `Age` and customer platform registration `Tenure` from raw dates. Aggregated disjointed multi-channel purchasing columns into unified behavioral variables (`Total_Spending` and tracking ratios).
3. **Feature Scaling (z-score Normalization):** Implemented `StandardScaler` to handle multi-unit feature spaces. This prevents distance metrics from heavily weighting large absolute numerical values (like Income) over low-range counts (like Web Visits).
4. **Hyperparameter Tuning via Inertia:** Iterated through multiple cluster allocations ($K = 2 \dots 10$) using the **Elbow Method** to minimize the Within-Cluster Sum of Squares (WCSS) and identify $K=6$ as the mathematically optimal grouping structure.
5. **High-Dimensional Variance Projection:** Deployed **PCA (Principal Component Analysis)** to reduce the 7-dimensional behavioral space into 2 principal orthogonal components for clean 2D spatial cluster visualization and overlap tracking.

---

## 👥 Identified Customer Personas & Go-To-Market Strategies

Through aggregation profiling, the K-Means engine isolated 6 completely distinct customer archetypes:

| Cluster | Identified Persona | Statistical Profile | Recommended Marketing Action |
| :---: | :--- | :--- | :--- |
| **0** | **Older Affluent In-Store VIPs** | High Income, High Spending, High Store Count, Minimal Web Browsing. | High-end physical direct mailers, dedicated priority managers, invitations to exclusive in-store events. |
| **1** | **At-Risk Digital Browsers** | Low Income, Low Spending, Maximum Web Views, Critically High Recency ($70+$ Days). | **High Retention Risk.** Trigger low-cost automated email cart-recovery hooks. Avoid expensive physical print costs. |
| **2** | **Tech-Savvy Power Users** | Mid-High Income, High Spending, Leading E-Commerce Purchasing Cadence. | Core online engine. Push app-exclusive reward accelerators and early-access platform inventory drops. |
| **3** | **Aging Low-Engagement Shoppers** | Seniors, Restricted Incomes, Low Spending Volumes, Heavy Brand Disengagement. | Focus on volume bundle promotions and standard household necessities through low-overhead baseline communication. |
| **4** | **Prime-Age Elite High-Spenders** | Peak Working/Earning Age, Maximum Income, Maximum Spending Volume. | **Crown Jewel Cohort.** Offer complimentary express logistics, white-glove delivery, and luxury cross-promotions. |
| **5** | **Highly Active Deal-Hunters** | Price-Sensitive, High Web Visits, Leading Freshness Matrix ($<20$ Day Recency). | **Largest Active Segment.** Deploy high-frequency digital flash sales, cart threshold drops, and viral peer referral campaigns. |

---

## 💻 UI/UX Implementation Details

The web dashboard is built using Streamlit and styled with a custom-engineered CSS presentation layer to reflect a corporate product interface:
* **Memory Optimization:** Uses `@st.cache_resource` compilation to pin serialized `joblib` artifacts directly inside memory, bypassing heavy disk I/O reads on every single slider interaction.
* **Semantic UI Alerting:** Injects real-time conditional inline CSS to completely alter the output presentation panel's hue based on the predicted segment risk profile (e.g., Warning Reds for Churn Alerts vs. Regal Purples for Premium Assets).
* **Accessibility Overrides:** Avoids non-standard raw HTML widgets inside the input workspace to allow full compatibility with native system browser dark/light mode toggling.

---

## 🛠️ Local Development & Replication Blueprint

To boot this project inside a local Python environment, run the following commands:

```bash
# 1. Clone the repository workspace
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# 2. Initialize necessary system libraries
pip install -r requirements.txt

# 3. Boot up the local web-app framework instance
streamlit run segmentation.py
