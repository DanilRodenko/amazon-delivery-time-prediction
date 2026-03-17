# 📦 Amazon Delivery Time Analysis & Prediction

> End-to-end data science project: EDA, geoanalysis, feature engineering, and ML modelling to predict and classify delivery times.

---

## 📌 Problem

Delivery time is one of the most critical metrics in e-commerce logistics. This project explores what factors drive delivery performance and builds models to predict how long a delivery will take — both as a regression (exact minutes) and a classification (Fast / Medium / Slow).

*Personal motivation: built this while working as a delivery driver at Amazon. Wanted to understand the data behind the job.*

---

## 📊 Dataset

- **Source:** Kaggle — Amazon Delivery dataset (India)
- **Features:** agent age & rating, store/drop coordinates, order date & time, pickup time, weather, traffic, area type, vehicle type, product category

---

## 🔍 Key Findings (EDA)

**Delivery Time Distribution**
- Median delivery time: **~125 minutes (~2 hours)**
- 50% of deliveries fall between **90–160 minutes**
- Range: 10 min (fastest) to 270 min (slowest)

**What slows deliveries down:**
- **Weather:** Foggy & Cloudy conditions add ~35–40 min vs Sunny days
- **Traffic:** Jam conditions push median time above 150 min vs ~100 min in low traffic
- **Area:** Semi-Urban deliveries average ~239 min — nearly double Metropolitan areas
- **Agent Rating:** Higher-rated agents deliver measurably faster (r ≈ -0.29)

**Geoanalysis:**
- Most deliveries cover < 20 km (local city logistics)
- Distance correlates with delivery time, but weakly — traffic and weather matter more
- Built interactive route map with Folium (store → drop locations)

---

## 🛠 Feature Engineering

Created new features from raw timestamps and coordinates:
- `Distance_km` — Haversine formula from store/drop lat-lon coordinates
- `DayOfWeek`, `Hour` — extracted from order datetime
- `Prep_Time_Min` — time between order placement and pickup
- `Agent_AgeGroup` — binned age categories (Young / Middle / Senior)

---

## 🤖 Models & Results

### Regression (predict exact delivery time in minutes)

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | ~28 min | ~35 min | ~0.45 |
| Random Forest | ~20 min | ~26 min | ~0.68 |
| **Gradient Boosting** | **~19 min** | **~25 min** | **~0.70** |

Top features by importance: product category (Grocery anomaly), agent rating, distance, prep time.

### Classification (Fast ≤120 min / Medium 120–150 min / Slow >150 min)

| Model | Accuracy |
|---|---|
| Logistic Regression | ~70% |
| **Random Forest** | **~78%** |

Random Forest performs well on Fast (recall 0.90) and Slow (recall 0.82) classes. Medium class remains the hardest to distinguish — a common challenge with borderline categories.

---

## 💡 Business Insights

- **Semi-Urban areas** are the biggest logistics bottleneck (~239 min avg) — need dedicated routing strategy
- **Weather-aware scheduling** could reduce delays — fog/cloud conditions reliably increase delivery time
- **Agent rating is actionable:** investing in courier training/incentives has measurable impact on delivery speed
- **Peak hours** show clear patterns that can inform dynamic staffing decisions

---

## 🛠 Tech Stack

- Python: pandas, NumPy, scikit-learn, Matplotlib, Seaborn
- Geospatial: Folium (interactive map), Haversine distance calculation
- Models: LinearRegression, RandomForestRegressor, GradientBoostingRegressor, LogisticRegression, RandomForestClassifier
- Pipeline: sklearn Pipeline + ColumnTransformer for clean preprocessing

---

## ▶️ How to Run

```bash
git clone https://github.com/DanilRodenko/amazon-delivery-time-prediction
cd amazon-delivery-time-prediction
pip install -r requirements.txt
jupyter notebook delivery.ipynb
```

Download the dataset from [Kaggle](https://www.kaggle.com) and place `amazon_delivery.csv` in the project root.

---

## 👤 Author

**Danil Rodenko** — Data Scientist  
📍 Limerick, Ireland  
🔗 [github.com/DanilRodenko](https://github.com/DanilRodenko) | [danil.rodenko@gmail.com](mailto:danil.rodenko@gmail.com)
