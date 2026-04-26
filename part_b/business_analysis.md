# **Promotion Effectiveness at a Fashion Retail Chain**



### **B1. Problem Formulation**



**(a) Kind of ML problem**

* The **target variable** is **'Number of items sold'** as this is what the retailer wants to optimize for each of the store.
* Candidate input features will be store\_id, type of promotion, store size, store location (urban, semi-urban, and rural), monthly footfall, local competition density, customer demographics such as age, gender, average bucket size, and time-related features such as transaction\_date, month, quarter, year, is\_weekend, is\_festival, etc.
* This would be a **supervised regression** problem as the target variable i.e. **'Number of items sold**' is a continuous numerical value and other input features are available to us. Here, the model learns to predict expected items sold for each of the five promotion types, and the recommended promotion is the one yielding the highest predicted value.



**(b) Justification for the target variable**

* **Items sold is more reliable** because it isn’t affected by promotions or pricing changes, so it shows true customer demand. It’s easy to track in transaction records without worrying about discounts, free items, etc. Most importantly, it **matches the business goal of selling more units**, rather than focusing on revenue or profit, which will require different approaches.
* This denotes the **principle of avoiding proxy targets** that the treatment itself can distort. In real-world ML, the target should directly reflect what the business wants to change and not be mechanically affected by the interventions such as discounts, etc.



**(c) One single global model or not**

* A **single global model isn’t suitable** because it assumes all 50 stores react to promotions in the same way, which is unlikely - different stores (like rural vs. busy urban ones) have very different customer behavior. Instead, **group similar stores into a few clusters** based on traits like location, footfall, and competition, then **train one model per group** so each model reflects shared patterns without overfitting.
* Another option is to use models like **Random Forest or Gradient Boosting with store-specific features**, allowing one model to still capture differences between stores.
* **These approaches strikes a balance**: Separate models per store would have too little data and be unreliable, while one global model would ignore important differences; clustering or mixed methods sit in between and work best.



### **B2. Data and EDA Strategy**



**(a) Data joining and dataset design**

* The data is provided in four tables: Transactions, Store attributes, Promotion details and Calendar data
* Join transactions with store attributes using store\_id
* Join promotion details using promotion identifiers or transaction dates
* Join calendar data using transaction\_date
* **Grain of final dataset:** One row per store per month
* **Aggregations:** Total items sold per store per month, average basket size, promotion applied during the month, festival and weekend proportion



**(b) Exploratory Data Analysis (EDA)**

* We will firstly check for any **missing data** and see if those are to be handled before training
* **Promotion vs Sales (Bar Chart):** Compare the average number of items sold across different promotion types to determine which promotions perform best
* **Sales Trends Over Time (Line Chart):** Examine monthly sales data to identify patterns, seasonal effects, and overall trends
* **Correlation Heatmap:** Discover variables that strongly influence items\_sold, helping inform feature selection and model development
* **Sales Distribution (Histogram / Boxplot):** Assess the distribution of sales data to identify skewness and outliers that may need transformation or robust modeling approaches



**(c) Managing imbalances**

* Since the majority (80%) of transactions happen without promotions, the model **may develop a bias** toward non-promotion scenarios and its ability to accurately capture the effects of promotions may be limited
* To fix this we can: a. Use **resampling methods** such as oversampling or undersampling, b. Add a **binary feature** to indicate whether a promotion is present, c. Apply **weighting techniques** during model training. d. **Assess model performance separately** for promotion and non-promotion cases



### **B3. Model Evaluation and Deployment**



**(a) Train-Test Split and Metrics**

* With three years of monthly data, using a **random split is inappropriate**. For example, the model could be trained on March 2024 data and tested on January 2023 data, which effectively **exposes it to future information (data leakages)**
* To avoid this, a **time-based split should be used**: train the model on the first two years and test it on the most recent months. This approach reflects real-world conditions, where the model predicts future outcomes based only on past data
* **Evaluation metrics:** 1. **RMSE (Root Mean Squared Error)**: Penalizes large prediction errors more heavily, which is important when large forecasting mistakes can lead to inventory issues, 2. **MAE (Mean Absolute Error)**: Measures average prediction error and provides an easily interpretable metric for business stakeholders
* **Interpretation:** 1. **Lower RMSE** indicates fewer large prediction errors, 2. **Lower MAE** indicates consistent prediction accuracy across stores



**(b) Explaining Model Recommendations:** To understand why the same store receives different promotion recommendations across months:



* Use feature importance from models like Random Forest or tools like SHAP which tell us which features most influenced a specific prediction.
* Examine the influence of factors such as seasonality (month), festival indicators or past sales patterns
* Example:

&#x20;    **December → Higher demand → Loyalty Points Bonus encourages repeat purchases**

&#x20;    **March → Lower demand → Flat Discount appeals to price-sensitive customers**



**(c) End-to-End Deployment Process**


* **Step 1 — Building and Saving the Model**

We will build a neural network in **TensorFlow/Keras** that takes a store's features as input and outputs predicted items sold for each of the 5 promotions. Once trained, we save it using TensorFlow's **SavedModel format** — this bundles the model architecture, weights, and preprocessing steps all together into one folder e.g. pythonmodel.save('promotion\_model/v1'). This saved folder can be loaded later at any time without retraining, exactly like loading a saved file.



* **Step 2 — Monthly Batch Inference (No Retraining)**

On the 1st of each month, a script loads the saved model and runs predictions for all 50 stores: The output is a simple table — one row per store, one column saying which promotion to run. This gets pushed to the marketing dashboard. No retraining happens here.



* **Step 4 — Monitoring with TensorFlow Extended (TFX)**

TensorFlow has a built-in monitoring toolkit called **TFX (TensorFlow Extended)**. Two components matter most here:

**a. TensorFlow Data Validation (TFDV)** — checks whether this month's input data looks similar to what the model was trained on. If footfall distributions or competition levels have shifted significantly, it raises a flag before predictions are even made.

**b. TensorFlow Model Analysis (TFMA)** — after actual sales data comes in the following month, this compares what the model predicted vs what actually happened, and tracks whether errors are growing over time. If errors consistently exceed a threshold (say, MAE grows more than 50% above the original training MAE), an alert is triggered and the team knows it's time to retrain.



### **Final Conclusion**



**This approach allows the company to base its promotion decisions on data, improving sales outcomes while remaining responsive to regional variations, customer behavior, and seasonal trends**

