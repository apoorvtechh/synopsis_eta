import streamlit as st
import pandas as pd

# ============================================================
# 🧭 SIDEBAR NAVIGATION
# ============================================================
st.set_page_config(layout="wide")
st.sidebar.title("🧭 Navigation")

section = st.sidebar.radio(
    "Go to section:",
    [
        "🏠 Project Overview",
        "📊 Dataset + EDA",
        "🧪 Models Experimentation",
        "🎯 Hyperparameter Tuning",
        "🤖 Final Model & Metrics",
        "🔮 DVC Pipeline, CI/CD Automation",
        "🧱 Tech Stack & Pipeline"
    ]
)

# ============================================================
# SECTION FUNCTIONS
# ============================================================

def section_overview():
    st.title("🏠 Project Overview")
    st.write("Add your project intro here...")


# 🌟 COMBINED DATASET + EDA SECTION (BEAUTIFIED)
# 🌟 COMBINED DATASET + EDA SECTION (BEAUTIFIED ONLY — NO WORDS CHANGED)
def section_dataset_eda():

    st.title("📊 Dataset Summary + Exploratory Data Analysis (EDA)")
    st.markdown(
        """
        <div style="background-color:#fff3cd; padding:12px; border-radius:8px; color:#000000;">
            This section combines the <b>Dataset Overview</b>, <b>Missingness Analysis</b>,
            <b>Abnormalities Detected</b>, <b>Target Column Analysis</b>,
            and <b>Statistical Test Summary</b>, so all EDA insights appear in one place.
        </div>
        """,
        unsafe_allow_html=True
    )
    # ======================================================
    # 📂 DATASET OVERVIEW
    # ======================================================

    # =======================================================
    # SAMPLE RAW DATA PROVIDED
    # =======================================================
    data = {
        "ID": [27639, 24120, 106030, 55430, 185100, 35990, 343950, 188050, 367220, 132430, 170120, 364460, 287940],
        "Delivery_person_ID": ["0x6889","0xd684","0xc0f2","0xbf0f","0x8616","0xabc8","0xaa2d","0x84b6","0x8f95","0xe137","0x897","0x6abb","0x4880"],
        "Delivery_person_Age": ["HYDRES15","LUDHRES18","ALHRES02","DEHRES19","INDORES010","BANGRES03","INDORES18","JAPRES11","PUNERES15","KOLRES16","MUMRES19","CHENRES010","PUNERES08"],
        "Delivery_person_Ratings": ["DEL03","DEL02","DEL03","DEL01","DEL02","DEL03","DEL01","DEL01","DEL02","DEL02","DEL02","DEL023","DEL033"],
        "Restaurant_latitude": [84.61,24.83,14.6,None,83.72,24.11,64.32,54.62,4.61,94.82,None,54.71,24.81],
        "Restaurant_longitude": [7.45,0.89,-25.45,None,22.75,12.97,22.75,26.9,18.6,22.53,None,13.06,18.53],
        "Delivery_location_latitude": [971078.36,18475.82,43681.83,-30.37,4004075.90,16677.64,3975.89,94075.79,621573.75,12988.36,131141.72,676280.25,408073.89],
        "Delivery_location_longitude": [885517.51,961530.97,16725.51,-78.07,284722.88,70913.03,42922.86,300726.91,108118.76,550722.66,307419.21,186513.08,520180.56],
        "Order_Date": ["05-03-2022","14-02-2022","15-02-2022","13-02-2022","08-03-2022","01-03-2022","29-03-2022","24-03-2022","18-03-2022","14-02-2022","25-03-2022","01-03-2022","07-03-2022"],
        "Time_Orderd": ["23:25","22:30","19:20",None,"20:20","20:00","19:55","10:15","17:25","20:45",None,"09:30","23:25"],
        "Time_Order_picked": ["23:40","22:40","19:35","09:35","20:25","20:15","20:05","10:20","17:35","21:00","22:00","09:45","23:30"],
        "Weatherconditions": ["Sandstorms","Cloudy","Sunny","Sandstorms","Windy","Fog","Cloudy","Windy","Stormy","Sunny",None,"Fog","Windy"],
        "Road_traffic_density": ["Low","Low","Jam","Low","Jam","Jam","Jam","Low","Medium","Jam",None,"Low","Low"],
        "Vehicle_condition": [1,2,0,1,2,2,2,2,1,1,3,0,2],
        "Type_of_order": ["Buffet","Drinks","Meal","Buffet","Drinks","Drinks","Meal","Meal","Drinks","Meal","Meal","Meal","Meal"],
        "Type_of_vehicle": ["motorcycle","scooter","motorcycle","scooter","scooter","scooter","motorcycle","motorcycle","motorcycle","motorcycle","motorcycle","motorcycle","motorcycle"],
        "multiple_deliveries": [1,1,1,0,0,1,0,0,0,1,1,1,0],
        "Festival": ["No","No","No","No","No","No","Yes","No","No","No","No","No","No"],
        "City": ["Metropolitian","Metropolitian","Metropolitian","Metropolitian","Urban","Metropolitian","Metropolitian","Metropolitian","Metropolitian","Metropolitian","Metropolitian","Metropolitian","Metropolitian"],
        "Time_taken(min)": [112,26,29,11,37,32,45,27,16,24,38,26,27]
    }

    df_sample = pd.DataFrame(data)

    st.subheader("📄 Raw Data Preview (Sample)")
    st.dataframe(df_sample, use_container_width=True, height=400)

    # ======================================================
    # 🧩 MISSING VALUE & CORRELATION HEATMAPS
    # ======================================================
    st.markdown("---")
    st.subheader("🖼 Missing Values & Correlation Heatmap (Side by Side)")

    from PIL import Image

    img1 = Image.open("missingo.png").resize((600, 450))
    img2 = Image.open("corr.png").resize((600, 450))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img1, caption="Missing Data Visualization", use_container_width=False)
    with col2:
        st.image(img2, caption="Correlation Heatmap", use_container_width=False)

    # ======================================================
    # 🗺 DELIVERY MAP
    # ======================================================
    st.markdown("---")
    st.header("🗺 Delivery Points Map")

    st.write("""
    This map shows the **delivery locations across India**, plotted after cleaning  
    invalid latitude/longitude values.

    It helps us understand:
    - Delivery spread across cities  
    - Whether coordinates fall inside India  
    - Distribution density  
    - Whether invalid points were filtered effectively  
    """)

    map_img = Image.open("map.png").resize((1100, 550))
    st.image(map_img, caption="Delivery Locations (map.png)", use_container_width=False)

    # ======================================================
    # 🚨 ABNORMALITIES FOUND
    # ======================================================
    st.markdown("---")
    st.header("🚨 Abnormalities I Found in the Dataset")

    st.markdown("""
    ## 🔍 Observations (Missingness Patterns)

    1. **Delivery person–related columns are correlated** in missingness, suggesting that if one rider detail is missing, others are likely missing too → indicating **lack of rider data**.

    2. **Time Ordered column is also related to rider info**, which shows missingness might be due to **network/system failure** where the system could not log rider details + time of order.

    3. There is a **very high correlation between weather patterns and road traffic** in missingness.  
    This does **NOT** mean weather and traffic are correlated.  
    It means if weather is missing, traffic is also often missing → same data source failure.

    4. **Road traffic density correlates with rider**, possibly because traffic info was provided through rider's phone.

    ---

    ## 🔍 Observations (Data Quality Problems)

    1. The **star ratings of all minors (age 15) is 1**.  
    2. The **vehicle condition** of these minors is very bad.  
    3. No **weather and traffic conditions** are available for these riders.  
    4. **Age = 15** is below legal driving age → invalid data.  
    5. **Negative lat/long values are impossible** for India.

    ---

    ## 🔍 Observations (Ratings Column)

    1. Riders with **ratings = 1** seem anomalous compared to overall distribution.  
    2. A rating of **6** appears → impossible because max rating is 5.  
       → Must be fixed or removed.

    ---

    ## 🗺 Valid Values for Latitude & Longitude

    India lies:

    - **Latitude:** 6°44′ N to 35°30′ N  
    - **Longitude:** 68°07′ E to 97°25′ E  

    Therefore:
    - Negative values are invalid  
    - Values < 1 are corrupt  
    - 3640 rows contain invalid/messy coordinates  

    ---

    ## 🛠 Corrections

    1. Age column → numeric  
    2. Ratings → float  
    3. Dates/times → proper datetime  
    4. Vehicle condition → categorical  
    5. Multiple deliveries → integer  
    6. Target column → numeric  
    7. Distance → NaN for invalid lat/long  
    8. Negative coords → convert to absolute  
    9. Use advanced imputation for distance-related NaNs  

    ---

    ## 🧨 Final Notes

    - Missingness shows **systematic logging failures**  
    - Location data needs major cleanup  
    - Rider profile issues indicate **dirty/synthetic entries**
    """)

    # ======================================================
    # 🎯 TARGET COLUMN ANALYSIS
    # ======================================================
    st.markdown("---")
    st.header("🎯 Target Column Analysis — Time Taken")

    st.markdown("""
    **Observations:**

    - The target column is not fully continuous in nature.  
    - The target column shows **dual modality** with two peaks:  
      - around **17–18 minutes**  
      - around **26–27 minutes**  

    - Some extreme values (40–50 min) are **rare but valid**, not outliers.  

    - **Jarque-Bera Test = 0.0 → Reject normality**  
      → Data is **not normally distributed**.
    """)

    # ======================================================
    # 📊 SUMMARY OF ALL STATISTICAL TESTS
    # ======================================================
    st.markdown("---")
    st.header("📊 Summary of Statistical Tests Performed")

    st.markdown("""
    ### 🎯 Purpose  
    To identify which features **significantly affect delivery time (time_taken)**.

    ## 1️⃣ Outlier Analysis
    - Extreme time_taken linked with **jam/high traffic**  
    - Weather: fog, sunny, cloudy, windy, stormy, sandstorms  
    - Extreme distances higher than average  
    - Outliers are **valid**

    ## 2️⃣ Power Transformation
    - Applied **Yeo-Johnson**  
    - QQ plot shows improved distribution

    ## 3️⃣ ANOVA Tests
    - `traffic` → ✔ significant  
    - `weather` → ✔ significant  
    - `multiple_deliveries` → ✔ significant  
    - `vehicle_condition` → ✔ significant  
    - `city_type` → ✔ significant  
    - `pickup_time_minutes` → ✘ not significant  
    - `type_of_order` → ✘ not significant  

    ## 4️⃣ Chi-Square Tests
    - `is_weekend` → traffic → ✘ no association  
    - `festival` → traffic → ✔ significant  
    - `order_time_of_day` → traffic → ✔ significant  
    - `type_of_vehicle` → vehicle_condition → ✔ significant  
    - `weather` → traffic → ✘ no association  
    - `city_type` → traffic → ✔ significant  
    - `city_name` → traffic → ✘ no association  
    - `type_of_order` → pickup_time_minutes → ✘ no association  
    - `festival` → type_of_order → ✘ no association  

    ## ⭐ Final Insights
    **Most important predictors of delivery time:**  
    `traffic`, `weather`, `distance`, `multiple_deliveries`, `vehicle_condition`, `city_type`  

    **Low-impact features:**  
    `type_of_order`, `weekend`, `pickup_time`, `city_name`  

    **Time_taken is not normal**, but transformation fixes it.
    """)
    st.markdown("""
    <div style="
        background-color:#ffe9e8;
        padding:15px;
        border-left: 6px solid #d9534f;
        border-radius:6px;
        color:#000000;
    ">
    <b>🗑️ Columns Dropped During Preprocessing</b><br><br>

    The following columns were removed because they were either irrelevant,
    highly correlated, duplicate information, or not useful for modeling:

    <ul>
        <li>rider_id</li>
        <li>restaurant_latitude</li>
        <li>restaurant_longitude</li>
        <li>delivery_latitude</li>
        <li>delivery_longitude</li>
        <li>order_date</li>
        <li>order_time_hour</li>
        <li>order_day</li>
        <li>city_name</li>
        <li>order_day_of_week</li>
        <li>order_month</li>
    </ul>

    </div>
    """, unsafe_allow_html=True)

   

  







def section_baseline():

    st.title("🧪 Baseline Model Experimentation")

    st.write("""
    We conducted two major baseline experiments: 
    **(1) Imputing Missing Values** and **(2) Dropping Missing Values**, 
    to understand which data strategy gives better delivery time predictions.
    """)

    # =====================================================
    # SECTION 1 — IMPUTATION EXPERIMENTS
    # =====================================================
    st.markdown("---")
    st.header("🧹 1. Experiment: Imputing Missing Values")

    st.markdown("""
    In this experiment, missing values were handled using:
    - SimpleImputer (mode + 'missing' fill)
    - Missing Value Indicators
    - KNNImputer for numeric features
    - Full preprocessing pipeline (scaling + encoding)

    This approach keeps **all data** and captures patterns hidden inside missingness.
    """)


    # ---------------------------
    # IMPUTATION RESULTS TABLE
    # ---------------------------
    st.subheader("📊 Model Performance (After Imputation)")

    df_imp = pd.DataFrame([
        ["Random Forest (Imputed)", 3.2821, 0.8024, 4.1691],
        ["XGBoost (Imputed)", 3.2666, 0.8077, 4.1130]
    ], columns=["Model", "MAE", "R²", "RMSE"])

    st.dataframe(df_imp, use_container_width=True)

    st.info("Tree-based models performed strongest under imputation. XGBoost (Imputed) gave the best imputed performance.")

    # =====================================================
    # SECTION 2 — DROPPING MISSING VALUES
    # =====================================================
    st.markdown("---")
    st.header("🧽 2. Experiment: Dropping Missing Value Rows")

    st.markdown("""
    In this experiment, we removed all rows containing missing values.
    
    This creates a **clean dataset** without synthetic values:
    - No imputation noise  
    - Easier for tree models to fit 
    """)

    st.subheader("📊 Model Performance (Dropped Missing Values)")

    df_drop = pd.DataFrame([
        ["Linear Regression (Dropped)", 4.7280, 0.6008, 5.9082],
        ["Random Forest (Dropped)", 3.1287, 0.8252, 3.9095],
        ["XGBoost (Dropped)", 3.1084, 0.8298, 3.8581]
    ], columns=["Model", "MAE", "R²", "RMSE"])

    st.dataframe(df_drop, use_container_width=True)

    st.success("Dropping missing values produced the best overall performance — especially for XGBoost and Random Forest.")

    # =====================================================
    # MLflow Parallel Coordinates Plot
    # =====================================================
    st.markdown("---")
    st.header("📈 MLflow Experiment Comparison Plot")

    st.write("Below is the **MLflow parallel coordinates plot** comparing all experiment runs:")

    from PIL import Image

    mlflow_plot = Image.open("keep_drop.png")

    # Resize (reduce width & height)
    mlflow_plot = mlflow_plot.resize((1000, 500))   # adjust size as needed

    st.image(mlflow_plot, caption="MLflow Parallel Coordinates Plot (keep_drop.png)", use_container_width=False)


    st.info("This MLflow plot visually compares model runs logged during 'Keep vs Drop' experiments.")
    
    st.title("🧪 Final Model Experimentation")

    st.write("""
    In the final stage of experimentation, we compared several advanced machine learning 
    algorithms to identify the best-performing model for predicting delivery time.

    The models evaluated were:

    - ⭐ **LightGBM**
    - ⭐ **CatBoost**
    - ⭐ **XGBoost**
    - ⭐ **Random Forest**
    - ⭐ **SVR (Support Vector Regressor)**
    """)

    st.markdown("---")
    st.header("📊 Final Model Performance Results")

    # -------------------------
    # MODEL PERFORMANCE TABLE
    # -------------------------
    df_results = pd.DataFrame([
        ["LightGBM",       3.0649, 3.7834, 0.8363],
        ["CatBoost",       3.0835, 3.8247, 0.8327],
        ["XGBoost",        3.1525, 3.9304, 0.8234],
        ["Random Forest",  3.1313, 3.9140, 0.8248],
        ["SVR",            3.7408, 4.7349, 0.7436]
    ], columns=["Model", "MAE", "RMSE", "R² Score"])

    st.dataframe(df_results, use_container_width=True)

    st.info("""
    ✅ **LightGBM achieved the best performance overall**,  
       with the lowest MAE, lowest RMSE, and highest R² Score.

    🔹 CatBoost was a close second.  
    🔹 Random Forest and XGBoost performed competitively but slightly behind.  
    🔹 **SVR** clearly underperformed compared to tree-based models on this dataset.
    """)

    # ----------------------------------------
    # SHOW FINAL EXPERIMENT IMAGE
    # ----------------------------------------
    st.markdown("---")
    st.header("📟 Visual Overview of Final Experiments")

    from PIL import Image
    final_plot = Image.open("final_exp.png").resize((1100, 500))

    st.image(final_plot, caption="Final Model Experiment Comparison (final_exp.png)", use_container_width=False)

    



def section_hpt():

    st.title("🎯 Hyperparameter Tuning (Optuna)")

    st.write("""
    After evaluating multiple models, **LightGBM** and **CatBoost** showed the strongest 
    baseline performance.  
    Therefore, these two models were selected for **advanced hyperparameter tuning using Optuna**.
    """)

    st.markdown("---")
    st.header("🔧 Models Selected for Tuning")
    st.write("""
    - ⭐ **LightGBM** — best overall baseline performer  
    - ⭐ **CatBoost** — close second with strong stability  
    """)

    # ============================================================
    # LIGHTGBM BEST PARAMETERS
    # ============================================================
    st.markdown("---")
    st.subheader("🌟 Best Hyperparameters — LightGBM (Optuna)")

    df_lgb = pd.DataFrame({
        "Parameter": [
            "n_estimators", "reg_alpha", "reg_lambda", "num_leaves",
            "min_child_samples", "subsample", "max_depth",
            "colsample_bytree", "learning_rate"
        ],
        "Value": [
            731, 0.5201348791725293, 0.609484465536558, 100,
            60, 0.575426956719928, 15,
            0.8372839934225397, 0.014850959317240326
        ]
    })

    st.table(df_lgb)
    st.subheader("📊 Best LightGBM Optuna Metrics")

    df_lgb_metrics = pd.DataFrame([
        ["RMSE", 3.7549147645141883],
        ["MAE", 3.0449763471141154],
        ["R²", 0.8387827163080033]
    ], columns=["Metric", "Value"])

    st.dataframe(df_lgb_metrics, use_container_width=True)

    # ============================================================
    # CATBOOST BEST PARAMETERS
    # ============================================================
    st.markdown("---")
    st.subheader("🌟 Best Hyperparameters — CatBoost (Optuna)")

    df_cat = pd.DataFrame({
        "Parameter": [
            "depth", "iterations", "l2_leaf_reg", "learning_rate"
        ],
        "Value": [
            10, 1062, 0.10715589227169486, 0.012690362728359193
        ]
    })
    
    st.table(df_cat)
    st.subheader("📊 Best CatBoost Optuna Metrics")
    df_cat_metrics = pd.DataFrame([
        ["RMSE", 3.7527017901064372],
        ["R²", 0.8389726884450109],
        ["MAE", 3.032892922101842]
    ], columns=["Metric", "Value"])
    st.dataframe(df_cat_metrics, use_container_width=True)

    # ============================================================
    # SIDE-BY-SIDE OPTUNA PLOTS
    # ============================================================

    st.markdown("---")
    st.header("📊 Optuna Optimization Results")

    col1, col2 = st.columns(2)

    with col1:
            st.image("catboost_optuna.png", caption="CatBoost Optuna Plot")

    with col2:
            st.image("lightbgm_optuna.png", caption="LightGBM Optuna Plot")

    



def section_final_model():
    st.title("🤖 Final Model & Metrics")

    st.write("""
    After tuning LightGBM and CatBoost, I experimented with two ensemble strategies:
    **Stacking** and **Weighted Modeling**.

    Both performed well with similar performance, but Weighted Modeling was finally selected.
    """)

    # -------------------------
    # 📊 STACKING RESULTS
    # -------------------------
    st.subheader("🔷 Stacking Model Results")

    st.markdown("""
    **Models used in the Stacking Ensemble:**
    - **Base Models:**  
        • LightGBM (Optuna Tuned)  
        • CatBoost (Optuna Tuned)  
    - **Meta-Learner:**  
        • **Ridge Regression**  
    """)
    st.subheader("🔷 Metrics")

    df_stack = pd.DataFrame([
        ["MAE", 3.0209],
        ["RMSE", 3.7246],
        ["R²", 0.8414]
    ], columns=["Metric", "Value"])

    st.dataframe(df_stack, use_container_width=True)


    # -------------------------
    # 📊 WEIGHTED MODEL RESULTS
    # -------------------------
    st.subheader("🟩 Weighted Modeling Results")

    st.markdown("""
    **Weighted Ensemble Models:**
    - LightGBM → **60%**
    - CatBoost → **40%**
    """)
    st.subheader("🔷 Metrics")

    df_weight = pd.DataFrame([
        ["LightGBM Weight", "0.60"],
        ["CatBoost Weight", "0.40"],
        ["MAE", 3.0199],
        ["RMSE", 3.7240],
        ["R²", 0.8414]
    ], columns=["Metric", "Value"])

    st.dataframe(df_weight, use_container_width=True)


    # -------------------------
    # 📝 WHY WEIGHTED MODELING WAS CHOSEN
    # -------------------------
    st.markdown("""
    ### ✅ Why I Chose Weighted Modeling Instead of Stacking

    Even though both stacking and weighted ensembles performed similarly,  
    **Weighted Modeling was chosen because:**

    - It is **simpler** and avoids the stacking architecture overhead  
    - **Ridge meta-learner added complexity without improving performance**  
    - Weighted averaging is **more stable** across runs  
    - Easier to **deploy in production** (no meta-model layer)  
    - Offers **direct interpretability** of how much each model contributes  
    - Final metrics were **slightly better or equal** to stacking  

    **Final Selected Model → Weighted Ensemble (0.60 LightGBM + 0.40 CatBoost)**  
    """)
     # --------------------------------------------------------
    # 📷 SHOW WEIGHT EXPERIMENT PLOT
    # --------------------------------------------------------
    st.markdown("---")
    st.header("📉 Weight Combination Experiment Visualization")

    st.write("This plot shows how performance changed as I tested different weight combinations between LightGBM & CatBoost.")

    from PIL import Image
    weighted_img = Image.open("weighted_img.png")
    st.image(weighted_img, caption="Weighted Model Experiment Plot (LightGBM vs CatBoost)", use_container_width=True)







def section_predict():
    st.title("⚙️ DVC Pipeline, CI/CD Automation & Model Workflow")
    # =====================================================
    # 📦 DVC PIPELINE
    # =====================================================
    st.markdown("---")
    st.header("📦 DVC Pipeline: End-to-End ML Workflow")

    st.markdown(
        """
        <div style="padding:15px; background-color:white; border:1px solid #ccc; 
        border-radius:8px; color:black;">
            The machine learning pipeline—from raw Swiggy data to final model 
            registration—is automated using <b>DVC</b>. 
            Each stage ensures reproducibility, versioning, and traceability.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.code(
        """stages:
  clean_data:
    cmd: python src/data/data_cleaning.py
    deps:
      - src/data/data_cleaning.py
      - data/raw/swiggy.csv
    outs:
      - data/cleaned/swiggy_cleaned.csv

  split_data:
    cmd: python src/data/data_processing.py
    deps:
      - src/data/data_processing.py
      - data/cleaned/swiggy_cleaned.csv
      - params.yaml
    outs:
      - data/interim/train.csv
      - data/interim/test.csv
    params:
      - Data_Preparation.test_size
      - Data_Preparation.random_state

  preprocess:
    cmd: python src/features/data_preprocessing.py
    deps:
      - src/features/data_preprocessing.py
      - data/interim/train.csv
      - data/interim/test.csv
    outs:
      - data/processed/train_trans.csv
      - data/processed/test_trans.csv
      - models/preprocessor.joblib

  train:
    cmd: python src/models/train_model.py
    deps:
      - src/models/train_model.py
      - data/processed/train_trans.csv
      - params.yaml
    outs:
      - models/catboost_model.joblib
      - models/lgbm_model.joblib

  evaluate:
    cmd: python src/models/evaluation.py
    deps:
      - src/models/evaluation.py
      - data/processed/test_trans.csv
      - models/catboost_model.joblib
      - models/lgbm_model.joblib
      - params.yaml

  register_model:
    cmd: python src/models/register.py
    deps:
      - src/models/register.py
      - models/catboost_model.joblib
      - models/lgbm_model.joblib
      - data/processed/test_trans.csv
      - params.yaml
    outs: []
        """,
        language="yaml"
    )

    st.markdown(
        """
        <div style="padding:15px; background-color:white; border:1px solid #ccc; 
        border-radius:8px; color:black;">
            <b>Key DVC Behaviors:</b><br>
            • Data and models are version-controlled<br>
            • Each stage runs only when dependencies change<br>
            • Enables fully reproducible experimentation<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # 🚀 CI/CD PIPELINE
    # =====================================================
    st.markdown("---")
    st.header("🚀 CI/CD Automation: GitHub Actions + AWS")

    st.markdown(
        """
        <div style="padding:15px; background-color:white; border:1px solid #ccc; 
        border-radius:8px; color:black;">
            The CI/CD pipeline automates testing, model validation, containerization, 
            and prepares the application for deployment.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.code(
        """name: CI-CD
on: push

jobs:
  CI-CD:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirement_action.txt

      - name: Configure AWS Credentials (for DVC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-north-1

      - name: DVC Pull from S3 Bucket
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          dvc pull

      - name: Test Model Loading
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          pytest tests/test_model_loading.py -q

      - name: Test Model Performance
        env:
          MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        run: |
          pytest tests/test_performance.py -q

      - name: Configure AWS Credentials for ECR
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID_B }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY_B }}
          aws-region: eu-north-1

      - name: Login to Amazon ECR
        id: login-ecr
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build, tag, and push docker image to Amazon ECR
        env:
          REGISTRY: ${{ steps.login-ecr.outputs.registry }}
          REPOSITORY: delivery-time-estimator
          IMAGE_TAG: latest
        run: |
          docker build -t $REPOSITORY .
          docker tag $REPOSITORY:latest $REGISTRY/$REPOSITORY:$IMAGE_TAG
          docker push $REGISTRY/$REPOSITORY:$IMAGE_TAG

      - name: Refresh EC2 Auto Scaling Group
        run: |
          aws autoscaling start-instance-refresh \
            --auto-scaling-group-name delivery-asg \
            --region eu-north-1 \
            --preferences MinHealthyPercentage=100
        """,
        language="yaml"
    )

    st.markdown(
        """
        <div style="padding:15px; background-color:white; border:1px solid #ccc; 
        border-radius:8px; color:black;">
            <b>What CI/CD Ensures:</b><br>
            • Ensures latest data & MLflow model artifacts are synced<br>
            • Validates model loading + performance automatically<br>
            • Builds and pushes Docker image for serving<br>
        </div>
        """,
        unsafe_allow_html=True
    )

    # =====================================================
    # 🌐 FINAL SECTION
    # =====================================================
    st.markdown("---")
    st.header("🌐 Model Serving Integration")

    st.markdown(
        """
        <div style="padding:15px; background-color:white; border:1px solid #ccc; 
        border-radius:8px; color:black;">
            The final <b>Weighted Ensemble Model (LightGBM + CatBoost)</b> is packaged inside a Docker
            container and made available through an API endpoint.  
            The Streamlit UI (this app) connects to that API to return the predicted delivery time.
        </div>
        """,
        unsafe_allow_html=True
    )





def section_tech():
    st.title("🧱 Tech Stack & Pipeline")


# ============================================================
# SECTION DISPATCHER
# ============================================================

if section == "🏠 Project Overview":
    section_overview()

elif section == "📊 Dataset + EDA":
    section_dataset_eda()

elif section == "🧪 Models Experimentation":
    section_baseline()

elif section == "🎯 Hyperparameter Tuning":
    section_hpt()

elif section == "🤖 Final Model & Metrics":
    section_final_model()

elif section == "🔮 DVC Pipeline, CI/CD Automation":
    section_predict()

elif section == "🧱 Tech Stack & Pipeline":
    section_tech()
