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
        "🛰️ API Service (FastAPI)",
        "🌥️ AWS Deployment (Auto Scaling)",
        "🔥 Stress Testing (Postman)",
        "🔮 DVC Pipeline, CI/CD Automation",
        "🧱 Tech Stack & Pipeline",
        "🔗 Project Repositories & Author"
    ]
)


# ============================================================
# SECTION FUNCTIONS
# ============================================================

def section_overview():

    st.title("🏠 Project Overview")

    # ============================================================
    # ⭐ WHAT THIS PROJECT IS (HIGHLIGHT CARD)
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#eaf3ff;
            padding:25px;
            border-radius:16px;
            border:1px solid #c7dcff;
            box-shadow:0 3px 10px rgba(0,0,0,0.08);
            margin-top:15px;
        ">
        <h2 style="color:#083b6e; margin-bottom:12px;">🚀 ETAFlow — Scalable Delivery Time Prediction System</h2>

        <p style="color:#222; font-size:17px; line-height:1.7;">
        This project delivers a <b>full production-grade Delivery Time Estimation system</b> 
        similar to Swiggy, Zomato, and Uber Eats.  
        It combines <b>Machine Learning, MLOps, Cloud Engineering, CI/CD, Docker, FastAPI,
        AWS Auto Scaling, and real-time monitoring</b> into a single unified pipeline.
        </p>

        <p style="color:#222; font-size:17px; line-height:1.7;">
        The entire workflow — from data cleaning ➝ model training ➝ MLflow registry ➝ 
        FastAPI service ➝ Docker deployment ➝ automatic scaling — is fully automated 
        and designed for <b>real-world production environments</b>.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # 🌟 WHAT THE PROJECT DOES
    # ============================================================
    st.markdown("## 🌟 What This Project Does")

    st.write(
        """
        This system predicts **expected delivery time (ETA)** using a powerful weighted 
        ensemble model (LightGBM + CatBoost).  
        It ensures:
        
        - High accuracy  
        - Real-time response  
        - Auto-scaling under heavy traffic  
        - Fully automated CI/CD deployment  
        - Zero downtime updates  
        - Cloud monitoring and observability  
        """
    )

    # ============================================================
    # 🎯 PROJECT GOALS (CARD)
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#fff7e6;
            padding:25px;
            border-radius:16px;
            border:1px solid #f5d5a4;
            box-shadow:0 3px 10px rgba(0,0,0,0.08);
            margin-top:25px;
        ">
        <h2 style="color:#8a4b00; margin-bottom:15px;">🎯 Project Goals</h2>

        <p style="color:#222; font-size:17px; line-height:1.7;">
        The goal was to build a <b>fast, scalable, and cloud-ready ETA prediction pipeline</b> 
        that supports real production constraints:
        </p>

        <ul style="color:#222; font-size:16px; line-height:1.65;">
            <li>Deliver ETA predictions in <b>milliseconds</b></li>
            <li>Automatically scale during peak loads (validated with a 100k-request stress test)</li>
            <li>Deploy seamlessly using <b>Auto Scaling Groups + Launch Templates</b></li>
            <li>Continuously integrate and deploy using <b>GitHub Actions CI/CD</b></li>
            <li>Track every experiment and model using <b>MLflow</b></li>
            <li>Make the entire ML pipeline reproducible using <b>DVC</b></li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # 🔧 KEY COMPONENTS OF THE SYSTEM
    # ============================================================
    st.markdown("## 🔧 Key Components of the System")

    st.write(
        """
        This project is not just a machine learning model —  
        it is a **complete engineering ecosystem**, including:

        ### 🔹 Data Engineering  
        - Heavy EDA  
        - Fixing corrupted coordinates  
        - Feature engineering  
        - Outlier handling  
        - Missing value strategy (drop vs impute)

        ### 🔹 Machine Learning  
        - Advanced experimentation  
        - Hyperparameter tuning (Optuna)  
        - Weighted ensemble model  
        - MLflow model registry  

        ### 🔹 Deployment  
        - FastAPI as the inference server  
        - Docker containerization  
        - Amazon ECR image storage  
        - EC2 Auto Scaling deployment  

        ### 🔹 MLOps & Cloud  
        - GitHub Actions CI/CD  
        - Instance Refresh (zero-downtime updates)  
        - CloudWatch real-time metrics  
        - Auto Scaling based on CPU  
        """
    )

    # ============================================================
    # 🏆 KEY OUTCOMES (HIGHLIGHT CARD)
    # ============================================================
    st.markdown(
        """
        <div style="
            background-color:#e8fff3;
            padding:25px;
            border-radius:16px;
            border:1px solid #b9ecd5;
            box-shadow:0 3px 10px rgba(0,0,0,0.08);
            margin-top:25px;
        ">
        <h2 style="color:#106644; margin-bottom:10px;">🏆 Key Outcomes</h2>

        <ul style="color:#0d422f; font-size:17px; line-height:1.7;">
            <li>Achieved a strong performance: <b>MAE ≈ 3.01</b> and <b>R² ≈ 0.84</b></li>
            <li>Total traffic generated: 100,000 API requests</li>
            <li>Auto Scaling increased EC2s from <b>1 → 2 → 3</b> automatically</li>
            <li>Zero downtime during new deployment using Instance Refresh</li>
            <li>Fully automated ML + Deployment pipeline with CI/CD + DVC</li>
            <li>Production-grade FastAPI service running via Docker</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ============================================================
    # 📝 SUMMARY TEXT
    # ============================================================
    st.markdown("## 📝 Final Summary")

    st.write(
        """
        This project demonstrates how a modern tech company would build a 
        **scalable, reliable, fully-automated ML system** for real-time delivery predictions.

        It combines:
        - ML engineering  
        - Cloud deployment  
        - Infrastructure automation  
        - Reproducible pipelines  
        - Monitoring and stress testing  

        making it a complete **MLOps portfolio-grade project**.
        """
    )


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
    st.subheader("🖼 Missing Values & Correlation Heatmap")

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

    st.write("""
    This section covers how the project uses **DVC** for pipeline automation  
    and **GitHub Actions + AWS** for CI/CD and model verification.
    """)

    # =====================================================
    # 📦 DVC PIPELINE
    # =====================================================
    st.markdown("---")
    st.header("📦 DVC Pipeline (End-to-End Workflow)")

    st.write("""
    The full ML workflow—from raw data to final model—is tracked through DVC.
    Below is the pipeline structure:
    """)

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
        language="yaml",
    )
    # ------------------------------
    # 📌 WHAT EACH STAGE DOES (CARD)
    # ------------------------------

    st.markdown("""
    <div style="
        background-color:#e8fff3; 
        padding:14px 22px;
        display:inline-block;
        border-radius:10px;
        border:1px solid #b8e6cf;
        box-shadow:0 2px 4px rgba(0,0,0,0.06);
        margin-top:20px;
        margin-bottom:10px;
    ">
        <h3 style="color:#124d35; margin:0; font-size:20px;">
            📌 What Each Stage Does
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    - **clean_data** → Cleans raw Swiggy data and removes errors.  
    - **split_data** → Creates reliable train/test datasets for experimentation.  
    - **preprocess** → Builds full transformation pipeline (encoding + scaling) and stores it.  
    - **train** → Trains both LightGBM and CatBoost models.  
    - **evaluate** → Validates the models using unseen test data.  
    - **register_model** → Registers the best-performing model for deployment.  
    """)




    # =====================================================
    # 🚀 CI/CD PIPELINE
    # =====================================================
    st.markdown("---")
    st.header("🚀 CI/CD Pipeline (GitHub Actions + AWS)")

    st.write("""
    The CI/CD pipeline automatically tests models, validates performance,  
    and packages the final model into a deployable Docker image.
    """)

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

      - name: Install Dependencies
        run: |
          pip install -r requirement_action.txt

      - name: Configure AWS for DVC
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: eu-north-1

      - name: DVC Pull
        run: dvc pull

      - name: Test Model Loading
        run: pytest tests/test_model_loading.py -q

      - name: Test Model Performance
        run: pytest tests/test_performance.py -q

      - name: Login to ECR
        uses: aws-actions/amazon-ecr-login@v2

      - name: Build & Push Docker Image
        run: |
          docker build -t delivery-time-estimator .
          docker tag delivery-time-estimator:latest ${{ steps.login-ecr.outputs.registry }}/delivery-time-estimator:latest
          docker push ${{ steps.login-ecr.outputs.registry }}/delivery-time-estimator:latest

      - name: Refresh Auto Scaling Group
        run: |
          aws autoscaling start-instance-refresh \
            --auto-scaling-group-name delivery-asg \
            --region eu-north-1 \
            --preferences MinHealthyPercentage=100
""",
        language="yaml",
    )

    # ------------------------------------------
    # 🧪 CI/CD TESTS PERFORMED (BEAUTIFUL CARD)
    # ------------------------------------------
    st.markdown("""
    <div style="
        background-color:#e8fff3; 
        padding:14px 22px;
        display:inline-block;
        border-radius:10px;
        border:1px solid #b8e6cf;
        box-shadow:0 2px 4px rgba(0,0,0,0.06);
        margin-top:20px;
        margin-bottom:10px;
    ">
        <h3 style="color:#124d35; margin:0; font-size:20px;">
            🧪 CI/CD — Tests Performed Automatically
        </h3>
    </div>
    """, unsafe_allow_html=True)




    # ------------------  CI/CD Steps  ------------------
    st.markdown("""
    ### 🔄 Automated CI/CD Steps  
    - **Checkout Code** → Pulls your repository to the CI runner.  
    - **Install Dependencies** → Sets up Python + required packages.  
    - **DVC Pull** → Downloads latest data & artifacts from the S3 remote.  
        ### ✅ Automated Tests
        - **Model Loading Test** → Ensures the latest MLflow model loads without any errors.  
        - **Model Performance Test** → Validates that MAE/RMSE stay above the defined threshold before deployment.    
    - **Build Docker Image** → Packages the updated API + model.  
    - **Push Image to ECR** → Uploads the new version to AWS Elastic Container Registry.  
    - **Refresh Auto Scaling Group** → Deploys updated image to all EC2 instances automatically.  
    """)

def section_fastapi():

    st.title("🛰️ FastAPI Service — Real-Time ETA Prediction")

    st.write(
        """
        This section describes how the production-grade **FastAPI** service is built to deliver
        real-time Swiggy delivery time predictions using:

        - 🚀 MLflow Model Registry  
        - 🧠 Weighted Ensemble (LightGBM + CatBoost)  
        - 🧹 Automated Input Cleaning  
        - 🔧 Saved Preprocessing Pipeline  
        """
    )

    # =============================
    # API ARCHITECTURE CARD
    # =============================
    st.markdown(
        """
    <div style="
    background-color:#eaf3ff;
    padding:24px;
    border-radius:16px;
    border:1px solid #c6dcff;
    box-shadow:0 3px 10px rgba(0,0,0,0.1);
    margin-top:16px;
    ">

    <h3 style="color:#0a2c4d; margin-bottom:12px;">📦 How the API Works</h3>

    <p style="color:#000; line-height:1.55;">
        The FastAPI backend loads the <b>latest registered model</b> from MLflow, including:
    </p>

    <ul style="color:#000; line-height:1.6; padding-left:20px;">
        <li>Saved <b>preprocessor</b></li>
        <li><b>CatBoost</b> model</li>
        <li><b>LightGBM</b> model</li>
        <li>Blending weights (LightGBM = 0.60, CatBoost = 0.40)</li>
    </ul>

    <p style="color:#000; line-height:1.55; margin-top:12px;">
        For every request, the backend performs:
    </p>

    <ul style="color:#000; line-height:1.6; padding-left:20px;">
        <li>Raw JSON is cleaned via <b>perform_data_cleaning()</b></li>
        <li>Features transformed using the stored <b>preprocessor</b></li>
        <li>Both models generate predictions</li>
        <li>Final ETA = <b>weighted average</b> of the two outputs</li>
    </ul>

    </div>
    """,
        unsafe_allow_html=True,
    )


    st.markdown("### 🧠 API Startup — Load Latest MLflow Model")

    st.code(
        """
mlflow.set_tracking_uri(TRACKING_URI)
client = MlflowClient()

MODEL_NAME = "Swiggy-Ensemble-Model"

latest_ver = client.get_latest_versions(MODEL_NAME, stages=None)[0].version
model_uri = f"models:/{MODEL_NAME}/{latest_ver}"

model_bundle = mlflow.sklearn.load_model(model_uri)

preprocessor = model_bundle["preprocessor"]
cat_model = model_bundle["catboost"]
lgb_model = model_bundle["lightgbm"]
w_cat = model_bundle["weights"]["cat"]
w_lgb = model_bundle["weights"]["lgbm"]
        """,
        language="python",
    )

    st.markdown(
        """
        ✅ Always uses the **latest registered model**  
        ✅ No manual redeployment of model files  
        ✅ CI/CD only updates the MLflow Registry entry  
        """
    )

    st.markdown("### 📥 Request Body Schema")

    st.code(
        """
class InputData(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: float
    Delivery_person_Ratings: float
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: float
    Festival: str
    City: str
        """,
        language="python",
    )

    st.markdown(
        "This Pydantic model ensures that every request is validated **before** it hits the model."
    )

    st.markdown("### 🔄 Prediction Flow")

    st.code(
        """
@app.post("/predict")
def predict(data: InputData):
    raw_df = pd.DataFrame([data.dict()])

    cleaned_df = perform_data_cleaning(raw_df)
    if cleaned_df.empty:
        return {"error": "Input cleaning removed the row (invalid values)."}

    X = preprocessor.transform(cleaned_df)

    pred_cat = cat_model.predict(X)
    pred_lgb = lgb_model.predict(X)

    final_pred = float((w_cat * pred_cat) + (w_lgb * pred_lgb))

    return {
        "predicted_time_minutes": final_pred,
        "model_version_used": latest_ver,
        "weights": {"catboost": w_cat, "lightgbm": w_lgb}
    }
        """,
        language="python",
    )

    st.success(
        "The API returns predictions in **milliseconds**, making it ideal for real-time ETA estimation."
    )

    st.markdown("### ▶️ Running the API Locally")

    st.code("uvicorn app:app --host 0.0.0.0 --port 8000 --reload", language="bash")

    st.info("Open **http://localhost:8000/docs** for Swagger UI testing.")

    # =============================
    # FASTAPI VS FLASK CARD
    # =============================
    st.markdown(
    """
    <div style="
    background-color:#fff5e8;
    padding:22px 26px;
    border-radius:16px;
    border:1px solid #f7cfa2;
    box-shadow:0 4px 12px rgba(0,0,0,0.1);
    margin-top:28px;
    max-width:850px;
    ">

    <h3 style="color:#8a4b00; margin-bottom:12px;">
    ⚡ Why I Used FastAPI Instead of Flask
    </h3>

    <p style="color:#222; font-size:16px; line-height:1.55;">
    FastAPI was selected because it provides 
    <b>high performance, strict validation, async support,</b>
    and a cleaner development experience.
    </p>

    <ul style="color:#222; font-size:15px; line-height:1.6;">
    <li><b>🚀 Faster</b> — ASGI + Uvicorn. Perfect for ML inference.</li>
    <li><b>📘 Auto Docs</b> — Swagger UI + Redoc instantly available.</li>
    <li><b>🛡 Pydantic</b> — Strong, strict validation.</li>
    <li><b>⚙ Async</b> — Handles high traffic better than Flask.</li>
    <li><b>🧰 DevOps Friendly</b> — Great with Docker & CI/CD.</li>
    <li><b>💡 Cleaner Codebase</b> — Less boilerplate.</li>
    </ul>

    <p style="color:#222; font-size:15px; margin-top:10px;">
    For real-time predictions needing <b>speed, reliability, scalability</b>,
    FastAPI was the superior choice.
    </p>

    </div>
    """,
    unsafe_allow_html=True
    )





def section_aws_deploy():

    st.title("🌥️ AWS Deployment — Auto Scaling, Load Balancer & Docker")

    st.write(
        """
        This section explains how my backend is deployed on AWS using 
        **Auto Scaling Group (ASG)**, **Application Load Balancer (ALB)**, 
        **Launch Templates**, and **Docker running on EC2**.
        """
    )

    # =============================
    # IMPORTANT CARD — High-Level Architecture
    # =============================
    st.markdown(
"""
<div style="
background-color:#eaf3ff;
padding:22px;
border-radius:14px;
border:1px solid #bfd8ff;
box-shadow:0 2px 8px rgba(0,0,0,0.06);
margin-top:16px;
">

<h3 style="color:#083b6e; margin-bottom:10px;">📦 Deployment Architecture Overview</h3>

<p style="color:#111; line-height:1.55;">
My deployment pipeline is fully automated using AWS infrastructure:
</p>

<ul style="color:#111; line-height:1.55; margin-left:12px;">
    <li><b>Auto Scaling Group (ASG)</b> manages EC2 creation and termination</li>
    <li><b>Application Load Balancer (ALB)</b> routes traffic only to healthy instances</li>
    <li><b>Launch Template + User Data</b> runs Docker and starts FastAPI automatically</li>
    <li><b>ECR</b> stores Docker images generated by CI/CD</li>
</ul>

</div>
""",
        unsafe_allow_html=True,
    )

    # =============================
    # NORMAL SECTION — ASG
    # =============================
    st.subheader("1️⃣ Auto Scaling Group (ASG)")
    st.write(
        """
        The Auto Scaling Group handles all backend deployment automatically:

        - Launches EC2 instances based on the Launch Template  
        - Ensures desired capacity  
        - Replaces instances during **Instance Refresh**  
        - Scales out when CPU increases  
        - Scales in to reduce cost  
        """
    )

    # =============================
    # NORMAL SECTION — ALB
    # =============================
    st.subheader("2️⃣ Application Load Balancer (ALB)")
    st.write(
        """
        All API traffic goes through the ALB, not directly to EC2.

        ALB responsibilities:
        - Distribute traffic across EC2 instances  
        - Perform health checks (`/docs`)  
        - Remove unhealthy instances  
        - Support zero-downtime rolling deployments  
        """
    )

    # =============================
    # IMPORTANT CARD — User Data Automation
    # =============================
    st.markdown(
"""
<div style="
background-color:#fff8e8;
padding:22px;
border-radius:14px;
border:1px solid #fde3b0;
box-shadow:0 2px 8px rgba(0,0,0,0.06);
margin-top:20px;
">

<h3 style="color:#8a4b00; margin-bottom:10px;">⚙️ Automatic EC2 Deployment (User Data Script)</h3>

<p style="color:#222; line-height:1.55;">
Every new EC2 instance automatically installs and runs the API using User Data:
</p>

<ul style="color:#222; line-height:1.55; margin-left:12px;">
    <li>Installs Docker + AWS CLI</li>
    <li>Fetches secrets from SSM</li>
    <li>Logs into ECR</li>
    <li>Pulls the latest Docker image</li>
    <li>Runs the FastAPI server on port <b>8000</b></li>
</ul>

<p style="color:#222; line-height:1.55;">
No manual deployment steps are required.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    # =============================
    # NORMAL SECTION — Updating to New Version
    # =============================
    st.subheader("3️⃣ How New Versions Are Deployed (Zero Downtime)")
    st.write(
        """
        When I push new code:

        1. CI/CD builds a new Docker image  
        2. Pushes it to Amazon ECR  
        3. I trigger **Instance Refresh** in ASG  

        ASG then:
        - Terminates one old EC2  
        - Launches a new EC2  
        - New instance pulls latest Docker image  
        - ALB waits until it's healthy  
        - Moves to next instance  

        → **Zero downtime deployment**  
        """
    )

    # =============================
    # NORMAL SECTION — Auto Scaling Behavior
    # =============================
    st.subheader("4️⃣ Auto Scaling During High Load")
    st.write(
        """
        When CPU usage rises during stress tests:

        - ASG automatically launches additional EC2 instances  
        - User Data deploys the API instantly  
        - ALB begins routing traffic to them  

        When traffic decreases — ASG terminates extra instances → **cost savings**.
        """
    )

    # =============================
    # NORMAL SECTION — Summary
    # =============================
    st.subheader("📝 Final Summary")
    st.code(
"""
Git Push → CI/CD builds Docker image → pushes to ECR
→ Instance Refresh triggered
→ ASG launches new EC2
→ User Data deploys Docker container
→ ALB routes traffic only to healthy instances
→ New version goes live with zero downtime
""",
        language="text",
    )
    st.title("🗺️ Deployment Architecture Diagram")

    st.write(
        """
        This diagram shows the complete production workflow for the Delivery Time Estimator.
        It illustrates how code moves from **GitHub → CI/CD → ECR → ASG → EC2 → ALB → Users**.
        """
    )

    ascii_diagram = """
+---------------------------+
|        GitHub Repo        |
+-------------+-------------+
              | Push Code
              v
+---------------------------+
|     GitHub Actions CI     |
|  - Build Docker Image     |
|  - Run Tests              |
|  - Push to ECR            |
+-------------+-------------+
              |
              | Docker Image
              v
+-------------------------------------------+
|        Amazon ECR (Docker Registry)       |
|   513278912561.dkr.ecr.eu-north-1...      |
+-------------+-----------------------------+
              |
              | Pulled automatically by ASG
              v
+--------------------------------------------------------------+
|               AWS Auto Scaling Group (ASG)                   |
|                     delivery-asg                             |
|  - Maintains desired instance count                          |
|  - Launches new EC2 on scale-out                             |
|  - Replaces EC2 on Instance Refresh                          |
+-------------+------------------------------------------------+
              | Launch Template
              v
+----------------------------------------------------------+
|           EC2 Launch Template + User Data               |
|----------------------------------------------------------|
|  User Data Script:                                       |
|   - Install Docker, AWS CLI                              |
|   - Fetch MLflow URI / Keys from SSM                     |
|   - Login to ECR                                          |
|   - Pull latest Docker image                              |
|   - Run FastAPI container on port 8000                   |
+-------------+--------------------------------------------+
              |
              v
+-----------------------------+    +-----------------------------+
|       EC2 Instance #1       |    |       EC2 Instance #2       |
| - Docker: delivery-api      |    | - Docker: delivery-api      |
| - FastAPI /predict /docs    |    | - FastAPI /predict /docs    |
+-------------+---------------+    +-------------+---------------+
              |
              | Registered Targets (Port 8000)
              v
+-------------------------------------------+
|     Application Load Balancer (ALB)       |
|     - Routes Traffic                      |
|     - Health Check: /docs                 |
+-------------+-----------------------------+
              v
+----------------------+
|      End Users       |
|  Browsers / Postman  |
+----------------------+
"""

    st.code(ascii_diagram, language="text")
    
    
def section_stress_test():

    st.title("🔥 Stress Testing (Postman Load Test)")

    # -------------------------
    # WHY STRESS TEST WAS DONE
    # -------------------------
    st.markdown(
"""
<div style="
background-color:#ffece8;
padding:22px;
border-radius:14px;
border:1px solid #ffc4b8;
box-shadow:0 2px 8px rgba(0,0,0,0.06);
margin-top:16px;
">

<h3 style="color:#9b1f00; margin-bottom:10px;">🎯 Purpose of the Stress Test</h3>

<p style="color:#222; line-height:1.55;">
I performed a large-scale stress test to verify whether my 
<b>AWS Auto Scaling Group (ASG)</b> was correctly configured to add 
new EC2 instances under heavy traffic.  
This was essential to ensure my FastAPI backend is <b>reliable, scalable, and production-ready</b>.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------------
    # HOW THE TEST WAS PERFORMED
    # -------------------------
    st.subheader("🧪 How I Performed the Stress Test")

    st.write(
        """
        To generate real production-like traffic, I used **Postman Runner**.

        **Load Setup (Postman):**
        - Sent **10,000 requests** per Postman Runner  
        - Opened **10 Postman windows in parallel**  
        - Total effective traffic: **~100,000 API requests**  
        - All requests hit the **Application Load Balancer → Target Group → ASG → EC2**

        I specifically chose Postman because:
        - It’s simple to simulate large volumes of API calls  
        - It gives clear request success/failure metrics  
        - Easy to run parallel windows for massive load generation  
        """
    )

    # -------------------------
    # RESULTS
    # -------------------------
    st.subheader("📈 What Happened During the Test")

    st.write(
        """
        As the Postman load ramped up:

        - The first EC2 instance’s **CPU spiked above 50%**
        - This crossed the Auto Scaling threshold configured in ASG
        - ASG automatically launched **a second EC2 instance**
        - As load kept increasing, ASG launched **a third EC2 instance**
        - Each new instance:
            - Executed the User Data script  
            - Pulled the latest Docker image  
            - Started the FastAPI server  
        - ALB routed traffic only after each new instance became **Healthy**

        ✔ This confirmed that my auto-scaling setup works **exactly as designed**.
        """
    )

    # -------------------------
    # TARGET GROUP SCALING (1 → 2 → 3)
    # -------------------------
    st.subheader("📸 Auto Scaling Results (1 → 2 → 3 Instances)")

    st.write(
        """
        Below are the screenshots from the **Target Group dashboard**, showing the real-time
        scale-out behavior during the stress test.
        """
    )

    st.image("testasg1.png", caption="🔹 Before Stress Test — 1 EC2 instance (minimum capacity)")
    st.image("testasg2.png", caption="🔹 During Stress Test — 2 EC2 instances (scale-out triggered)")
    st.image("testasg3.png", caption="🔹 High Load Peak — 3 EC2 instances (maximum configured capacity)")

    # -------------------------
    # CLOUDWATCH METRICS
    # -------------------------
    st.subheader("📊 Monitoring During Stress Test")

    st.write(
        """
        CloudWatch metrics clearly showed the system response:

        - CPU utilization increased sharply → triggered Auto Scaling  
        - Request count spiked → confirmed Postman was generating real load  
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            "cpu_ut.png",
            caption="🔹 CPU Utilization Across EC2 Instances (Spike → Scale Out Triggered)",
     
        )

    with col2:
        st.image(
            "request.png",
            caption="🔹 Incoming Request Count Increasing During Stress Test",

        )

    # -------------------------
    # FINAL RESULT
    # -------------------------
    st.subheader("✅ Final Outcome")

    st.write(
        """
        The stress test proved that:

        - ✔ Auto Scaling triggers exactly at the configured CPU threshold  
        - ✔ New EC2 instances automatically deploy the Docker container  
        - ✔ ALB smoothly load-balances traffic between instances  
        - ✔ No downtime occurred during the scale-out  
        - ✔ The backend can handle real production traffic reliably  

        **Conclusion:**  
        My architecture is fully **production-ready**, **self-scaling**, and **resilient** under load.
        """
    )






def section_tech():

    st.title("🧱 Tech Stack & Pipeline")

    # ---------------------------
    # High-level overview card
    # ---------------------------
    st.markdown(
"""
<div style="
background-color:#eef3ff;
padding:22px;
border-radius:14px;
border:1px solid #c8d9ff;
box-shadow:0 2px 8px rgba(0,0,0,0.06);
margin-top:16px;
">

<h3 style="color:#0a2d73; margin-bottom:10px;">🔧 Complete Technology Stack Used in This Project</h3>

<p style="color:#222; line-height:1.55;">
This project integrates <b>Machine Learning, FastAPI, Docker, AWS Auto Scaling, CI/CD, 
Cloud monitoring, and DVC pipelines</b> to build a fully production-ready system.
</p>

</div>
""",
        unsafe_allow_html=True,
    )

    # ---------------------------
    # Backend Tech Stack
    # ---------------------------
    st.subheader("⚙️ Backend & API Technologies")

    st.write(
        """
        - **FastAPI** – High-performance asynchronous API framework  
        - **Pydantic** – Input validation and schema enforcement  
        - **Uvicorn** – ASGI server for FastAPI  
        - **MLflow Model Registry** – Managing model versions  
        - **Pandas / NumPy** – Data transformations  
        - **Weighted Ensemble Model (CatBoost + LightGBM)**  
        """
    )

    # ---------------------------
    # Machine Learning
    # ---------------------------
    st.subheader("🤖 Machine Learning & Experimentation")

    st.write(
        """
        - **LightGBM**, **CatBoost**, **Logistic Regression**, **SVM**, **KNN**, **Random Forest**  
        - **Hyperparameter tuning**  
        - **Preprocessing pipeline** stored and reused  
        - **MLflow tracking** for experiments & model lineage  
        """
    )

    # ---------------------------
    # Deployment & DevOps
    # ---------------------------
    st.subheader("🚀 Deployment & DevOps")

    st.write(
        """
        - **AWS EC2 Auto Scaling Group (ASG)**  
        - **Application Load Balancer (ALB)**  
        - **Launch Template + User Data** for automatic deployment  
        - **Amazon ECR** – Docker image registry  
        - **Docker** – Containerizing the API  
        - **Instance Refresh** – Zero-downtime redeployment  
        - **SSM Parameter Store** – Secure environment variables  
        """
    )

    # ---------------------------
    # Cloud Monitoring
    # ---------------------------
    st.subheader("📊 Monitoring & Observability")

    st.write(
        """
        - **AWS CloudWatch Metrics** – CPU usage, request count  
        - **CloudWatch Logs** – Application logs  
        - **Target Group health checks** – Ensure API responsiveness  
        """
    )

    # ---------------------------
    # CI/CD & Automation
    # ---------------------------
    st.subheader("🔄 CI/CD Automation & Data Versioning")

    st.write(
        """
        - **GitHub Actions** – Build & push Docker images to ECR  
        - **DVC (Data Version Control)** – Reproducible ML pipeline  
        - **YAML-based pipelines** – Training, evaluation, deployment steps  
        - **Automated retraining + registry update**  
        """
    )

    # ---------------------------
    # Load Testing
    # ---------------------------
    st.subheader("🔥 Load & Stress Testing Tools")

    st.write(
        """
        - **Postman Runner** – Sent 100k API requests  
        - Verified auto-scaling behavior under load  
        """
    )

    # ---------------------------
    # Developer Tools
    # ---------------------------
    st.subheader("🛠 Developer Tools Used")

    st.write(
        """
        - **VS Code** – Development  
        - **Git & GitHub** – Version control  
        - **Python Virtual Environments**  
        - **Jupyter Notebook** for experimentation  
        """
    )

def section_repos():

    st.title("🔗 Project Repositories & Author")

    st.write(
        """
        Below are the two main repositories for this project — one for the  
        **production deployment** and one for the **experimentation phase**.
        """
    )

        # ---- Card for Repos ----
    st.markdown(
    """
    <div style="background-color:#f1f5ff;padding:22px;border-radius:14px;
    border:1px solid #c5d4ff;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:12px;">

    <h3 style="color:#0b2e72; margin-bottom:12px;">📦 Project Repositories</h3>

    <ul style="color:#111; line-height:1.55; font-size:16px;">
    <li>
        <b>Production Repository:</b><br>
        <a href="https://github.com/apoorvtechh/delivery_time_estimator" target="_blank">
            github.com/apoorvtechh/delivery_time_estimator
        </a>
    </li>

    <br>

    <li>
        <b>Experimentation Repository:</b><br>
        <a href="https://github.com/apoorvtechh/Swiggy_project_Experimentation" target="_blank">
            github.com/apoorvtechh/Swiggy_project_Experimentation
        </a>
    </li>
    </ul>

    </div>
    """,
        unsafe_allow_html=True,
    )


        # ---- Author Card ----
    st.markdown(
    """
    <div style="background-color:#fff8e8;padding:22px;border-radius:14px;
    border:1px solid #ffddb3;box-shadow:0 2px 8px rgba(0,0,0,0.06);margin-top:18px;">

    <h3 style="color:#9b5300; margin-bottom:10px;">👨‍💻 Author</h3>

    <p style="color:#222; font-size:16px; line-height:1.55;">
    <b>Apoorv Gupta</b><br>
    Email: <a href="mailto:apoorvtechh@gmail.com">apoorvtecgg@gmail.com</a><br>
    GitHub: <a href="https://github.com/apoorvtechh" target="_blank">github.com/apoorvtechh</a>
    </p>

    </div>
    """,
            unsafe_allow_html=True,
        )




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
    
elif section == "🛰️ API Service (FastAPI)":
    section_fastapi()

elif section == "🌥️ AWS Deployment (Auto Scaling)":
    section_aws_deploy()

elif section == "🔮 DVC Pipeline, CI/CD Automation":
    section_predict()
    
elif section == "🧱 Tech Stack & Pipeline":
    section_tech()
elif section == "🔥 Stress Testing (Postman)":
    section_stress_test()
elif section == "🔗 Project Repositories & Author":
    section_repos()

