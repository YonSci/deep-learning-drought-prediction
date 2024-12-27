
# Deep Learning Approaches to Drought Prediction


## **Introduction**


<style>body {text-align: justify}</style>

Welcome to the GitHub repository for **Deep Learning Approaches to Drought Prediction**. This project explores the application of advanced deep learning models to predict drought conditions with improved accuracy and timeliness.  We aim to develop AI-driven solutions that support climate resilience and water resource management.

Drought is a significant natural disaster that impacts agriculture, water resources, ecosystems, and socio-economic activities. Early warning systems play a crucial role in mitigating the impacts of drought by providing timely information to decision-makers. 

Ethiopia frequently experiences devastating droughts that severely impact agricultural production and food security for millions of people. To mitigate the impacts of these extrem events , we need to develop accurate and reliable drought prediction models. This project aims to develop a deep learning-based drought prediction model that uses the state-of-the-art data driven modeling techniques to predict drought conditions with high accuracy at sufficient lag/lead times.

Machine/Deep learning techniques have the potential to revolutionize drought forecasting by improving accuracy and lead time. By leveraging advanced data analytics and deep learning algorithms, more accurate and realable drought prediction models can be developed.

Our work focuses on employing cutting-edge techniques, such as **Recurrent Neural Networks (RNNs)**, **Long Short-Term Memory (LSTM) networks**, **Convolutional Neural Networks (CNNs)**, and **Transformer** models, to analyze complex spatio-temporal patterns of drought. Drought predictions can help decision-makers mitigate the impacts of droughts and plan for sustainable resource allocation.

The overarching goals of the project are:  
1. Prepare and compile features/predictors from various datasets relevant for drought prediction.  
2. Calculate the Standardized Precipitation Index (SPI) as a standardized drought indicator.  
3. Implement and evaluate deep learning models for drought prediction.   
4. Develop a machine learning workflow that is both reproducible and scalable to other regions or predictors. 
5. Develop actionable early warning tool to support decision-making.    

This repository includes:
- **Code and Models:** Implementation of deep learning architectures for drought prediction.
- **Datasets:** Data preprocessing steps and preprocessed datasets used for training and testing of ML/DL models.
- **Documentation:** Step-by-step guides for replicating the experiments.
- **Visualization Tools:** Visualize drought patterns and model outputs.

## Methodology

In this project , we employ a robust and scalable  multi-step approach to develop a machine/deep learning-based drought prediction model. The methodology involves:

![CNN](./image/wirkflow3.png)

### **1. Data Collection and Processing**

#### **Data Collection**

- **Daily precipitation** data was obtained from the **Climate Hazards Group InfraRed Precipitation with Station (CHIRPS)** dataset, covering the period from 1981 to 2020.

- Python script was used to automate the process of downloading from the CHIRPS website and processing the data into a suitable format for analysis.

- The **Climate Data Operator (CDO)** tools were utilized to **clip data to the Ethiopian domain**, **merge files** into consolidated netCDF files, and **aggregate daily data into monthly** time steps.

- Predictor data collection: Monthly average **ERA5-Land reanalysis** data with a 9 km resolution, spanning the period from 1981 to 2020, was utilized as predictor data.

    - Local/regional predictors

        1) Temperature
        2) Precipitation
        3) Soil Temperature
        4) Wind and  Pressure
        5) Vegetation
        6) Radiation and Heat
        7) Evaporation 
        8) Soil Water
        9) Runoff

    - Atmospheric and Oceanic predictors 
    
        - Both atmospheric and oceanic indices were gathered from 1981-2022.

        - Nino region SST indices (Nino 1+2, Nino 3, Nino3.4, Nino 4)  from Extended Reconstruction SSTs Version 5 (ERSSTv5) dataset is collected.

        - Southern Oscillation Index (SOI).
        
        - North Atlantic Oscillation (NAO) Index.
        
        - Indian Ocean Dipole (IOD) Index.

        - The collected atmospheric climate indices include the following:
        
            - Outgoing Long Wave Radiation Equator.
            - Zonal Winds Equator (200mb).
            - Trade Wind Index (zonal) West Pacific (850mb)
            - Trade Wind Index (zonal) Central Pacific  (850mb)
            - Trade Wind Index (zonal) East Pacific  (850mb)
            - Zonally Average Temperature Anomalies ( 500mb)

#### **Preprocessing Steps**
- Handle missing values 
- Ensure temporal alignment of datasets
- **SPI Calculation**
    - NCL code was created to compute the Standardized Precipitation Index (SPI) at various intervals, including SPI1, SPI2, SPI3, SPI4, SPI6, SPI12, SPI15, SPI24 and others.

![CNN](./image/spi3-2015.png)

For example , the above 3-month SPI map for 2015 calculated from CHRIPS dataset and can be used to visualize the drought patterns in the region.

The result from the 3-month SPI map matches the drought patterns observed in the region during 2015. According to USAID, GFDRE, Famine Early Warning System reports:

- North and central/eastern Ethiopia has experienced the worst drought in more than 50 years.  
- The drought affected nearly 10 million Ethiopians.

- In 2015, after a false start, the belg rains came a month late in northern and central Ethiopia and kiremt season was delayed and the rains were erratic and below average.  

- February to May Belg rains were erratic and well below average; and the subsequent June to September Kinemt rains started late and were also significantly below average.

---

### **2. Exploratory Data Analysis (EDA)**

**Exploratory Data Analysis (EDA)** is an analytical approach aimed at uncovering the inherent characteristics of datasets, utilizing **statistical (non-graphical)** and **visualization (graphical)** techniques.

Objective: to gain insights into the data by summarizing its main characteristics:

 - Find patterns
 - Identify outliers
 - Explore the relationship between variables
 - Helps to indentify features (aka feature selection)

Some of the EDA techniques used in this project include:

- Basic Statistical Summary
- Visualization 
    - Histogram and Density Plot
    - Box Plots
    - Violin Plot
    - Time Series Plot and Bar Chart
    - Correlation Analysis
    - Bivariate Relationships (Bivariate Scatter and Pair Plot )
- Automatic EDA Tools

---

### **3. Feature Engineering**

Feature engineering is a machine learning technique of creating new features or modifying existing ones from raw data to enhance the performance of machine learning models. It involves extracting meaningful information from raw data, reducing noise, and transforming variables to make them more suitable for modeling. Some common techniques used in feature engineering include:

1) Domain Knowledge Based Feature Engineering 

2) Time Series Feature Engineering

    - Datetime Features: Create month, seasons, yearly as features 
    - Create cyclic monthly features (seasonality components.)
    - Lag-based Features 

3) Location based features

    - Spatial Encoding/Geohashing
    - Coordinate-Based Transformations (Sine and Cosine Transformation)

4) Scaling and Normalization

5) Mathematical Transformations 

6) Creating Interaction Features (Polynomial Feature Generation)

---

### **4. Feature Selection and Dimensionality Reduction**

The goal of feature selection is to select a subset of the most relevant features that are most useful for the model. The following methods can be used for feature selection:
  - Statistical and filter based methods. 
  
  - Warpper Methods 
  
  - Machine Learning Approaches

The goal of dimensionality reduction is to reduce the number of features while retaining the most important information in the data. The figure bellow shows both the feature selection and dimensionality reduction methods:

![CNN](./image/feature_selection.png)

---

### **5. Baseline Model & Evaluation Metrics**

**Baseline models** can be used to compare the performance of different models. Some common baseline models shown in the figure below:

**Evaluation metrics** are used to measure the performance of a model. Some common evaluation metrics are shown in the figure below:

![CNN](./image/bas_eval.png)

---

### **6. Model Training, Hyperparameter Tuning, Cross-Validation, Prediction, Evaluation, &  Interpretation**

**Model Training:** The following image shows the general catagories of machine/deep learning models used in the tranining process:

![CNN](./image/ml_models.png)

The following image shows detail ML/DL models used in the tranining process:

![CNN](./image/ml_models1.png)


**Hyperparameter Optimization:** Use hyperparameter optimization techniques (e.g., grid search, random search, Bayesian optimization) for improved performance.

**Cross-Validation:** Employ k-fold cross-validation to ensure model robustness.

**Prediction:** Forecast drought severity or likelihood in specific regions.
 
**Evaluation:** Compare predictions against historical data or expert observations.

**Interpretation:** Use techniques like SHAP values or feature importance plots to understand model decisions.


The following image shows the workflow of model training, hyperparameter tuning,cross-validation,prediction, evaluation, and interpretation:

![CNN](./image/dl_method.png)


---

### **7. Model Deployment and Documentation**
**Deployment:** Create APIs or integrate models into dashboards for stakeholders.

**Documentation:** Provide clear guidelines for replicating the workflow and interpreting outputs.

The following image shows the workflow of model deployment and documentation:

![CNN](./image/dep_doc2.png)

---

This methodology ensures accurate and actionable drought predictions while emphasizing reproducibility and scalability, empowering decision-makers with timely insights to address drought challenges.


---


### Comparison of Ground Truth, Persistence, Climatology & CNN predictions

The following image shows the comparison of ground truth, persistence, climatology, and CNN predictions:

![CNN](./image/cnn-predicted.png)

---

**Contact Us**

Teferi D. Demissie | Climate Scientist  | International Livestock Research Institute (ILRI) | t.demissie@cgiar.org | Mobile: +251 944 115131

Yonas Mersha | Data Science Consultant | United Nations Economic Commission for Africa (UNECA) | yonas.yigezu@un.org | Mobile: +251 948216748