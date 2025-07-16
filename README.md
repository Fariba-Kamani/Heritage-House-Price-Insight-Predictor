# House Price Predictor - A Predictive Regression Model for Predicting the Sale Price of the Houses In Ames, Iowa

[House Price Predictor](https://house-price-predictor-3b59c8aa4c1c.herokuapp.com/)

## Table of Contents

- [Project Setup](#project-setup)
- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis](#hypothesis-and-how-to-validate)
- [Mapping Business Requirements to Data Visualisation and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)

## Project Setup

This project was built using the official [Code Institute Heritage Housing project template](https://github.com/Code-Institute-Solutions/milestone-project-heritage-housing-issues). The template provided a pre-configured development environment, including key tools such as Jupyter Notebooks, a virtual Python environment, and common data science libraries. I used it as a foundation to structure my project, manage dependencies, and streamline development in Codespaces. All template placeholder content has been removed or adapted to reflect the final implementation of my predictive analytics solution.

[Back to top](#table-of-contents)

## Dataset Content

* The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
* The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

[Back to top](#table-of-contents)

## Project Terms & Jargon

- **Client**: The fictional individual (Lydia Doe) who inherited four houses and seeks pricing insights.
- **Property**: A house located in Ames, Iowa, included in the dataset.
- **Sale Price**: The amount a house was sold for. This is what we aim to predict.
- **Attribute (or Feature)**: A characteristic of a house, such as its size, number of bedrooms, or the quality of the kitchen.
- **Prediction**: An estimate of a house’s sale price based on its attributes.

[Back to top](#table-of-contents)

## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

[Back to top](#table-of-contents)

## Hypothesis and how to validate?

As part of the exploratory data analysis (EDA) phase, several hypotheses were formed regarding the factors influencing house prices. These were validated using both statistical correlation analysis (Spearman and Pearson), visual insights from plots, and model-driven feature importance scores from tree-based regressors (ExtraTrees, Random Forest) and regularized linear models (Lasso, Ridge).

* H1: Houses with greater total living area (GrLivArea) are more expensive.

  * Validate:

    * **EDA Insight:** The distribution of GrLivArea is right-skewed, with most homes between 1000–2000 sq ft. Sale prices tend to increase with living area, especially below 4000 sq ft. Outliers beyond this range show more variance.

    * **Correlation:** Strong positive correlation with SalePrice (Spearman: 0.70, Pearson: 0.708).

    * **Model Insight:** Ranked among the top 5 most important features across multiple models.

  * **Conclusion:** Strongly supported — larger living area is one of the most influential predictors of house price.

* H2: Higher overall quality (OverallQual) is associated with higher sale prices.

  * Validate: 

    * **EDA Insight:** Most houses are rated 5–7 in quality. Sale price increases exponentially with quality rating, especially for homes rated 8 and above.

    * **Correlation:** Highest correlation with SalePrice (Spearman: 0.809, Pearson: 0.790).

    * **Model Insight:** Ranked as the top feature in the Extra Trees model and also prominent in Lasso/Ridge.

  * **Conclusion:** Very strongly supported — overall quality is the most powerful predictor of sale price.
* H3: Houses with a garage (GarageArea > 0) sell for higher prices than those without.

  * Validate:

    * **EDA Insight:** A positive relationship is seen between garage size and price, particularly in the 400–800 sq ft range. However, some high-priced houses lack garages, suggesting compensating factors.

    * **Correlation:** Moderate correlation with SalePrice (Spearman: 0.64, Pearson: 0.62).

    * **Model Insight:** Present but not dominant in model rankings.

  * **Conclusion:** Partially supported — garage size contributes to price but is less predictive than living area or quality. Other features can offset the absence of a garage.


[Back to top](#table-of-contents)

## The rationale to map the business requirements to the Data Visualisations and ML tasks

This project addresses two key business requirements defined by the client. Each requirement is mapped to specific data science tasks involving exploratory visual analysis and machine learning modeling:

  * **Business Requirement 1:** Understand how house attributes correlate with sale price

    **Mapped to:**

      * Exploratory Data Analysis (EDA)

      * Correlation analysis (Pearson & Spearman)

      * Visualizations (scatter plots, histograms, boxplots)

      * Predictive Power Score (PPS) matrix

      * Distribution vs SalePrice plots for top features

    **Rationale:**

      * Visual and statistical tools helped identify key drivers of house prices, such as OverallQual, GrLivArea, and GarageArea.

      * Spearman and Pearson correlations quantified the strength of relationships between each feature and the target.

      * The PPS matrix offered additional insight into potential non-linear associations.

      * These findings supported the formation and validation of hypotheses and guided the choice of features for model training.
  
  * **Business Requirement 2:** Predict house sale prices

    **Mapped to:**

      * Machine Learning Regression Task

      * Model training using:

        ExtraTreesRegressor (selected as the final model)

        Lasso and Ridge Regression (to explore feature importance)

        RandomForestRegressor

      * Evaluation using MAE, RMSE, and R² metrics

      * Streamlit app for interactive prediction

    **Rationale:**

      * A broad model search was initially performed using multiple regression algorithms.

      * The top-performing models — ExtraTrees, Random Forest, and Lasso/Ridge — were further tuned and analyzed.

      * ExtraTrees was selected as the final model for its strong predictive performance and robustness.

      * Lasso and Ridge were valuable for identifying the most informative features.

      * The deployed Streamlit app enables the client to estimate sale prices by entering property details, including those of the inherited homes. 

[Back to top](#table-of-contents)

## ML Business Case

**Goal**

The goal of this machine learning task is to predict the sale price (SalePrice) of a house in Ames, Iowa, based on its physical and temporal characteristics. This directly supports the client's second business requirement: determining the combined and individual value of four inherited houses and enabling future real-time predictions for any similar property.

**Problem Framing**

  * ML Task Type: Supervised regression

  * Target Variable: SalePrice (continuous numeric value)

  * Features: House attributes (e.g., size, quality, year built) selected based on data quality, correlation with the target, and domain relevance

  * Training Data: A public dataset with ~1,500 records of residential properties in Ames, Iowa

**Model Output**

Predicts the expected sale price for:

  * Each of the four inherited houses (with known features)

  * Any other house in Ames with similar attributes

  * Enables the client to sum the total value of inherited properties

  * Powers a user-facing dashboard that supports live prediction with custom inputs

**Success Criteria**

  * Primary metric: R² score ≥ 0.75 on both train and test sets

  * Secondary metrics: Low Mean Absolute Error (MAE) and Mean Squared Error (MSE)

**Business Value**

  * Replaces guesswork with data-driven price estimates, avoiding reliance on real estate knowledge from other regions

  * Helps the client maximize profit from selling inherited houses

  * Empowers the client to assess future property opportunities using the same prediction tool

  * Delivers insights on which house features most influence sale price through data visualizations and correlation analysis


[Back to top](#table-of-contents)

## Epics and User Stories
* The project was split into 5 Epics based upon the Data Visualisation and Machine Learning tasks and within each of these, user stories were set out to enable an agile methodology.

### Epic - Information Gathering and Data Collection
  * **User Story** - As a data practitioner, I want to load the Ames Housing dataset from a reliable source, so that I can begin the analysis with a complete dataset.

    * Acceptance Criteria:

      1 - Dataset is successfully loaded into a Pandas DataFrame.

      2 - No file errors or loading issues.

  * **User Story** - As a data practitioner, I want to understand the structure and schema of the dataset, so that I can identify variable types and spot any immediate issues.

    * Acceptance Criteria:

      1- Column names, data types, and unique values are displayed.

      2- Initial inspection reveals types of variables (numerical, categorical, datetime).

  * **User Story** - As a data practitioner, I want to explore missing values and data types, so that I can determine appropriate cleaning strategies.

    * Acceptance Criteria:

      1- Percentage of missing values per column is calculated.

      2- Strategy for handling missing values is documented.

### Epic - Data Visualization, Cleaning, and Preparation

* **User Story** - As a data analyst, I want to visualize the most correlated variables with SalePrice, so that I can meet the client’s requirement to understand how attributes relate to house prices.

    * Acceptance Criteria:

      1- Perform correlation and/or PPS study.

      2- Top 10 correlated variables are visualized against SalePrice using appropriate plots.

* **User Story** - As a data analyst, I want to clean missing values and format data types correctly, so that my dataset is ready for modeling. 

    * Acceptance Criteria:

      1- All missing values are handled (imputed, removed, or flagged).

      2- Columns are in correct formats (e.g., numeric, categorical).
    
* **User Story** - As a data analyst, I want to transform skewed features and encode categorical variables, so that my data meets modeling assumptions.

    * Acceptance Criteria:

      1- Skewness is evaluated and corrected using transformations.

      2- Categorical variables are encoded using suitable encoders.

* **User Story** - As a data analyst, I want to identify the most important features for prediction, so that I can improve model accuracy and meet the dashboard requirements.

    * Acceptance Criteria:

      1- Feature selection is based on correlation or feature importance.

      2- Documented rationale behind selected features.

* **User Story** - As a data scientist, I want to split my dataset into train, validation, and test sets, so that I can evaluate models on unseen data.

    * Acceptance Criteria:

      1- Split ratios and random seed are defined.

      2- Train/test/validation subsets maintain integrity.

### Epic - Model Training, Optimization and Validation

* **User Story** - As a data scientist, I want to train baseline regression models, so that I can choose the most appropriate approach to predict house prices.

    * Acceptance Criteria:

      1- Multiple regression models (e.g., Linear, RandomForest) are trained.

      2- Initial evaluation metrics are calculated and compared.

* **User Story** - As a data scientist, I want to optimize model hyperparameters using cross-validation, so that I can maximize prediction accuracy.

    * Acceptance Criteria:

      1- Hyperparameter optimization is performed.

      2- R2 ≥ 0.75 is achieved on both train and test sets.

* **User Story** - As a data scientist, I want to validate the model’s performance and interpret metrics, so that I can deliver reliable results.

    * Acceptance Criteria:

      1- R2, RMSE, and MAE are reported.

      2- Overfitting or underfitting is addressed.

### Epic - Dashboard Planning, Designing, and Development

* **User Story** - As a dashboard developer, I want to design a dashboard structure that meets the client’s requirements, so that all requested features are implemented.

    * Acceptance Criteria:

      1- Includes a project summary, correlation insights, prediction page for the 4 houses, custom input page, and technical summary.

* **User Story** - As a dashboard developer, I want to visualize the top features affecting SalePrice, so that Lydia understands the most important variables.

    * Acceptance Criteria:

      1- Visuals include bar plots and importance scores.

      2- Key insights labeled clearly.

* **User Story** - As a dashboard developer, I want to create a page where users can enter custom house attributes, so they can get real-time sale price predictions.

    * Acceptance Criteria:

      1- Interactive widgets allow users to input house features.

      2- Predicted price and total for 4 inherited houses are displayed.

* **User Story** - As a dashboard developer, I want to include a technical page showing model performance and pipeline steps, so that advanced users understand the modeling logic.

    * Acceptance Criteria:

      1- Display model pipeline, steps, and metrics.

      2- Optional: visual pipeline flow.

* **User Story** - As a user of the dashboard, I want to input the top five features affecting sale price, so that I can get a quick and reasonably accurate prediction for a property.

    * Acceptance Criteria:

      1- The page displays input widgets for the top 5 most important features (e.g., OverallQual, GrLivArea, etc.).

      2- A "Predict Sale Price" button triggers a prediction using a trained ML pipeline.

      3- The predicted price is displayed clearly in currency format.

      4- The model uses statistical defaults for any missing features not shown to the user.

* **User Story** - As a user of the dashboard, I want to be informed when some features were automatically filled, so that I understand the limits of prediction accuracy.

    * Acceptance Criteria:

      1- If any features are autofilled, a warning is displayed listing them.

      2- The user is informed that more complete data will improve the prediction quality.

      3- The full input used for the prediction (including autofills) is shown in a readable format.

* **User Story** - As a user of the dashboard, I want to optionally provide additional house features, so that the prediction is more accurate when I have more data available.

    * Acceptance Criteria:

      1- A checkbox labeled "Provide more details (optional)" toggles additional feature inputs.

      2- The additional inputs are based on the full set of features used during model training.

      3- If the user does not provide values for some of these, the model uses statistical defaults (median/mode) for them.

      4- Only missing (non-input) features are autofilled — the rest reflect user values.

* **User Story** - As a client or stakeholder, I want to view the project hypotheses and their validation outcomes, so that I can understand whether the data supports the initial assumptions.

    * Acceptance Criteria:

      1- The page clearly states the original hypotheses made at the beginning of the project.

      2- Each hypothesis is followed by a brief explanation of whether it was validated, refuted, or partially supported by the data.

      3- The conclusions are derived from relevant exploratory data analysis or statistical validation steps.

      4- The information is presented in a readable format using Streamlit components (e.g., st.success, st.warning, or st.info).

      5- The page avoids technical jargon and communicates insights clearly to non-technical stakeholders.

### Epic - Dashboard Deployment and Release

* **User Story** - As a developer, I want to deploy the dashboard on a cloud platform, so that the client can access the solution easily.

    * Acceptance Criteria:

      1- Hosted on Heroku.

      2- Functional and responsive.

* **User Story** - As a developer, I want to test the deployed version for performance and usability, so that I can ensure it's reliable.

    * Acceptance Criteria:

      1- Loads in reasonable time.

      2- Works on different screen sizes.

* **User Story** - As a data practitioner, I want to provide a usage guide or README, so Lydia knows how to navigate the dashboard.

    * Acceptance Criteria:

      1- Includes screenshots and usage notes.

      2- Describes limitations.

## Dashboard Design

This Streamlit dashboard was designed to fulfill the business requirements defined at the beginning of the project. It offers a user-friendly, interactive experience that enables the client to explore insights and make predictions regarding house sale prices in Ames, Iowa.

**Navigation Structure**

The sidebar menu provides access to the following dashboard pages:

  * **Project Summary**

    * Brief introduction to the project.

    * Explanation of project terms and domain-specific jargon.

    * Links to full project documentation (e.g., the README).

    * Clear outline of the business requirements.
  
  * **Correlation Insights**

    Display of top correlated variables with SalePrice, based on:

    * Pearson Correlation

    * Spearman Correlation

    * Predictive Power Score (PPS)

    Visualizations:

    * Heatmaps (threshold-masked for clarity)

    * Parallel Categories Plot showing interactions between multiple features and SalePrice

    * Descriptive interpretation of how these variables impact sale price.

  * **Hypothesis**

    Validation of three project hypotheses using:

    * Correlation analysis

    * Feature distributions

    * Visual evidence
  
    Each hypothesis includes a conclusion section on whether it was supported or refuted by the data.
  
  * **Technical Summary**

  * Chosen Model: ExtraTreesRegressor selected based on best cross-validation R² score and low standard deviation.

  * Comparison of baseline models:

    * Lasso: 0.8147

    * Ridge: 0.7762

    * Random Forest: 0.7853

  * Pipeline Overview:

    * Data preprocessing steps

    * Feature selection

    * Model training and evaluation

  * Visual representation of the pipeline flow.

  * Evaluation metrics:

    * R² score on train and test sets

    * MAE/MSE

  * Summary of top 5 most important features contributing to predictions.

  * **Inherited Houses Estimator**

    * Table input of the 4 inherited houses’ attributes.

    * Individual predicted sale prices for each house.

    * Clear summary of the total estimated sale value for the inherited portfolio.
  
  * **House Price Predictions**

    * Live prediction tool to estimate the sale price of any house in Ames, Iowa.

    * Interactive input widgets for:

      * Top 5 most important features

      * Additional optional features (expandable)

  * Warnings shown if default values are used for missing inputs.

  * Output includes:

    * Predicted sale price

    * List of autofilled/defaulted features

    * Overview table of submitted input values


[Back to top](#table-of-contents)

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

[Back to top](#table-of-contents)

## Deployment

### Heroku

* The App live link is: <https://YOUR_APP_NAME.herokuapp.com/>
* Set the .python-version Python version to a [Heroku-24](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

[Back to top](#table-of-contents)

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

[Back to top](#table-of-contents)

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

[Back to top](#table-of-contents)

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

[Back to top](#table-of-contents)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

[Back to top](#table-of-contents)

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

[Back to top](#table-of-contents)

## Bugs:
 * Bug Explanation: ValueError: cannot reindex on an axis with duplicate labels
What Happened:
While comparing the distributions of original vs. cleaned variables, we used the following logic to prepare the data for categorical bar plots:
df1 = pd.DataFrame({"Type": "Original", "Value": df_original[var]})
df2 = pd.DataFrame({"Type": "Cleaned", "Value": df_cleaned[var]})
dfAux = pd.concat([df1, df2], axis=0)
However, pandas.concat() by default preserves the original row indices. Since both df1 and df2 came from the same DataFrame (with identical indices), the resulting dfAux contained duplicate index labels.

When passed to seaborn.countplot(), these duplicate indices led to:
ValueError: cannot reindex on an axis with duplicate labels
This is because Seaborn internally tries to align and scale the data, and non-unique index values cause ambiguity during reindexing operations.
How We Fixed It
To eliminate this ambiguity, we simply reset the index after concatenation:
dfAux = pd.concat([df1, df2], axis=0).reset_index(drop=True)
This ensures that the combined DataFrame has a clean, unique index, which is safe for downstream plotting operations in Seaborn.

Issue:
While running the app, the following error occurred:

StreamlitSetPageConfigMustBeFirstCommandError:
set_page_config() can only be called once per app page, and must be called as the first Streamlit command in your script.

This happened because st.set_page_config() was placed inside the __init__() method of the MultiPage class (multipage.py), which is executed after other Streamlit commands had already run — violating Streamlit’s requirement that set_page_config() must be the first Streamlit-related command executed.

Fix:
To resolve this:

st.set_page_config() was moved to the top of app.py, before any other Streamlit commands or imports that might use Streamlit.

import streamlit as st
st.set_page_config(page_title="House Price Estimator", page_icon="🏘️", layout="centered")

The call to st.set_page_config() was removed from multipage.py to prevent multiple or late calls.


