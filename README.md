# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Template Instructions

Welcome,

This is the Code Institute student template for the Heritage Housing project option in Predictive Analytics. We have preinstalled all of the tools you need to get started. It's perfectly ok to use this template as the basis for your project submissions. Click the `Use this template` button above to get started.

You can safely delete the Template Instructions section of this README.md file,  and modify the remaining paragraphs for your own project. Please do read the Template Instructions at least once, though! It contains some important information about the IDE and the extensions we use.

## How to use this repo

1. Use this template to create your GitHub project repo

2. In your new repo click on the green Code button

3. Then, from the Codespaces tab, click Create codespace on main.

5. Wait for the workspace to open. This can take a few minutes.

6. Open a new terminal and `pip3 install -r requirements.txt`

7. Open the jupyter_notebooks directory and click on the notebook you want to open.

8. Click the kernel button and choose Python Environments.

Note that the kernel says Python 3.12.1 as it inherits from the workspace so it will be Python-3.12.1 as installed by Codespaces. To confirm this you can use `! python --version` in a notebook code cell.

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to *Account Settings* in the menu under your avatar.
2. Scroll down to the *API Key* and click *Reveal*
3. Copy the key
4. In your Cloud IDE, from the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with *Regenerate API Key*.

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

## Project Terms & Jargon

- **Client**: The fictional individual (Lydia Doe) who inherited four houses and seeks pricing insights.
- **Property**: A house located in Ames, Iowa, included in the dataset.
- **Sale Price**: The amount a house was sold for. This is what we aim to predict.
- **Attribute (or Feature)**: A characteristic of a house, such as its size, number of bedrooms, or the quality of the kitchen.
- **Prediction**: An estimate of a house’s sale price based on its attributes.


## Business Requirements

As a good friend, you are requested by your friend, who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, to  help in maximising the sales price for the inherited properties.

Although your friend has an excellent understanding of property prices in her own state and residential area, she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

* 1 - The client is interested in discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
* 2 - The client is interested in predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

## Hypothesis and how to validate?

* H1: Houses with greater total living area (GrLivArea) are more expensive.
    * Validate: Plot a scatterplot `sns.scatterplot(x=df['GrLivArea'], y=df['SalePrice'])`
                Calculate correlation `df[['GrLivArea', 'SalePrice']].corr()`
                perform linear regression? ``from sklearn.linear_model import LinearRegression
model = LinearRegression().fit(df[['GrLivArea']], df['SalePrice'])``
Validation Success Criteria: Positive correlation coefficient (e.g., > 0.6).Scatterplot shows upward trend. Regression model gives reasonable R².
* H2: Higher overall quality (OverallQual) is associated with higher sale prices.
    * Validate using boxplot `sns.boxplot(x=df['OverallQual'], y=df['SalePrice'])`
        Group means `df.groupby('OverallQual')['SalePrice'].mean().plot(kind='bar')`
        Calculate correlation `df[['OverallQual', 'SalePrice']].corr()`
Validation Success Criteria: Clear positive trend in boxplot and bar chart. Strong positive correlation (e.g., > 0.6). Statistically significant difference between quality levels.
* H3: Houses with a garage (GarageArea > 0) sell for higher prices than those without.
    * Validate using Create a binary column: `df['HasGarage'] = df['GarageArea'] > 0`
        Compare mean prices: `df.groupby('HasGarage')['SalePrice'].mean()`
        Boxplot comparison: `sns.boxplot(x=df['HasGarage'], y=df['SalePrice'])`
Validation Success Criteria: Houses with garages show higher mean SalePrice. Significant visual and statistical differences.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

* **Business Requirement 1:** Understand which attributes affect sale price.
    * Mapped to: Exploratory data analysis (EDA) using visualisations such as correlation heatmaps, scatter plots, boxplots.
    * Reason: These visualisations reveal patterns and relationships needed for insight and feature selection.
* **Business Requirement 2:** Predict house sale price.
    * Mapped to: Regression ML task using algorithms such as Random Forest, XGBoost, or Linear Regression.
    * Reason: The model can generalise to unseen data and help the client price current and future properties.

## ML Business Case

* In the previous bullet, you potentially visualised an ML task to answer a business requirement. You should frame the business case using the method we covered in the course.
* **Goal:** Predict `SalePrice` (target variable) based on house features (independent variables).
* **Features:** Selected based on correlation with `SalePrice`, data completeness, and domain relevance.
* **Learning Method:** Supervised learning, regression.
* **Model Output:** Predicted sale price as a continuous numeric value.
* **Success Metric:** R² score ≥ 0.85 on test set, and low MAE/MSE.
* **Business Value:** Enables informed decision-making when pricing inherited or future properties.


## Dashboard Design

* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items that your dashboard library supports.
* Eventually, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but eventually you needed to use another plot type)

- **Home / Project Summary**
  - Text summary of the project, user story, business needs

- **Data Visualisation**
  - Correlation heatmap
  - Scatter plots (e.g. GrLivArea vs SalePrice)
  - Boxplots (e.g. Neighborhood vs SalePrice)
  - Interpretation text under each plot


- **Predict Sale Price of Inherited Houses**
  - Table of four inherited houses
  - Display predicted prices with clear messages

- **Live Prediction Tool**
  - Input widgets: Bedrooms, LotArea, OverallQual, etc.
  - Predict button
  - Output section showing predicted price

## Unfixed Bugs

* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

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

## Main Data Analysis and Machine Learning Libraries

* Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
* You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

* The text for the Home page was taken from Wikipedia Article A
* Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
* The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

* The photos used on the home and sign-up page are from This Open Source site
* The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)


* In case you would like to thank the people that provided support through this project.

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


