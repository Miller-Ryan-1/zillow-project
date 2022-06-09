## Zillow Home Sale Price Predictor

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Project Summary
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

#### Project Objectives
> - Document code, process (data acquistion, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a Jupyter Notebook report.
> - Create modules (acquire.py, wrangle.py) that make your process repeateable.
> - Construct a model to predict customer churn using Regression techniques.
> - Deliver a 5 minute presentation consisting of a high-level notebook walkthrough using your Jupyter Notebook from above; your presentation should be appropriate for your target audience.
> - Answer panel questions about your code, process, findings and key takeaways, and model.

#### Business Goals
> - Identify Key Drivers of Home Sale Price
> - Construct a ML classification model that predicts a home's sales price based on key features.
> - Document your process well enough to be presented or read like a report.

#### Audience
> - Zillow Data Science Team
> - CodeUp Students!

#### Project Deliverables
> - A final report notebook (zillow_final_report.ipynb)
> - A predictor of home price driven by best ML model
> - All necessary modules to make my project reproducible

#### Data Dictionary
- Note: Includes only pre-encoded features of signifigance:

|Target|Datatype|Definition|
|:-------|:--------|:----------|
| selling_price | 51461 non-null: int64 | actual sales price of home, in USD |

|Feature|Datatype|Definition|
|:-------|:--------|:----------|
| bedrooms       | 51461 non-null: int64 | number of bedrooms |
| bathrooms        | 51461 non-null: in64 | number of bathrooms in home |
| sqft       | 51461 non-null: int64 | home interior size, in square feet |
| yearbuilt        | 51461 non-null: int64 | year the home was built |
| fireplaces        | 51461 non-null: int64 | number of fireplaces in home, if any |
| lotsize        | 51461 non-null: int64 | size of property home sits on, in square feet |
| pools        | 51461 non-null: int64 | if the property nas a pool or not (1 or 0) |
| garages       | 51461 non-null: int64 | how many car-garages areas they have |
| fips_name        | 51461 non-null: object | Orange, Ventura or Los Angeles county |

#### Initial Hypotheses

> - **Hypothesis 1 -**
> - Pools increase home value

> - **Hypothesis 2 -** 
> - Larger lot size increases home value

> - **Hypothesis 3 -** 
> - Larger square footage increases home value

> - **Hypothesis 4 -** 
> - More bedrooms increases home value

> - **Hypothesis 5 -** 
> - Orange county has the highest home values, followed by Ventura then Los Angeles

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Executive Summary - Conclusions & Next Steps
<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

> - Problem: Zillow needs to improve our home price prediction models. 
> - Actions: Examined 2017 actual selling price data for 3 Southern California counties, along with features collected by Zillow, to determine the biggest predictors of selling price.  Then built and tested a price prediction model based on these features.
> - Conclusions: Location has the biggest impact on selling price.  Across all geographies, square footage is the best predictor of selling price.  Lot size, having a pool, and number of bedrooms also impacts selling price.  From a modeling standpoint, a 2nd degree polynomial is the best when looking at all the data, however for more granularity, each county has different optimal models.
> - Recommendations: Further divide datasets into price tranches for future modeling.  Re-examine quality scoring as a metric, and possibly expand its use.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Pipeline Stages Breakdown

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

##### Plan
- [x] Create README.md with data dictionary, project and business goals, come up with initial hypotheses.
- [x] Acquire data from the Zillow (Codeup) Database and create a function to automate this process. Save the function in an acquire.py file to import into the Final Report Notebook.
- [x] Clean and prepare data for the first iteration through the pipeline, MVP preparation. Create a function to automate the process, store the function in a prepare.py module, and prepare data in Final Report Notebook by importing and using the funtion.
- [x]  Investigate data, formulate hypotehsis, visualize analsyis and run statistical tests when necessary (ensuring signifigance and hypotheses are created and assumptions met).  Document findings and takeaways.
- [x] Establish a baseline accuracy.
- [x] Train multiple different classification models, to include hyperparameter tuning.
- [x] Evaluate models on train and validate datasets.
- [x] Choose the model with that performs the best and evaluate that single model on the test dataset.
- [x] Create csv file with predictions on test data.
- [x] Document conclusions, takeaways, and next steps in the Final Report Notebook.

___

##### Plan -> Acquire
> - Store functions that are needed to acquire zillow home data from the Codeup data science database server; make sure the acquire.py module contains the necessary imports for anyone with database access to run the code.
> - The final function will return a pandas DataFrame.
> - Import the acquire function from the acquire.py module and use it to acquire the data in the Final Report Notebook.
> - Complete some initial data summarization (`.info()`, `.describe()`, `.value_counts()`, ...).
> - Plot distributions of individual variables.
___

##### Plan -> Acquire -> Prepare/Wrange
> - Store functions needed to wrangle the zillow home data; make sure the module contains the necessary imports to run the code. The final functions (prep.py and splitter.py) should do the following:
    - Split the data into train/validate/test.
    - Handle any missing values.
    - Handle erroneous data and/or outliers that need addressing.
    - Encode variables as needed.
    - Create any new features, if made for this project.
> - Import the prepare functions from the wrangle.py and splitter.py modules and use them to prepare the data in the Final Report Notebook.
___

##### Plan -> Acquire -> Prepare -> Explore
> - Answer key questions, my hypotheses, and figure out the features that can be used in a regression model to best predict the target variable, selling_price. 
> - Run at least 2 statistical tests in data exploration. Document my hypotheses, set an alpha before running the tests, and document the findings well.
> - Create visualizations and run statistical tests that work toward discovering variable relationships (independent with independent and independent with dependent). The goal is to identify features that are related to churn (the target), identify any data integrity issues, and understand 'how the data works'. If there appears to be some sort of interaction or correlation, assume there is no causal relationship and brainstorm (and document) ideas on reasons there could be correlation.
> - Summarize my conclusions, provide clear answers to my specific questions, and summarize any takeaways/action plan from the work above.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model
> - Feature Selection and Encoding: Are there any variables that seem to provide limited to no additional information? If so, remove them.  Also encode any non-numerical features of signifigance.
> - Establish a baseline accuracy to determine if having a model is better than no model and train and compare at least 4 different models.
> - Train (fit, transform, evaluate) multiple models, varying the algorithm and/or hyperparameters you use.
> - Compare evaluation metrics across all the models you train and select the ones you want to evaluate using your validate dataframe.  In this case we used Precision (Positive Predictive Value).
> - Based on the evaluation of the models using the train and validate datasets, choose the best model to try with the test data, once.
> - Test the final model on the out-of-sample data (the testing dataset), summarize the performance, interpret and document the results.
___

##### Plan -> Acquire -> Prepare -> Explore -> Model -> Deliver
> - Introduce myself and my project goals at the very beginning of my notebook walkthrough.
> - Summarize my findings at the beginning like I would for an Executive Summary.
> - Walk the management team through the analysis I did to answer my questions and that lead to my findings. (Visualize relationships and Document takeaways.) 
> - Clearly call out the questions and answers I am analyzing as well as offer insights and recommendations based on my findings.

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

### Reproduce My Project

<hr style="border-top: 10px groove blueviolet; margin-top: 1px; margin-bottom: 1px"></hr>

You will need your own env file with database credentials along with all the necessary files listed below to run my final project notebook. 
- [x] Read this README.md
- [ ] Download the aquire.py, wrangle.py, splitter.py, evaluate.py, explore.py, model_comparator.py and zillow_final_report.ipynb files into your working directory.
- [ ] For more details on the analysis, download the zillow_homeprice_workbook.ipynb file as well as the zillow_quality_feature_workbook.ipynb for an investigation into Zillow's quality score metric.
- [ ] Add your own env file to your directory. (user, password, host)
- [ ] Run the final_report.ipynb notebook

##### Credit to Faith Kane (https://github.com/faithkane3) for the format of this README.md file.