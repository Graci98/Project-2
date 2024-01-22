# Probability of Credit Card Default

![Credit Default](Image/credit_default.png)


## Overview
### This project will develop models that predict the probability of credit card default. We will use a dataset that consists of information for 30,000 customers, including whether they ended up defaulting the next month. We will perform exploratory data analysis to derive any important insights from the data that should improve the effectiveness of the models. After the models are developed, we’ll compare their performances and determine which has the most sufficient predictive power.

## Getting Started: 


#### 1. Import the required libraries and read in and clean the dataset:

###### Libraries imported; pandas, pathlib, numpy, matplotlib.pylot, seaborn, sklearn, statsmodel.stats.outliers_influence, statsmodel.api, and scipy. Then we read and cleaned the CSV file "default_credit_card_client" by checking for null values. Also adjusted some of the categories for a cleaner data set. 

###### MARRIAGE Variable:
* Original Categories: 1 = married; 2 = single; 3 = others
* Adjustment: Categorize all instances where 'MARRIAGE' is 0 as "others."

###### EDUCATION Variable:
* Original Categories: 1 = graduate school; 2 = university; 3 = high school; 4 = others
* Adjustment: Categorize all instances where 'EDUCATION' is 0, 5, or 6 as "others."

###### PAY_1 to PAY_6 Variables:
* Original Categories: -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
* Adjustment: Categorize all instances where these variables are -2, -1, or 0 as "pay duly," and adjust "pay duly" from -1 to 0. 


#### Target and Feature Variables:

#### This research employed a binary variable, "DEFAULT" (Yes = 1, No = 0), as the target variable, which designates whether the individual defaulted the following month. This study used the following 23 variables as explanatory variables and the documentation for those variables is as follows:

* LIMIT_BAL: Amount of the given credit (New Taiwan dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.

* SEX: (1 = male; 2 = female).

* EDUCATION: (1 = graduate school; 2 = university; 3 = high school; 4 = others).

* MARRIAGE: Marital status (1 = married; 2 = single; 3 = others).

* AGE: (year).

##### PAY_1 - PAY_6: History of past payment. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one  month; 2 = payment delay for two months; . . .; 9 = payment delay for nine months and above. The data tracked the past monthly payment records (from April to September, 2005) as follows:
*    PAY_1 = the repayment status in September, 2005;
*    PAY_2 = the repayment status in August, 2005;
*    PAY_3 = the repayment status in July, 2005;
*    PAY_4 = the repayment status in June, 2005;
*    PAY_5 = the repayment status in May, 2005;
*    PAY_6 = the repayment status in April, 2005.
    
##### BILL_AMT1 - BILL_AMT6: Amount of bill statement (New Taiwan dollar).
*    BILL_AMT1 = amount of bill statement in September, 2005;
*    BILL_AMT2 = amount of bill statement in August, 2005;
*    BILL_AMT3 = amount of bill statement in July, 2005;
*    BILL_AMT4 = amount of bill statement in June, 2005;
*    BILL_AMT5 = amount of bill statement in May, 2005;
*    BILL_AMT6 = amount of bill statement in April, 2005.
    
##### PAY_AMT1 - PAY_AMT6: Amount of previous payment (New Taiwan dollar).
*    PAY_AMT1 = amount paid in September, 2005;
*    PAY_AMT2 = amount paid in August, 2005;
*    PAY_AMT3 = amount paid in July, 2005;
*    PAY_AMT4 = amount paid in June, 2005;
*    PAY_AMT5 = amount paid in May, 2005;
*    PAY_AMT6 = amount paid in April, 2005.

  

### Exploratory Data Analysis

###### We did some Exploratory Data Analysis (EDA) in our project to identify patterns, trends, and correlations in the dataset. First we separated the data into target and feature variables so we can examine the distribution of each variable. One part of Exploratory data anaylsis is Multivariate analysis where you analyze and interpret interactions between three or more variables in a dataset, which we saw high correlation between the varialbes BILL_AMT1 - BILL_AMT6. We delved in deeper and perfomerd a Bivariate analysis which is where two variables are examined simultaneously in order to look for patterns. The overall correlation matrix shows that these features are strongly correlated. 



### Pre-Processing



## Develop machine learning models to fit the data:

#### Trained and tested three models Logistic regression, Random Forest, and Naive Bayes. 
 
### 1. Logistic Regression Result:

##### Logistic Regression Model Balanced Accuracy Score: 0.63
##### Classification Report
![Logistic Regression Classification Report](Image/lr_classificaiton_report.png)

##### Confusion Matrix
![Logistic Regression Confusion Matrix](Image/lr_confusion_matrix.png)

##### Analysis: 
##### Class 0 (Non-default) Predictions:
##### Precision: 96% (High accuracy in predicting non-default cases).
##### Recall: 83% (Effective identification of actual non-default instances).
##### F1-Score: 89% (Overall performance for non-default class).

##### Class 1 (Default) Predictions:
##### Precision: 31% (Lower precision in predicting default cases).
##### Recall: 68% (Moderate ability to capture actual default instances).
##### F1-Score: 43% (Trade-off between precision and recall for default class).

##### Overall Performance:
##### Accuracy: 81% (Overall correctness of predictions).
##### Macro Average F1-Score : 66% (Unweighted average of class-specific F1-Scores).
##### Weighted Average F1-Score : 84% (Weighted average considering class imbalance).


### 2. Random Forest Results:

##### Random Forest Model Balanced Accuracy Score: 0.65
##### Classification Report
![Randon Forest Classification Report](Image/rf_classificaiton_report.png)

##### Confusion Matrix
![Randon Forest Confusion Matrix](Image/rf_confusion_matrix.png)

##### Analysis: 
##### Class 0 (Non-default) Predictions:
##### Precision: 84% (High precision in predicting non-default cases).
##### Recall: 94% (High ability to identify actual non-default instances).
##### F1-Score: 89% (Overall strong performance for non-default class).

##### Class 1 (Default) Predictions:
##### Precision: 64% (Moderate precision in predicting default cases).
##### Recall: 36% (Limited ability to capture actual default instances).
##### F1-Score: 46% (Moderate overall performance for default class).

##### Overall Performance:
##### Accuracy: 81% (Overall correctness of predictions).
##### Macro Average F1-Score: 67% (Unweighted average of class-specific F1-Scores).
##### Weighted Average F1-Score: 79% (Weighted average considering class imbalance


### 3. Naive Bayes Results: 

##### Naive Bayes Model Balanced Accuracy Score: 0.69
##### Classification Report 
![Naive Bayes Classification Report](Image/nb_classification_report.png)

##### Confusion Matrix
![Naive Bayes Confusion Matrix](Image/nb_confusion_matrix.png)

##### Analysis:

##### Class 0 (Non-default) Predictions:
##### Precision: 86% (High precision in predicting non-default cases).
##### Recall: 83% (Good ability to identify actual non-default instances).
##### F1-Score: 85% (Overall strong performance for non-default class).

##### Class 1 (Default) Predictions:
##### Precision: 48% (Moderate precision in predicting default cases).
##### Recall: 55% (Moderate ability to capture actual default instances).
##### F1-Score: 51% (Moderate overall performance for default class).

##### Overall Performance:
##### Accuracy: 77% (Overall correctness of predictions).
##### Macro Average F1-Score: 68% (Unweighted average of class-specific F1-Scores).
##### Weighted Average F1-Score: 77% (Weighted average considering class imbalance).


### Feature Importance Analysis: 

## Resources:

### Data Source:
* Yeh,I-Cheng. (2016). default of credit card clients. UCI Machine Learning Repository. https://doi.org/10.24432/C55S3H.



    
