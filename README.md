# Credit-Card-Default-Prediction
This project aims to build a machine learning model to predict the likelihood of a credit card holder defaulting on their payments. By leveraging historical transaction data and customer information, the model helps financial institutions manage risk and make informed lending decisions.

867: PREDICTIVE MODELLING
CREDIT DEFAULT MODELLING AT TAIWAN INTERNATIONAL BANK
TEAM STIRLING
BUSINESS REPORT
Ananya, Lois Ye, Kristin Li, Ritu Gautam, Junaid Saeed, Kamal Jazar, Ali Shah Amin
2
TABLE OF CONTENTS
Summary ...................................................................................................................................................3
Model Description ..................................................................................................................................4
1.1 Context ............................................................................................................................................................. 4
1.2 Steps to build a model ...................................................................................................................................... 4
Data Exploration ................................................................................................................................................. 5
Feature Engineering .............................................................................................................................................. 8
Model Building & Assessment .......................................................................................................................... 12
Model Selection................................................................................................................................................ 16
Credit Default Modelling – Case Study .....................................................................188
Question 1 ............................................................................................................................................................ 18
Question 2(a) ........................................................................................................................................................ 18
Question 2(b) ........................................................................................................................................................ 20
Question 2(c) ........................................................................................................................................................ 24
3
SUMMARY
This report presents an analysis of credit issuance within the financial sector, using a dataset of 24,000 clients and an additional 1,000 pilot customers. The objectives include making credit issuance recommendations, assessing the impact of gender on decision-making, and discussing broader implications for ethics and equality in artificial intelligence-driven decisions.
In the first section, we develop a predictive model to recommend credit issuance for the 1,000 pilot customers, based on existing client data. The second section explores the role of gender by comparing models with and without the "SEX" variable. We observe differences in credit issuance outcomes when applying models to male and female applicants separately. A graphical representation depicts the percentage of males and females receiving credit across various threshold values. Lastly, we delve into the ethical implications of data-driven decision-making. We emphasize the importance of responsible AI practices and compliance with relevant regulations to ensure fairness and ethical considerations. Recommendations include ongoing monitoring, transparency, and collaboration with stakeholders to align practices with anti-discrimination legislation and ethical standards.
This report underscores the need for a balance between profitability and ethical decision-making in financial services. Responsible AI practices and continuous evaluation of models are essential to maintain fairness, trust, and equality in credit issuance decisions. Furthermore, you can access the code for this report by following this github repository link.
4
MODEL DESCRIPTION
1.1 CONTEXT
In response to the growing importance of credit risk management and the need to enhance its credit decision-making process, Taiwan International Bank embarked on a comprehensive project to develop a robust credit default prediction model. The objective of this initiative was to leverage advanced data analytics and machine learning techniques to optimize the bank's lending operations while minimizing the risk associated with credit default.
This Model Building Report serves as a detailed documentation of the entire process undertaken in building, assessing, and validating the credit default prediction model. The report encompasses various phases, from data exploration and feature engineering to model development, testing, and ethical considerations. Throughout the report, we delve into the step-by-step methodology employed in crafting this predictive model, emphasizing critical aspects such as data preprocessing, feature selection, and model evaluation. The focus on transparency and accountability remains central, as the bank aims to ensure fairness, accuracy, and compliance with regulatory standards in its credit decision-making process.
Moreover, this report addresses specific aspects of the model's performance, including the evaluation of different algorithms, threshold selection, and extensive testing on pilot datasets. Furthermore, it presents findings related to the impact of gender on the model's predictions, highlighting ethical considerations and potential biases in credit approval decisions.
By providing a comprehensive overview of the model's development and performance, this report equips with valuable insights and information for informed decision-making in the realm of credit risk management.
1.2 STEPS TO BUILD A MODEL
The process of model building involves several crucial steps to construct a predictive model that can effectively address the problem at hand. It typically begins with data exploration, where the dataset is thoroughly examined to understand its structure, features, and any potential data anomalies. This step helps in gaining insights into the data's characteristics and informs decisions about data preprocessing.
Following data exploration, the next step is feature engineering. In this phase, relevant features are selected, and new features may be created or transformed to enhance the model's ability to capture patterns and make accurate predictions. Effective feature engineering can significantly impact the model's performance by providing it with meaningful and informative inputs.
Once the data is preprocessed and features are engineered, the focus shifts to model building and assessment. This step involves selecting an appropriate machine learning algorithm, training the model on the prepared dataset, and fine-tuning its hyper-parameters. Model performance is assessed using suitable evaluation metrics, such as accuracy, precision, recall, or F1-score, depending on the nature of the problem. Iterative adjustments are made to optimize the model's performance.
5
Finally, the model is subjected to testing to evaluate its generalizability. A separate dataset, not previously seen by the model during training and assessment, is used to assess how well the model performs on new, unseen data. This step is crucial to ensure that the model is capable of making accurate predictions on real-world data and not overfitting to the training data. Continuous testing and validation are often part of the model development process to ensure its reliability and effectiveness in practical applications.
A. DATA EXPLORATION:
During our data exploration phase, it was crucial to obtain a comprehensive understanding of each column. We achieved this by referencing the data dictionary and identifying data types, which provided valuable insights into the dataset. To meet our objectives, we conducted analyses on both individual and paired data, assessed the distribution of each column, summarized significant findings to inform effective feature engineering, and subsequently proceeded to develop an MVP product using the pre-engineered data.
Data Analysis:
Our analysis commenced by focusing on the 'Default' column, which serves as our dependent variable, indicating whether a client is likely to default on their credit line. The analysis initiates with an exploration of both individual and paired data.
Univariate Data Analysis: During this stage, we examined the distribution of both categorical and numerical variables. Among the categorical variables, the 'SEX' distribution unveiled that the dataset primarily comprises female account holders, with a larger proportion being undergraduate students compared to individuals from other educational backgrounds. Furthermore, the dataset predominantly includes single account holders.
Figure a. Data
Exploration
b. Feauture
Engineering
c. Model Building
& Assessment
d. Model
Selection
6
In our examination of the numerical variables, we initially concentrated on the distribution of the 'LIMIT_BAL' (credit limit), which ranged from $0 to $1 million. Our analysis indicated that the majority of clients held credit limits below $100,000, with the proportion gradually decreasing with each $100,000 increment.
To gain a more profound insight into payment behavior, we conducted a comparative analysis of repayment statuses for the preceding 1 to 6 months. The distribution of repayment statuses for the 1st, 3rd, and 6th months ago unveiled that most customers made payments exceeding the minimum required amount but falling short of settling the entire outstanding balance. This observed pattern mirrors common real-life scenarios within the financial sector.
Bivariate Data Analysis: To investigate the correlation between two variables, we employed Tableau to gain valuable insights. Our approach involved the examination of various X variables while applying filters against the target variable, denoted as "Default." The outcomes were numeric, with 1 denoting instances of customer credit defaults (represented in dark blue), and 0 indicating no defaults (illustrated in light blue). In our scrutiny of the distribution of defaults versus no defaults within the Age category, we discerned a descending trend. Nevertheless, the pinnacle for accounts with no defaults at age 29 did not precisely mirror the behavior observed in defaulted accounts. Age displayed a pattern where both default and no default accounts diminished as age advanced. Conversely, the Education category exhibited a similar trend, with both default and no default accounts showcasing a decrease following the peak observed for undergraduate students.
The payment status signifies how often clients settle their balances, which could range from full payments to partial or significant delays. Upon analyzing the payment status for the preceding 1, 3, and 6 months, a consistent trend emerges among accounts with no defaults. These customers tend to make payments exceeding the minimum monthly balance requirement. In contrast, for accounts with defaults, we notice a more erratic trend with no discernible stable pattern of consistently exceeding the minimum monthly payment obligation.
7
To establish a connection between monthly balance statements and bill payments, it was crucial to grasp the sequence in which payments are made. We observed that payment amounts correspond to the previous month's statement. Starting with payment amount 1, we compared it to bill amount 2, and so forth.
In the initial month, we noted defaults happening for lower amounts, specifically under $250,000. However, as the monthly balance extended to six months, defaults started occurring on balances exceeding $300,000, with a wider distribution. This indicates that as the balance amounts increase, the likelihood of defaults also increases.
Through the analysis of both bivariate and univariate data, we have successfully identified patterns and correlations among categorical and numerical variables. These insights will serve as a foundation for our upcoming MVP (Minimum Viable Product) model, which will validate our initial findings and guide us in the development of more advanced models through feature engineering.
MVP Model (Pre-Feature Engineering):
Following the data exploration phase, our primary goal was to assess the models without any feature engineering to establish baseline performance metrics. Before running the four models listed below, we transformed variables such as "SEX," "EDUCATION," "MARRIAGE," and all "PAY_1" through "PAY_6" into categorical variables. Subsequently, we executed the models, and the results are presented as follows. Notably, the GBM (Gradient Boosting Machine) model achieved the highest Area Under the Curve (AUC) at 0.7873 (78.7%). These baseline results will serve as a robust point of comparison as we move forward to incorporate pertinent features into our models.
MVP Model Summary: Model Optimized Hyper - Parameters ROC-AUC Accuracy Sensitivity Specificity PPV NPV Profit
Logistic Regression w/o RFE
n/a
0.775
0.779
0.597
0.831
0.501
0.879 $526k
Logistic Regression w/RFE
n/a
0.729
0.753
0.599
0.796
0.455
0.875
$487k
8
Random
Forest
200 trees
0.776
0.729
0.666
0.747
0.428
0.888
$504k Gradient Boosting Machine 150 trees, 10% learning rate 0.787 0.765 0.650 0.798 0.478 0.889 $546k
B. FEATURE ENGINEERING:
In the development of our predictive model, the paramount objective lies in crafting meaningful and informative features from the dataset. This intricate process comprises several pivotal steps, encompassing the handling of missing data, categorical encoding, feature creation, and variable transformation.
Handling Missing Data:
Before embarking on the model development journey, we meticulously pre-cleaned the dataset to ensure its integrity by addressing any missing values. Notably, the value '0' serves as a representation for missing data in the Education and Marriage fields. We have consciously opted to preserve these '0' values, subsequently transforming them into categorical variables. In this context, '0' takes on the role of signifying the 'unknown' category, to later undergo transformation into a dummy variable. Importantly, this strategic decision is anticipated to seamlessly integrate into our modeling process without introducing any complexities.
Feature Creation:
This phase stands as the differentiating factor that sets our model apart from its peers. Our conviction rests on the notion that the existing variables, in isolation, fail to offer the depth necessary for precise default prediction. Consequently, we have embarked on a journey of feature creation, marked by attention to mathematical nuances. Please note, to avoid issues in calculating ratios with 0 denominators and/or log of negative values, we have imputed a constant of 0.0001 This choice is a precautionary measure, aimed at circumventing issues stemming from divisions and logarithmic calculations, and particularly avoiding the risk of encountering infinite results.
Our feature engineering journey commences with the construction of intuitive variables. This includes the derivation of the total bill amount, the total payment amount, and the cumulative sum of payment categories over the preceding six months. Additionally, we have introduced the calculation of the ratio between the total payment amount and bill amount, the computation of average payment amounts, and the formulation of a variable delineating payment trends. These novel features are intricately designed to enrich our model, offering a holistic comprehension of clients' payment behavior and financial dynamics. It is this meticulous feature engineering that truly distinguishes our model, endowing it with a unique advantage in predicting defaults. Certainly, here's a concise description of the new features created during data exploration:
9
•
Total Bill Amount Over 6 Months (TTL_BILL): TTL_BILL sums up the bill amounts for the last six months, providing a comprehensive view of the customer's financial commitment during this period. This feature offers insights into their spending patterns and financial responsibilities.
•
Total Payment Amount Over 6 Months (TTL_PYMT): TTL_PYMT aggregates the payment amounts made over the same six-month period, offering a holistic perspective on the customer's repayment behavior. It aids in assessing how promptly customers settle their bills and manage their financial obligations.
•
Total Payment Delay (TTL_PAY): TTL_PAY provides an overview of the customer's payment behavior by summing the payment delay statuses for six months. Despite being categorical, these statuses effectively reflect payment habits. A higher value indicates a tendency to delay payments, while a lower value signifies a habit of timely payments.
•
Payment-to-Bill Ratio (RATIO): The RATIO feature is calculated by dividing the total payment amount over six months by the total bill amount over the same period. This metric normalizes the impact of loan size, focusing solely on repayment habits.
•
Average Payment Amount (PAY_AMT_AVG): PAY_AMT_AVG calculates the mean (average) of six payment amounts, providing a single numerical representation of overall repayment performance.
•
Payment Trend (PAY_TREND): PAY_TREND assesses the trend in payment behavior over time. It computes the differences in payment amounts between successive months and aggregates these differences to indicate whether a customer's payments are increasing or decreasing over the months.
•
Bill Percentage Paid (BILL_PCT_PAID2 to BILL_PCT_PAID6): These features represent the percentage of the bill from the next month that was paid in the previous month. They offer valuable insights into how promptly customers repay their bills and manage their financial obligations, particularly over multiple months.
These newly engineered features significantly enrich our understanding of customer financial behavior, facilitating more accurate predictions and informed decision-making in credit risk assessment.
Certainly, the creation of more granular variables provides a deeper insight into customer payment habits. Here's a description of these variables:
•
No Payment Needed to 8 Months Payment Delay (COUNT_PAY_MINUS_2 to COUNT_PAY_8): These metrics count the number of times a customer exhibited specific payment behaviors within the last six months. The behaviors include scenarios where a customer did not need to make a payment, paid in full, paid the minimum amount, or delayed payment for n months (where n can be any value from 1 to 8). These variables offer a detailed and comprehensive view of customer payment behavior over the specified six-month period, allowing for a more nuanced assessment of their financial habits and credit risk.
10
The introduction of the CREDIT_UTILIZATION1 to CREDIT_UTILIZATION6 (Credit Utilization Ratios) is a significant enhancement to our feature set. These ratios are computed to gauge how effectively a customer utilizes their credit limit over a period of six months. Specifically, they calculate the proportion of a customer's credit limit that is being utilized by dividing the bill amount by the customer's credit limit for each respective month.
By considering both positive and negative bill amounts (where negative values have been set to '0' to represent potential overpayments), these ratios provide valuable insights into how customers manage their available credit and interact with their credit limits. This granular understanding of credit utilization patterns can play a pivotal role in assessing creditworthiness and predicting default risk.
Variable transformation:
The enhancements made to certain existing variables, along with the introduction of additional transformations, are instrumental in enriching our feature set and boosting the predictive value of our model. Here's an overview of these transformations:
•
Logarithm of Payment Amounts (LOG_PAY_AMT1 to LOG_PAY_AMT6, LOG_LIMIT_BAL, LOG_TTL_PYMT, and LOG_TTL_BILL): The logarithmic transformation of payment amounts, credit limits, total payments, and total bills serves multiple purposes. It aids in normalizing the data, mitigating issues related to heteroscedasticity, and bringing out patterns that might not be discernible in the original scale. This transformation enhances the interpretability of these variables and can improve their suitability for modeling.
•
Pay Category Squared (PAY_1_SQR to PAY_6_SQR): Squaring the existing payment categories (PAY_1 to PAY_6) provides a unique perspective on the magnitude of payment delays. Higher values in these squared variables indicate more significant delays, allowing us to identify extreme cases and gain a deeper understanding of the severity of payment issues. This transformation adds a nonlinear dimension to these variables, potentially capturing nuanced patterns in payment behavior.
11
These transformations collectively contribute to a more comprehensive and nuanced feature set, enabling our model to capture intricate relationships and patterns in the data, ultimately enhancing its predictive power and accuracy in assessing credit risk and predicting defaults.
Categorical Encoding:
The final stage of our feature engineering process focuses on aligning data types with both business requirements and technical considerations. Out of the 75 variables obtained after the initial feature engineering steps, only 9 of them deviate from the desired data types. These variables are shown in the following screenshot. Despite being numerical in nature, we have chosen to represent certain variables as categories or labels rather than numerical values, aligning them with the specific context and relevance they hold within our modeling framework.
This careful data type adjustment ensures that our model effectively captures the underlying relationships and patterns in the data, ultimately contributing to its predictive accuracy and suitability for credit risk assessment and default prediction. Finally, we have generated dummy variables for all the categorical variables.
In summary, AI models can be likened to black boxes. While we have a general understanding of their operations, numerous intricate and undisclosed calculations take place behind the scenes. What may seem like a straightforward process involves an intricate web of complex computations. One aspect that remains within our
12
control is the inputs we provide to these models, which are derived through meticulous feature engineering. In this process, we have undertaken categorical encoding, feature creation, and variable transformation.
As a consequence of these efforts, the dataset has expanded to encompass a total of 135 variables, each contributing to the model's ability to extract valuable insights and make informed predictions.
C. MODEL BUILDING & ASSESSMENT:
Data Preparation
With feature engineering now complete, our data was prepared for modeling by designating the "DEFAULT" column as our target vector, and all remaining columns in the dataframe, excluding "DEFAULT" and "ID," were selected as our X features. Notably, we decided to exclude "ID" as an X variable due to the significant structural differences between the pilot data we intended to predict and the ID values in the training dataset. Modifying these IDs could potentially impact prediction quality. To ensure a balanced approach, we trained the model both with and without "ID" as an X variable. Our observations indicated that "ID" had limited feature importance, and we were willing to accept a modest trade-off in AUC to ensure higher quality predictions on the pilot dataset
To evaluate model performance and establish a common basis for comparison, we employed several metrics based on the confusion matrix. In the context of this binary classification problem, the confusion matrix is a 2x2 matrix representing (1) true negatives, (2) false positives, (3) true positives, and (4) false negatives.
These metrics will be instrumental in assessing the effectiveness of our models and making informed decisions throughout our analysis.
To assess the models, we employed a comprehensive set of metrics, each providing valuable insights into their performance:
1.
ROC-AUC (Receiver Operator Characteristic - Area Under the Curve): ROC-AUC measures the area under the curve of the receiver operating characteristic. It quantifies the model's ability to distinguish between positive and negative cases, with a higher score indicating better discrimination.
2.
Accuracy: Accuracy represents the proportion of correctly predicted samples (both true positives and true negatives) out of the total samples.
Accuracy = 𝑇𝑃+𝑇𝑁𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁
3.
Sensitivity (Recall): Sensitivity, also known as recall, calculates the proportion of correctly predicted positive cases (true positives) relative to the total actual positive cases.
Recall= 𝑇𝑃𝑇𝑃+𝐹𝑁
4.
Specificity (Fall-Out): Specificity assesses the proportion of correctly predicted negative cases (true negatives) relative to the total actual negative cases.
Specificity= 𝑇𝑁𝑇𝑁+𝐹𝑃
5.
Positive Predictive Value (Precision): Precision measures the proportion of correctly predicted positive cases (true positives) relative to the total predicted positive cases.
Precision= 𝑇𝑃𝑇𝑃+𝐹𝑃
13
6.
Negative Predictive Value: Negative Predictive Value quantifies the proportion of correctly predicted negative cases (true negatives) relative to the total predicted negative cases.
Negative Predictive Value=𝑇𝑁𝑇𝑁+𝐹𝑁
7.
Profit: Profit represents the projected profit of a sample of 1000 loans. It takes into account the proportion of true negatives and false negatives, as these individuals would be approved for a loan. The calculation involves multiplying these proportions by the respective revenue or loss associated with their actual default outcome.
Profit=(𝑇𝑁𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁×1000×$1500)−(𝐹𝑁𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁×1000 ×$5000)
While all the above-mentioned metrics were monitored to assess model performance and understand trade-offs, our primary scoring criterion for the models was based on maximizing profit. This decision aligns with our real-world objective of optimizing the model for actual profitability, making it a pivotal metric in our evaluation process.
Determining the Class Threshold
As the classification models predict a probability value, we must determine the threshold at which we will consider a prediction “positive” (predicted value of 1, in this case indicating a prediction that the individual will default). For the purpose of this analysis, we considered the cost-benefit of an incorrect prediction. A false positive, in which we predict that the individual will default and in fact doesn’t cost us $1500 in lost revenue. A false negative in which we predict that the individual will not default and in fact will, cost us $5000 in unpaid loans. Leveraging these two costs, we calculated the critical fractile. 15001500+5000=0.23076923
Contrasting this value with the proportion of one’s in the dataset, we observe that this value is slightly (~1%) higher. This value will serve as our initial threshold value during model selection. We will then validate this in our final model by recalculating our success metrics across various thresholds.
Model Selection
The models in scope for this analysis are Logistic Regression, Random Forest, and Gradient Boosting Machine. Due to high train time and complexity, we did not explore Support Vector Machines or Neural Networks.
1. Logistic Regression
The logit model yields an intuitive and easy to understand set of coefficients to determine feature importance. The initial train/test split yielded an ROC-AUC of 0.785 with a calculated profit score of $519,062. Through recursive feature elimination, the top 20 features were selected and fit to a second logit model yielding and ROC-AUC of 0.766 and profit score of $503,645. Both a reduction from the initial fitted model with all X variables present.
14
Figure: Success metrics and ROC plot for logit model fit on all X features (left) and top 20 features selected through RFE (right)
Figure: Top 20 features selected through recursive feature elimination (RFE)
2. Random Forest
Random Forests are an ensemble classifier that leverages many decision trees constructed in parallel, which then “vote” on the classification outcome. To determine the optimal hyper parameters for our random forest model we conducted a five-fold cross validation for 6 different values of estimators (trees) from 100 to 600 trees in steps of 100 through grid search. This resulted in an optimized model with 200 trees yielding and ROC-AUC of 0.7767 and a profit score of $511,458.
15
Figure: Optimized Random Forest model with 200 trees after 5-fold cross-validation
The optimized model yielded interesting results when assessing feature importance with high “weight” given to age, the sum of payments over the full 6 months, payment category patterns, credit utilization in the most recent months, as well as the credit limit of the individual. These insights indicate the wide range of factors and interaction between them that may help to build intuition as to whether particular patterns impact whether an individual will default.
Figure: Feature importance of the optimized Random Forest model with 200 trees
3. Gradient Boosting Machine
Gradient Boosting Machine models are another ensemble classifier that leverage trees. However, unlike a Random Forest, the trees are constructed sequentially and place a higher “weight” on data points not well explained by prior trees. In addition to the number of trees, we also examined several learning rates for our 5 fold cross validation through grid search. The hyper-parameters tested were: # 𝑜𝑓 𝑡𝑟𝑒𝑒𝑠∶ [100,150,200,250,300,350,400,500,600] 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔 𝑟𝑎𝑡𝑒: [0.01,0.1,0.15]
The resulting optimized parameters for this model were 150 trees and 0.1 (10%) learning rate. This produced an ROC-AUC of 0.7915 and a profit score of $543,229.
16
Figure: Optimized GBM model with X trees and Y learning rate
Similar to the Random Forest Model, examining the feature importance plot provides several insights, regarding the cross section of factors influencing the probability of default in our model.
Figure: Feature importance of the optimized GBM model with 150 trees and 10% learning rate
As the GBM model produces the highest scores in terms of both AUC and profit, we have selected this as our final model to predict default on the pilot dataset.
17
D. MODEL SELECTION:
Certainly, a comparative study of model performance is essential to select the best-performing model. Please provide the performance summary table, and I'll be happy to assist you with the analysis and model selection based on the provided metrics. Model Optimized Hyper-Parameters ROC-AUC Accuracy Sensitivity Specificity PPV NPV Profit
Logistic Regression w/o RFE
n/a
0.79
0.78
0.60
0.83
0.49
0.88
$519k
Logistic Regression w/RFE
n/a
0.77
0.79
0.55
0.90
0.52
0.87
$504k
Random
Forest
200 trees
0.78
0.73
0.67
0.75
0.43
0.89
$511k Gradient Boosting Machine 150 trees, 10% learning rate 0.79 0.77 0.64 0.80 0.48 0.89 $543k
Validating Class Threshold
Finally, in assessing that we have selected the optimal threshold to maximize our primary scoring metric (profit), we calculated the profit across various thresholds using our final model. This confirmed that the threshold that yields the highest profit is the previously calculated value of 0.23076923.
18
CREDIT DEFAULT MODELLING – CASE STUDY
Question 1: Determine which of the 1000 pilot customers should be issued credit. Once done, create a spreadsheet with only one column, A1:A1000, of 0s and 1s, representing your recommendation for issuing credit to each of 1000 pilot customers in the order of their IDs as per the data (1 issue, 0 do not issue).
In response to Question 1, we undertook meticulous data preparation to ensure consistency between the trained dataset and the pilot data. This involved applying the same feature engineering steps, including handling missing values, encoding categorical variables, and data type conversions, to the pilot data.
Subsequently, we leveraged our previously trained Gradient Boosting model, utilizing class thresholds set at 0.23076923, to generate predictions for the pilot data. The model's predictions were binary in nature, with '1' indicating a predicted default and '0' denoting a predicted non-default. Among the 1000 records within the pilot data, our model identified a total of 310 records as predicted defaults based on the patterns it had learned and the predefined threshold values. These predictions serve as our recommendations for issuing credit to each of the 1000 pilot customers, aligning with the order of their IDs as per the data.
The predictions of the chosen model have been saved under the file named "Team Stirling.xlsx" where we have converted the defaulter and non-defaulter predictions to whether the individual should be issued credit or not. That is, if the model predicted 0 means they are non-defaulter and should be issued credit thus, the value should be 1 in the excel sheet. Similarly, if model predicted 1 means they are defaulters and credit should not be issued so, the value will be 0 in the excel.
Question 2(a): Rerun your best model from Q1 with and without the use of the “SEX” variable. Comment on the resultant predictive performance of the two models.
To assess the predictive performance of models with and without the inclusion of the "SEX" variable, we conducted a thorough evaluation by dividing the dataset into training and testing sets and utilizing the Gradient Boosting algorithm. Here are the key findings from our analysis: Performance Metrics Model including “SEX” variable Model excluding “SEX” variable Delta
AUC score
79.136%
79.128%
-0.01%
Accuracy
76.75%
76.75%
0.00%
Sensitivity (aka Recall)
64.185%
64.09%
-0.10%
19
Specificity (aka Fall-Out)
80.316%
80.342%
0.03%
Positive Predictive Value (aka Precision)
48.059%
48.057%
0.00%
Negative Predictive Value
88.768%
88.745%
-0.02%
F1 Score
54.927%
54.964%
0.04%
Projected Profit
$542,604.17
$541,875.00
Loss incurred of $729.17
AUC (Area under the ROC Curve): The model's ability to differentiate between positive and negative classes, as measured by the AUC, remains consistent at 0.791 in both cases, indicating similar discriminatory power.
Figure: ROC curve charts for model with “SEX” (left) and without “SEX” variable (right)
Accuracy: The accuracy of the model remains virtually unchanged in both scenarios, hovering around 76.7%. This suggests that the presence or absence of the "SEX" variable has minimal impact on overall prediction accuracy.
Sensitivity: The model's ability to correctly identify positive cases (customers who will default) is similar in both scenarios, with a sensitivity of approximately 64.1%. This is crucial for minimizing the risk of missing customers who may default.
Specificity: Both models demonstrate strong performance in correctly identifying negative cases (customers who will not default), with a specificity of around 80.3%. This indicates the model's ability to avoid falsely classifying non-defaulters as defaulters.
Precision: The inclusion of the "SEX" variable does not significantly affect precision, as both scenarios yield an approximate 48% correct positive predictions.
Projected Profit: The projected profit of the model is nearly identical in both scenarios, with a slight advantage for the model including "SEX," generating $542,604 compared to $541,875 without the variable.
Our comprehensive evaluation of models with and without the inclusion of the "SEX" variable reveals that the predictive performance remained remarkably consistent across various metrics. Both models demonstrated similar levels of accuracy, sensitivity, specificity, precision, AUC, and projected profit. The "SEX" variable had a
20
minimal influence on the models' ability to predict credit default. However, it is essential to consider the broader ethical and legal implications of including gender information in credit decision models.
Question 2(b). Now apply these two models (i.e., with and without gender) separately to males and females. What do you observe?
To assess whether there is a noteworthy distinction in how the predictive model (GBM) handles males and females based on their predictive probabilities, we have employed a two-sample t-test. This statistical test is commonly used to compare the means of two independent groups or populations, allowing us to determine if there is a substantial difference attributable to random sampling variability or if a genuine divergence exists between the groups. Here is the step-by-step process we followed:
a.
The dataset is split into two distinct groups based on the "SEX" variable, with one group representing males and the other females.
b.
We then predicted the probabilities of credit default for both the male and female groups using the test dataset.
c.
We formulated the null and alternate hypotheses as follows:
Null Hypothesis (H0): There is no significant difference between the two groups.
Alternate Hypothesis (H1): There is a significant difference between the two groups.
d.
We set the alpha level (significance threshold) to 0.05, a commonly used value in statistical testing.
e.
Next, we compared the p-value obtained from the t-test with the alpha level. If the p-value is less than 0.05, we reject the null hypothesis and accept the alternate hypothesis.
f.
Our results indicate that the hypothesis test has revealed a significant difference between the male and female groups, implying that our predictive model is treating them differently.
The results indicate that the model's predictions indeed vary significantly for males and females. The test has validated that ‘SEX’ has statistical significance and each gender subset within the variable is being treated differently by the model. The next step is to evaluate whether the model inherently creates a bias against any gender category i.e., male or female when the ‘SEX’ information is collated as part of data generation process and kept as a dependant variable to predict values.
APPROACH: To assess concerns of gender bias within the model, we initiatied our evaluation process by predicting probabilities for each inidivual. For that we trained and tested the model once with ‘Sex’ as a parameter and another series by excluding the ‘Sex’ variable entirely. We then compared the gender-based results in terms of
21
default and non-default percentages. For example, we examined the percentages of credit issued to males when ‘Sex’ was included in the model versus when it was not.
We performed multiple iterations of the model to predict probabilities while setting different cutoff threshold values (e.g., 13%). If the predicted probability fell below this threshold, they were categorized as a non-defaulter; otherwise, they were categorized as a defaulter. We explored a range of thresholds spanning from 13% to 93%, with a 10% gap between each threshold iteration. We acknowledge that in a real-world scenario, credit issuing organization tend to avoid setting higher threshold levels because it would result in the model categorizing the majority of applications as non-defaulters. This would lead to an unacceptable increase in the false negative percentage, where applicants who would actually default are categorized as non-defaulters by the model. However, our goal was to observe whether there was any discermible trend or impact on specific genders. Through this approach, we aim to establish whether default rates for any specific gender increased, decreased or remain consistent when compared to the other gender. This would help us determine if the model was exhibiting underlying bias when provided with the ‘Sex’ variable information.
ANALYSIS: Based on the predicted probabilities and allocated cut-off threshold, binary values of 0 and 1 were assigned to each pilot applicant. A value of 0 represented non-default, while 1 denoted default. To analyse the influence of gender as a subset of ‘Sex’ variable, we plotted the percentages of default and non-default cases within each gender category, namely male and female, across the different iterated threshold levels.
22
The following observations provide valuable insights:
1.
Male applicants consistently exhibited a higher default percentage across all threshold levels compared to female applicants, in models with ‘Sex’ variable (with gender) and without it (without gender). For example, at a threshold level of 0.23 approximately 37.3% males were predicted to default, while only 26.9% females were predicted to do when ‘Sex’ variable was included. Simultaneously, similar trend was observed with 35.4% males defaulting compared to 26.9% of females in the prediction without ‘Sex’ variable. This suggests that a higher percentage of females have access to credit as compared to males.
2.
The magnitude of the difference in default percentage between the two genders decreased in the predictions without gender dataset across all thresholds. For instance, when average default percentages for males and females across all thresholds were calculated, the values were 19.9% and 14.8% respctively. This indicates an average default disparity between of 5.1%, meaning that, on average, 5.1% more males than females were predicted to default when gender information was included. However, the average default disparity decreased to 3.5% when gender information was omitted, with an average 19.0% males compared to 15.4% of females being predicted to default.
3.
The results align with our intituitive expectation that as threshold levels increase, default percentages decrease, resulting in more credit being issued to both males and females in both datasets. At even higher thresholds levels, such as upto 0.73, higher female percentage of females had access to credit compared to males, and this trend remained consistent between both datasets. Only in the dataset without gender variable a distinction emerged beyond the threshold of 0.73. At 0.83, 0.5% of males were predicted to default versus 0.8% of females.
The model’s prediction of higher loans being disbursed to females reflects the composition of original dataset on which the model was trained. The gender breakdown in the original dataset for both default and non-default is as follows:
Taking into account the insights mentioned earlier, our objective is to delve deeper into the influence of the ‘Sex’ variable on each gender individually when determining probability predictions at each threshold level. To achieve this, we will examine how males and females were treated differently as non-defaulters within their respecgtive gender subsets, both with and without the ‘Sex’ variable.
23
As we deduce from the graph above, the male gender category receives more non-defaults predictions when the ‘Sex’ variable is excluded than when it is included (averaging across all threshold percentages, it’s 81.0% vs 80.1% respecively). The disparity between these percentages at each threshold level diminishes as the threshold level increases, and after reaching 0.73, both sets of prediction percentages exhibit similar behavior in both datasets
In contrast to males, a higher percentage of females gained access to credit when the ‘Sex’ variable was included in the model. This pattern remained consistent for females through out, extending upto the maximum tested threshold level of 0.93, where the credit loans were disbursed to all applicants.
The following graph illustrates the relationship between the percentage deference in non-defaulters with the ‘Sex’ variable and without it in each gender dataset (Delta Bias = Male Prediction with ‘Sex’ variable – Male Prediction without ‘Sex’ variable)
-5.0%
-4.0%
-3.0%
-2.0%
-1.0%
0.0%
1.0%
2.0%
3.0%
0.13 0.23 0.33 0.43 0.53 0.63 0.73 0.83 0.93
BIAS - DIFFERENCE IN PERCENTAGES ACROSS BOTH MODELS
(Δ% = % NON DEFAULT WITH GENDER - % NON DEFAULT WITHOUT GENDER)
Male Female
24
It is clearly evident from the graph above that at lower threshold levels, the model with ‘Sex’ variable gives a biased prediction against the male gender. The model treats male gender in significantly more biased way in the lower threshold levels until 0.43 threshold cut – off is achieved post that it is fairly consistent. Alternatively it is completely trend is completely opposite for the female gender, where in the lower thresholds females gain more access to credit when the gender information is available to the dataset.
Our analysis revealed that the model exhibited some bias in favor of females and against males, particularly at lower threshold levels when the ‘Sex’ variable was included. However, as the threshold increased, the bias diminished which suggests that ‘Sex’ variable influenced the model’s predictions, but it was not the sole factor.
Further investigation is needed to understand the root causes of this bias and ensure fair and equitable credit assessment for all genders. Adjustments to the model or data collection process may be necessary to mitigate this bias and ensure that loan decisions are made without discrimination.
Question 2(c). What are the implications of your analyses for the debate on equality/anti-discrimination/ethics and data-driven decision-making? Be specific: if you suggest something, support it with the analyses.
Equality and anti-discrimination have long been pivotal concerns in modern society. Historically, the spotlight was primarily on human decision makers, addressing questions about how to prevent humans from discriminating against one another. Discrimination, a subset of broader ethical considerations, was at the forefront of discussions regarding fair and ethical behavior in decision-making processes.
To contextualize these concerns, it's important to note that many leading economies have implemented policies to combat discrimination in various sectors, including financial services. These decisions spark mixed reactions, with some viewing it as a triumph for gender equality, while others expressed concerns about its implications for business practices. Now, let's focus on the analysis conducted to see the implications for the debate on equality, anti-discrimination, and data-driven decision-making:
1.
Identification of Bias: The analyses conducted shed light on the potential presence of gender-based bias within the credit assessment model. These findings underscore the importance of vigilance in identifying and addressing biases that can inadvertently seep into data-driven decision-making processes.
2.
Inequality Awareness: The observed disparities between genders in credit predictions highlight the fact that data-driven algorithms, if not carefully designed and monitored, can perpetuate and even exacerbate existing inequalities.
The data consistently demonstrates that male applicants exhibit higher default rates compared to female applicants, regardless of whether the "SEX" variable is included or excluded. This is statistically significant as it reflects a potential bias in credit decisions against males. Notably, the gender-based difference in default rates diminishes when the "SEX" variable is excluded from the model. This is statistically significant as it indicates that the inclusion of gender information influences the model's predictions, contributing to gender-based disparities.
3.
Fairness and Equity: The analyses underscore the broader debate on fairness and equity in data-driven decision-making. While algorithms are designed to make objective decisions, they can inadvertently perpetuate biases present in historical data. The statistical evidence confirms the existence of gender bias within the credit provision model, particularly when gender information is included. This poses
25
ethical and legal challenges as it suggests that the model may inadvertently discriminate against certain
genders, violating principles of anti-discrimination.
The analyses underscore the significant societal impact of data-driven decisions, particularly in financial services. Biased credit assessments can affect access to financial resources, economic opportunities, and overall quality of life. Addressing bias in this context is not just an ethical concern but also a social imperative. It encourage a more nuanced examination of how multiple identity factors, such as gender, race, age, and others, interact within data-driven decision-making. This intersectionality highlights the need for more comprehensive anti-discrimination strategies. In conclusion, the analyses presented in this study offer insights that have direct and far-reaching implications for the ongoing debate on equality and anti-discrimination in data-driven decision-making. They emphasize the importance of proactive measures to identify and mitigate bias, transparency in algorithmic processes, adherence to legal frameworks, and a deep commitment to fairness and equity. Ultimately, the goal is to harness the power of data-driven decision-making while ensuring that it serves as a tool for promoting equality rather than perpetuating discrimination.
