# Credit-Card-Default-Prediction
This project aims to build a machine learning model to predict the likelihood of a credit card holder defaulting on their payments. By leveraging historical transaction data and customer information, the model helps financial institutions manage risk and in making bias-free informed lending decisions.

## Executive Summary
This report presents an analysis of credit issuance within the financial sector, using a dataset of 24,000 clients and an additional 1,000 pilot customers. The objectives include making credit issuance recommendations, assessing the impact of gender on decision-making, and discussing broader implications for ethics and equality in artificial intelligence-driven decisions.

Firstly, we develop a predictive model to recommend credit issuance for the 1,000 pilot customers, based on existing client data. Lastly, we delve into the ethical implications of data-driven decision-making. We emphasize the importance of responsible AI practices and compliance with relevant regulations to ensure fairness and ethical considerations. Recommendations include ongoing monitoring, transparency, and collaboration with stakeholders to align practices with anti-discrimination legislation and ethical standards.

This report underscores the need for a balance between profitability and ethical decision-making in financial services. Responsible AI practices and continuous evaluation of models are essential to maintain fairness, trust, and equality in credit issuance decisions. 

## Data Collection
The dataset used in this project is the Taiwan Credit Card Default dataset, which includes various features related to customer demographics, payment history, and credit card usage. The dataset is available on UCI Machine Learning Repository. The dataset used in this project is the Taiwan Credit Card Default dataset, which includes various features related to customer demographics, payment history, and credit card usage. The dataset is available on UCI Machine Learning Repository.link https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

![image](https://github.com/user-attachments/assets/8c9752b6-681a-439f-91e5-76ec20351391)

## Data Exploration

Our analysis commenced by focusing on the 'Default' column, which serves as our dependent variable, indicating whether a client is likely to default on their credit line. The analysis initiates with an exploration of both individual and paired data.
Univariate Data Analysis: During this stage, we examined the distribution of both categorical and numerical variables. Among the categorical variables, the 'SEX' distribution unveiled that the dataset primarily comprises female account holders, with a larger proportion being undergraduate students compared to individuals from other educational backgrounds. Furthermore, the dataset predominantly includes single account holders.

![image](https://github.com/user-attachments/assets/dcd8e25e-6758-4c58-8c06-af8ee0d31563)

In our examination of the numerical variables, we initially concentrated on the distribution of the 'LIMIT_BAL' (credit limit), which ranged from $0 to $1 million. Our analysis indicated that the majority of clients held credit limits below $100,000, with the proportion gradually decreasing with each $100,000 increment.
To gain a more profound insight into payment behavior, we conducted a comparative analysis of repayment statuses for the preceding 1 to 6 months. The distribution of repayment statuses for the 1st, 3rd, and 6th months ago unveiled that most customers made payments exceeding the minimum required amount but falling short of settling the entire outstanding balance. This observed pattern mirrors common real-life scenarios within the financial sector.

![image](https://github.com/user-attachments/assets/7aec558c-a4bb-4526-940c-9d2db050e550)

Bivariate Data Analysis: To investigate the correlation between two variables, we employed Tableau to gain valuable insights. Our approach involved the examination of various X variables while applying filters against the target variable, denoted as "Default." The outcomes were numeric, with 1 denoting instances of customer credit defaults (represented in dark blue), and 0 indicating no defaults (illustrated in light blue). In our scrutiny of the distribution of defaults versus no defaults within the Age category, we discerned a descending trend. Nevertheless, the pinnacle for accounts with no defaults at age 29 did not precisely mirror the behavior observed in defaulted accounts. Age displayed a pattern where both default and no default accounts diminished as age advanced. Conversely, the Education category exhibited a similar trend, with both default and no default accounts showcasing a decrease following the peak observed for undergraduate students.

![image](https://github.com/user-attachments/assets/8b209d45-fabd-4cf8-b157-4814b92f6776)
The payment status signifies how often clients settle their balances, which could range from full payments to partial or significant delays. Upon analyzing the payment status for the preceding 1, 3, and 6 months, a consistent trend emerges among accounts with no defaults. These customers tend to make payments exceeding the minimum monthly balance requirement. In contrast, for accounts with defaults, we notice a more erratic trend with no discernible stable pattern of consistently exceeding the minimum monthly payment obligation.
![image](https://github.com/user-attachments/assets/b1811386-afbd-4b25-b7b1-af228ad71d3f)

To establish a connection between monthly balance statements and bill payments, it was crucial to grasp the sequence in which payments are made. We observed that payment amounts correspond to the previous month's statement. Starting with payment amount 1, we compared it to bill amount 2, and so forth.
In the initial month, we noted defaults happening for lower amounts, specifically under $250,000. However, as the monthly balance extended to six months, defaults started occurring on balances exceeding $300,000, with a wider distribution. This indicates that as the balance amounts increase, the likelihood of defaults also increases.

![image](https://github.com/user-attachments/assets/03139255-b5a0-4bd0-8722-bd7e06972d21)

Through the analysis of both bivariate and univariate data, we have successfully identified patterns and correlations among categorical and numerical variables. These insights will serve as a foundation for our upcoming MVP (Minimum Viable Product) model, which will validate our initial findings and guide us in the development of more advanced models through feature engineering.

## MVP Model (Pre-Feature Engineering):
Following the data exploration phase, our primary goal was to assess the models without any feature engineering to establish baseline performance metrics. Before running the four models listed below, we transformed variables such as "SEX," "EDUCATION," "MARRIAGE," and all "PAY_1" through "PAY_6" into categorical variables. Subsequently, we executed the models, and the results are presented as follows. Notably, the GBM (Gradient Boosting Machine) model achieved the highest Area Under the Curve (AUC) at 0.7873 (78.7%). These baseline results will serve as a robust point of comparison as we move forward to incorporate pertinent features into our models.
MVP Model Summary:
![image](https://github.com/user-attachments/assets/757b56fc-02c5-4dfe-81e6-bfcdf152eba5)


## Feature Engineering
The feature engineering process is a critical component of this project, aimed at creating meaningful and informative features to improve model performance.

#### Handling Missing Data
Before model development, we pre-cleaned the dataset to address missing values. We preserved '0' as a representation for missing data in the Education and Marriage fields, transforming them into categorical variables labeled as 'unknown' and later into dummy variables. This decision was made to ensure seamless integration into the modeling process without introducing complexities.

#### Feature Creation
Feature creation is a key differentiator for our model, enhancing its predictive capabilities:

1. Total Bill Amount Over 6 Months (TTL_BILL): Summing up bill amounts for the last six months to provide insights into financial commitment.
2. Total Payment Amount Over 6 Months (TTL_PYMT): Aggregating payment amounts to assess repayment behavior.
3. Total Payment Delay (TTL_PAY): Summing payment delay statuses to reflect payment habits.
4. Payment-to-Bill Ratio (RATIO): Calculating the ratio between total payments and bills to focus on repayment habits.
5. Average Payment Amount (PAY_AMT_AVG): Computing the mean of payment amounts to summarize repayment performance.
6. Payment Trend (PAY_TREND): Assessing trends in payment behavior over time.
7. Bill Percentage Paid (BILL_PCT_PAID2 to BILL_PCT_PAID6): Indicating the percentage of the bill from the next month that was paid in the previous month.

   ![image](https://github.com/user-attachments/assets/b1a34e4a-45a7-413e-bd89-ebafb16a7bcf)
   ![image](https://github.com/user-attachments/assets/e2c197d2-5cf6-4051-969e-45105f2705d9)


9. Additional granular variables provide deeper insights into customer payment habits:

No Payment Needed to 8 Months Payment Delay (COUNT_PAY_MINUS_2 to COUNT_PAY_8): Counting specific payment behaviors over six months.
Credit Utilization Ratios (CREDIT_UTILIZATION1 to CREDIT_UTILIZATION6): Gauging how effectively customers utilize their credit limit.

![image](https://github.com/user-attachments/assets/035e19dc-9f0e-40c7-8bf7-5c383a6daa75)


#### Variable Transformation
The enhancements made to certain existing variables, along with the introduction of additional transformations, are instrumental in enriching our feature set and boosting the predictive value of our model. Here's an overview of these transformations:
- Logarithm of Payment Amounts (LOG_PAY_AMT1 to LOG_PAY_AMT6, LOG_LIMIT_BAL, LOG_TTL_PYMT, and LOG_TTL_BILL): The logarithmic transformation of payment amounts, credit limits, total payments, and total bills serves multiple purposes. It aids in normalizing the data, mitigating issues related to heteroscedasticity, and bringing out patterns that might not be discernible in the original scale. This transformation enhances the interpretability of these variables and can improve their suitability for modeling.
- Pay Category Squared (PAY_1_SQR to PAY_6_SQR): Squaring the existing payment categories (PAY_1 to PAY_6) provides a unique perspective on the magnitude of payment delays. Higher values in these squared variables indicate more significant delays, allowing us to identify extreme cases and gain a deeper understanding of the severity of payment issues. This transformation adds a nonlinear dimension to these variables, potentially capturing nuanced patterns in payment behavior.
![image](https://github.com/user-attachments/assets/3d300c11-3aac-4705-9eb6-59dd75795afb)

These transformations collectively contribute to a more comprehensive and nuanced feature set, enabling our model to capture intricate relationships and patterns in the data, ultimately enhancing its predictive power and accuracy in assessing credit risk and predicting defaults.

#### Categorical Encoding
The final stage of our feature engineering process focuses on aligning data types with both business requirements and technical considerations. Out of the 75 variables obtained after the initial feature engineering steps, only 9 of them deviate from the desired data types. These variables are shown in the following screenshot. Despite being numerical in nature, we have chosen to represent certain variables as categories or labels rather than numerical values, aligning them with the specific context and relevance they hold within our modeling framework.

![image](https://github.com/user-attachments/assets/aefa019e-9251-47af-897d-2363a9ddd426)

## Model Building and Assessment

Profit: Profit represents the projected profit of a sample of 1000 loans. It takes into account the proportion of true negatives and false negatives, as these individuals would be approved for a loan. The calculation involves multiplying these proportions by the respective revenue or loss associated with their actual default outcome.
Profit=((𝑇𝑁/(𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁))×1000×$1500)−((𝐹𝑁/(𝑇𝑃+𝑇𝑁+𝐹𝑃+𝐹𝑁))×1000 ×$5000)

### Determining the Class Threshold

As the classification models predict a probability value, we must determine the threshold at which we will consider a prediction “positive” (predicted value of 1, in this case indicating a prediction that the individual will default). For the purpose of this analysis, we considered the cost-benefit of an incorrect prediction. A false positive, in which we predict that the individual will default and in fact doesn’t cost us $1500 in lost revenue. A false negative in which we predict that the individual will not default and in fact will, cost us $5000 in unpaid loans. Leveraging these two costs, we calculated the critical fractile.

1500/(1500+5000)=0.23076923

Contrasting this value with the proportion of one’s in the dataset, we observe that this value is slightly (~1%) higher. This value will serve as our initial threshold value during model selection. We will then validate this in our final model by recalculating our success metrics across various thresholds.

## Model Selection
The models in scope for this analysis are Logistic Regression, Random Forest, and Gradient Boosting Machine. Due to high train time and complexity, we did not explore Support Vector Machines or Neural Networks.
1. Logistic Regression
The logit model yields an intuitive and easy to understand set of coefficients to determine feature importance. The initial train/test split yielded an ROC-AUC of 0.785 with a calculated profit score of $519,062. Through recursive feature elimination, the top 20 features were selected and fit to a second logit model yielding and ROC-AUC of 0.766 and profit score of $503,645. Both a reduction from the initial fitted model with all X variables present.

![image](https://github.com/user-attachments/assets/783ebc88-e3e8-4fb3-84cc-e7b9a23b87e2)

3. Random Forest
Random Forests are an ensemble classifier that leverages many decision trees constructed in parallel, which then “vote” on the classification outcome. To determine the optimal hyper parameters for our random forest model we conducted a five-fold cross validation for 6 different values of estimators (trees) from 100 to 600 trees in steps of 100 through grid search. This resulted in an optimized model with 200 trees yielding and ROC-AUC of 0.7767 and a profit score of $511,458.

![image](https://github.com/user-attachments/assets/85e1371f-6bcd-4a23-9302-81d0fae1bc88)

5. Gradient Boosting Machine
Gradient Boosting Machine models are another ensemble classifier that leverage trees. However, unlike a Random Forest, the trees are constructed sequentially and place a higher “weight” on data points not well explained by prior trees. In addition to the number of trees, we also examined several learning rates for our 5 fold cross validation through grid search. The hyper-parameters tested were: # 𝑜𝑓 𝑡𝑟𝑒𝑒𝑠∶ [100,150,200,250,300,350,400,500,600] 𝑙𝑒𝑎𝑟𝑛𝑖𝑛𝑔 𝑟𝑎𝑡𝑒: [0.01,0.1,0.15]
The resulting optimized parameters for this model were 150 trees and 0.1 (10%) learning rate. This produced an ROC-AUC of 0.7915 and a profit score of $543,229.

![image](https://github.com/user-attachments/assets/d4b959b8-8a51-4c00-910a-663563de15c7)

## Model Selection:
Certainly, a comparative study of model performance is essential to select the best-performing model. Please provide the performance summary table, and I'll be happy to assist you with the analysis and model selection based on the provided metrics.

![image](https://github.com/user-attachments/assets/0ce3433d-6ee8-4d16-8103-2b6b4ab9e112)

- Validating Class Threshold
Finally, in assessing that we have selected the optimal threshold to maximize our primary scoring metric (profit), we calculated the profit across various thresholds using our final model. This confirmed that the threshold that yields the highest profit is the previously calculated value of 0.23076923.

![image](https://github.com/user-attachments/assets/9f26ba3f-a940-4d94-b885-87cea57b8cac)


# Ethical Implications of Data-Driven Decision-Making in Credit Risk Assessment
The analysis of predictive models for credit card default prediction highlights several important considerations in the context of equality, anti-discrimination, and ethics in data-driven decision-making. These considerations are crucial for ensuring that financial institutions not only optimize their operations but also uphold ethical standards and fairness.

Equality and anti-discrimination have long been pivotal concerns in modern society. Historically, the spotlight was primarily on human decision makers, addressing questions about how to prevent humans from discriminating against one another. Discrimination, a subset of broader ethical considerations, was at the forefront of discussions regarding fair and ethical behavior in decision-making processes.
To contextualize these concerns, it's important to note that many leading economies have implemented policies to combat discrimination in various sectors, including financial services. These decisions spark mixed reactions, with some viewing it as a triumph for gender equality, while others expressed concerns about its implications for business practices. Now, let's focus on the analysis conducted to see the implications for the debate on equality, anti-discrimination, and data-driven decision-making:
1. Identification of Bias: The analyses conducted shed light on the potential presence of gender-based bias within the credit assessment model. These findings underscore the importance of vigilance in identifying and addressing biases that can inadvertently seep into data-driven decision-making processes.
2. Inequality Awareness: The observed disparities between genders in credit predictions highlight the fact that data-driven algorithms, if not carefully designed and monitored, can perpetuate and even exacerbate existing inequalities.
The data consistently demonstrates that male applicants exhibit higher default rates compared to female applicants, regardless of whether the "SEX" variable is included or excluded. This is statistically significant as it reflects a potential bias in credit decisions against males. Notably, the gender-based difference in default rates diminishes when the "SEX" variable is excluded from the model. This is statistically significant as it indicates that the inclusion of gender information influences the model's predictions, contributing to gender-based disparities.
3. Fairness and Equity: The analyses underscore the broader debate on fairness and equity in data-driven decision-making. While algorithms are designed to make objective decisions, they can inadvertently perpetuate biases present in historical data. The statistical evidence confirms the existence of gender bias within the credit provision model, particularly when gender information is included. This poses
ethical and legal challenges as it suggests that the model may inadvertently discriminate against certain
genders, violating principles of anti-discrimination.

The analyses underscore the significant societal impact of data-driven decisions, particularly in financial services. Biased credit assessments can affect access to financial resources, economic opportunities, and overall quality of life. Addressing bias in this context is not just an ethical concern but also a social imperative. It encourage a more nuanced examination of how multiple identity factors, such as gender, race, age, and others, interact within data-driven decision-making. This intersectionality highlights the need for more comprehensive anti-discrimination strategies. In conclusion, the analyses presented in this study offer insights that have direct and far-reaching implications for the ongoing debate on equality and anti-discrimination in data-driven decision-making. They emphasize the importance of proactive measures to identify and mitigate bias, transparency in algorithmic processes, adherence to legal frameworks, and a deep commitment to fairness and equity. Ultimately, the goal is to harness the power of data-driven decision-making while ensuring that it serves as a tool for promoting equality rather than perpetuating discrimination.












