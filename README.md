# Credit-Card-Default-Prediction
This project aims to build a machine learning model to predict the likelihood of a credit card holder defaulting on their payments. By leveraging historical transaction data and customer information, the model helps financial institutions manage risk and in making bias-free informed lending decisions.

## Executive Summary
This report presents an analysis of credit issuance within the financial sector, using a dataset of 24,000 clients and an additional 1,000 pilot customers. The objectives include making credit issuance recommendations, assessing the impact of gender on decision-making, and discussing broader implications for ethics and equality in artificial intelligence-driven decisions.

In the first section, we develop a predictive model to recommend credit issuance for the 1,000 pilot customers, based on existing client data. The second section explores the role of gender by comparing models with and without the "SEX" variable. We observe differences in credit issuance outcomes when applying models to male and female applicants separately. A graphical representation depicts the percentage of males and females receiving credit across various threshold values. Lastly, we delve into the ethical implications of data-driven decision-making. We emphasize the importance of responsible AI practices and compliance with relevant regulations to ensure fairness and ethical considerations. Recommendations include ongoing monitoring, transparency, and collaboration with stakeholders to align practices with anti-discrimination legislation and ethical standards.

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

![image](https://github.com/user-attachments/assets/aef39514-b83b-4cc9-bc77-f7c88913f99e)
Bivariate Data Analysis: To investigate the correlation between two variables, we employed Tableau to gain valuable insights. Our approach involved the examination of various X variables while applying filters against the target variable, denoted as "Default." The outcomes were numeric, with 1 denoting instances of customer credit defaults (represented in dark blue), and 0 indicating no defaults (illustrated in light blue). In our scrutiny of the distribution of defaults versus no defaults within the Age category, we discerned a descending trend. Nevertheless, the pinnacle for accounts with no defaults at age 29 did not precisely mirror the behavior observed in defaulted accounts. Age displayed a pattern where both default and no default accounts diminished as age advanced. Conversely, the Education category exhibited a similar trend, with both default and no default accounts showcasing a decrease following the peak observed for undergraduate students.

![image](https://github.com/user-attachments/assets/8b209d45-fabd-4cf8-b157-4814b92f6776)
The payment status signifies how often clients settle their balances, which could range from full payments to partial or significant delays. Upon analyzing the payment status for the preceding 1, 3, and 6 months, a consistent trend emerges among accounts with no defaults. These customers tend to make payments exceeding the minimum monthly balance requirement. In contrast, for accounts with defaults, we notice a more erratic trend with no discernible stable pattern of consistently exceeding the minimum monthly payment obligation.
![image](https://github.com/user-attachments/assets/b1811386-afbd-4b25-b7b1-af228ad71d3f)

To establish a connection between monthly balance statements and bill payments, it was crucial to grasp the sequence in which payments are made. We observed that payment amounts correspond to the previous month's statement. Starting with payment amount 1, we compared it to bill amount 2, and so forth.
In the initial month, we noted defaults happening for lower amounts, specifically under $250,000. However, as the monthly balance extended to six months, defaults started occurring on balances exceeding $300,000, with a wider distribution. This indicates that as the balance amounts increase, the likelihood of defaults also increases.

![image](https://github.com/user-attachments/assets/03139255-b5a0-4bd0-8722-bd7e06972d21)

Through the analysis of both bivariate and univariate data, we have successfully identified patterns and correlations among categorical and numerical variables. These insights will serve as a foundation for our upcoming MVP (Minimum Viable Product) model, which will validate our initial findings and guide us in the development of more advanced models through feature engineering.


## Feature Engineering
The feature engineering process is a critical component of this project, aimed at creating meaningful and informative features to improve model performance.

- Handling Missing Data
Before model development, we pre-cleaned the dataset to address missing values. We preserved '0' as a representation for missing data in the Education and Marriage fields, transforming them into categorical variables labeled as 'unknown' and later into dummy variables. This decision was made to ensure seamless integration into the modeling process without introducing complexities.

- Feature Creation
Feature creation is a key differentiator for our model, enhancing its predictive capabilities:

1. Total Bill Amount Over 6 Months (TTL_BILL): Summing up bill amounts for the last six months to provide insights into financial commitment.
2. Total Payment Amount Over 6 Months (TTL_PYMT): Aggregating payment amounts to assess repayment behavior.
3. Total Payment Delay (TTL_PAY): Summing payment delay statuses to reflect payment habits.
4. Payment-to-Bill Ratio (RATIO): Calculating the ratio between total payments and bills to focus on repayment habits.
5. Average Payment Amount (PAY_AMT_AVG): Computing the mean of payment amounts to summarize repayment performance.
6. Payment Trend (PAY_TREND): Assessing trends in payment behavior over time.
7. Bill Percentage Paid (BILL_PCT_PAID2 to BILL_PCT_PAID6): Indicating the percentage of the bill from the next month that was paid in the previous month.
8. Additional granular variables provide deeper insights into customer payment habits:

No Payment Needed to 8 Months Payment Delay (COUNT_PAY_MINUS_2 to COUNT_PAY_8): Counting specific payment behaviors over six months.
Credit Utilization Ratios (CREDIT_UTILIZATION1 to CREDIT_UTILIZATION6): Gauging how effectively customers utilize their credit limit.

- Variable Transformation
Enhancements and transformations improve feature suitability:

- Logarithm of Payment Amounts: Normalizing data and mitigating heteroscedasticity issues.
Pay Category Squared: Capturing nonlinear patterns in payment behavior.
- Categorical Encoding
Adjustments align data types with business and technical requirements, ensuring effective modeling. Dummy variables were generated for all categorical variables, expanding the dataset to 135 variables, enhancing model insights and predictive power.








