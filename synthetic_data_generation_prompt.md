# Synthetic Data Generation Prompt

Use this same prompt in ChatGPT, Claude, Grok, and Gemini. Save each model's output in a different folder:

- ChatGPT → `data/01_raw/Syn_01/`
- Claude → `data/01_raw/Syn_02/`
- Grok → `data/01_raw/Syn_03/`
- Gemini → `data/01_raw/Syn_04/`

Change the target customer count for each run between 100,000 and 500,000.

## Prompt to paste along with Uploaded base files

I have a base synthetic banking recommendation dataset with 15,000 customers. I am uploading the base files as reference.

Create a larger synthetic dataset with the same schemas, same file names, and realistic relationships between files.

Target customer count: randomly between 100000 to 250000

Please generate the actual CSV files and provide them as a downloadable ZIP. Do not print CSV rows in the chat. Use your internal data/file generation tools if available. Only give me the ZIP file and a short summary.

Required CSV files:

1. `customer_demographics.csv`
2. `product_propensity.csv`
3. `product_holdings.csv`
4. `transaction_aggregates.csv`
5. `raw_transactions.csv`
6. `digital_login.csv`
7. `digital_clicks.csv`
8. `product_metadata.csv`
9. `conversion_data_july_2026.csv`

Products:

* PL = Personal Loan
* CC = Credit Card
* HL = Home Loan
* SA = Savings Account
* RD = Recurring Deposit
* MF = Mutual Fund

Business rules to preserve:

* Savings Account should be the most common holding.
* Credit Card should be more common than Home Loan.
* Home Loan should be relatively low frequency.
* HNI / Mass Affluent customers should have higher Mutual Fund and Credit Card probability.
* Younger salaried / digitally active customers should have stronger Credit Card and Personal Loan signals.
* Older / value-seeker / family customers should have stronger RD and savings behavior.
* Less digitally active customers should have higher days since last login and lower login counts.
* Transaction-heavy customers should have higher transaction amounts, higher average ticket size, and stronger product activity.
* Product holdings, propensity, digital behavior, transactions, and conversions should be directionally consistent.
* Include realistic outliers in income, balances, transaction amounts, login behavior, and campaign exposure so outlier treatment can be tested.

Consistency rules:

* All files must use the same customer IDs.
* Customer IDs must be unique and stable across files.
* The generated CSV headers must exactly match the uploaded base files.
* June 2026 should be the feature month.
* July 2026 should be the conversion month.
* `conversion_data_july_2026.csv` should include:

  * `Customer ID`
  * `Date`
  * `Product converted`
* Conversions should mostly be for products the customer did not already hold in June 2026.
* Conversion probabilities should depend on customer profile, propensity, holdings gaps, transactions, and digital behavior.

After generating the ZIP, provide only:

1. Download link to the ZIP
2. Row count summary for each file
3. Basic sanity checks:

   * customer ID consistency
   * holding rates by product
   * conversion count by product
   * percentage of conversions into already-held products
   * whether schemas match the uploaded base files

Do not include the full CSV content in the response.
