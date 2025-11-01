# Pension-Fund-Strategy-Model
A Python machine learning model (RandomForestClassifier) that recommends the optimal salary contribution rate and asset allocation of stocks and bonds for a given age in order to achieve solvency for a pension fund. Another model that recommends the best asset allocation for said pension fund to be sustainable after retirement.

## Table of Contents

- [Purpose](#purpose)
- [Requirements](#requirements)
- [How to use](#how-to-use)

## Purpose

- Simulate pension fund accumulation and retirement sustainability.
- Apply ML to recommend:
  - Optimal salary contribution rates.
  - Optimal stock/bond allocations.
- **Not meant for real-world financial decisions. Purely experimental/educational.**

  
## Requirements

Execute the following command on the console.
`pip install -r requirements.txt`

## How to use

Initial variables such as starting salary, starting age, annuity payout rate etc., are defined at the beginning of the [script](./pension_fund.py) and can be easily modified.

The stock and bond market's mean return and volatility (standard deviation) are normally distributed with easily accessible parameters. 

There are two functions that generate training data for each model. The first function is `simulate_pension_fund_accumulation`, which each year randomly generates the stock/bond ratio with a U(0, 1) distribution and the contribution rate with a U(0, 0.5) distribution. The years span the course of the staring year to retirement. It lastly checks if the pension fund has achieved solvency. Data is stored in the format   `[year, contribution_rate, stock_ratio, solvency]`. 

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `solvency`            | True/False                                                                  |
| `contribution_rate`   | Float between 0 and 1                                                       |
| `stock_ratio`         | Float between 0 and 1                                                       |
| `year`                | Int                                              


The other function `simulate_pension_fund_retirement` randomly generates the stock/bond ratio with a U(0, 1) distribution for each year after retirement and checks if the funs is sustainable until `end_age`. The data is stored in the format `[year, stock_ratio, solvency]`. 

| Feature          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `solvency`       | True/False                                                                  |                                                 
| `stock_ratio`    | Float between 0 and 1                                                       |
| `year`           | Int                                                                         |

The amount of data generated can be edited with the variable `N` on `line 109` . 

Lastly the functions `recommend_investment_strategy` and `recommend_retirement_strategy` make the models respectively suggest the optimal contribution rate and asset  allocation to achieve solvency for the pension fund and best asset allocation for a sustainable fund.





