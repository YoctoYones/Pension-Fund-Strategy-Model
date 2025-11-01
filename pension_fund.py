import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# INITAL ASSUMPTIONS AND SET UP
age0 = 25  # Starting Age
end_age = 80
retirement_age = 67

salary0 = 75000  # Initial Salary
salary_growth = 1.06  # Historically average yearly US salary growth  
final_salary = salary0 * (salary_growth ** (retirement_age - age0))


target_replacement = 0.5  # Target replacement rate of the salaray at retirement
annuity_payout_rate = 0.04
required_bal = final_salary * target_replacement / annuity_payout_rate

# Market return is normally distributied with the parameters
market_stats = {
    'stock': {'mean': 1.07, 'sd': 0.2},
    'bond': {'mean': 1.02, 'sd': 0.045}
}



def simulate_pension_fund_accumulation(required_bal, salary, salary_growth, years, data = [], seed = None):
    ''' 
    Simulates the growth of a pension fund over a specified number of years (until retirement). THe asset allocation between stocks and bonds is randomly determined, as are the contribution rates each year.

    Inputs:     requiered_bal: float
                salary: float
                salary_growth: float
                years: int
                data: list of lists or empty list as default
                seed: int or None as default
    Outputs:     
                data: list of lists with the format [year, contribution_rate, stock_ratio, solvency]

    '''

    fund = 0
    
    rng = np.random.RandomState(seed)

    stock_ratio = np.random.uniform(0, 1)
    market_mean = stock_ratio * market_stats['stock']['mean'] + (1 - stock_ratio) * market_stats['bond']['mean']
    market_sd = (stock_ratio**2 * market_stats['stock']['sd']**2 + (1 - stock_ratio)**2 * market_stats['bond']['sd']**2) ** 0.5
 
    contribution_rates = np.random.uniform(0.05, 0.5, years)

    market_returns = rng.normal(market_mean, market_sd, years)
    for year in range(years):
        fund = fund * market_returns[year] + salary * contribution_rates[year]
        salary *= salary_growth

    solvency = fund >= required_bal

    for year in range(years):
        data.append([year, contribution_rates[year], stock_ratio, solvency])

    return data 

 
def simulate_pension_fund_retirement(required_bal, years, annuity_rate, data = [], seed = None):
    ''' 
    Simulates the drawdown of a pension fund over a specified number of years (during retirement). The asset allocation between stocks and bonds is randomly determined.

    Inputs:     requiered_bal: float
                years: int
                annuity_rate: float
                data: list of lists or empty list as default
                seed: int or None as default
    Outputs:     
                data: list of lists with the format [year, stock_ratio, solvency]

    '''

    fund = required_bal
    
    rng = np.random.RandomState(seed)

    stock_ratio = np.random.uniform(0, 1)
    market_mean = stock_ratio * market_stats['stock']['mean'] + (1 - stock_ratio) * market_stats['bond']['mean']
    market_sd = (stock_ratio**2 * market_stats['stock']['sd']**2 + (1 - stock_ratio)**2 * market_stats['bond']['sd']**2) ** 0.5
 
    withdrawal_rate = annuity_rate

    market_returns = rng.normal(market_mean, market_sd, years)
    for year in range(years):
        fund = fund * market_returns[year] - required_bal * withdrawal_rate

    if fund >= 0:
        solvency = True
    else:
        solvency = False

    for year in range(years):
        data.append([year, stock_ratio, solvency])

    return data


# Model training data
accumulation_data = []
retirement_data = []

N = 10000  # Change this to something larger for better results
for i in range(N):
    accumulation_data = simulate_pension_fund_accumulation(required_bal, salary0, salary_growth, retirement_age - age0, accumulation_data, seed = i)
    retirement_data = simulate_pension_fund_retirement(required_bal, end_age - retirement_age, annuity_payout_rate, retirement_data, seed = i)

accumulation_df = pd.DataFrame(accumulation_data, columns = ['year', 'contribution_rate', 'stock_ratio', 'solvency'])
retirement_df = pd.DataFrame(retirement_data, columns = ['year', 'stock_ratio', 'solvency'])


# Model training
accumulation_X = accumulation_df[['year', 'contribution_rate', 'stock_ratio']]
accumulation_y = accumulation_df['solvency']
retirement_X = retirement_df[['year', 'stock_ratio']]
retirement_y = retirement_df['solvency']

model_accumulation = RandomForestClassifier()
model_accumulation.fit(accumulation_X, accumulation_y)
model_retirement = RandomForestClassifier()
model_retirement.fit(retirement_X, retirement_y)

def recommend_investment_strategy(age, retirement_age, model_accumulation):
    '''
    Recommends an asset allocation based and contribution rate based on age in order to maximize the probability of pension fund solvency.

    Inputs:     age: int (has to be below retirement_age)
                retirement_age: int
                model_accumulation: RandomForestClassifier obj
    Outputs: contribution_rate: float
             stock_ratio: float

    '''

    year = retirement_age - age

    stock_ratio_grid = np.linspace(0.0, 1.0, 100)  
    contrib_grid = np.linspace(0.05, 0.5, 100)  

    combos = pd.DataFrame([(year, cr, sr) for cr in contrib_grid for sr in stock_ratio_grid], columns=['year', 'contribution_rate', 'stock_ratio'])

    combos['prop_solvency'] = model_accumulation.predict_proba(combos)[:, 1]

    best_combo = combos.loc[combos['prop_solvency'].idxmax()]

    return float(best_combo['contribution_rate']), float(best_combo['stock_ratio'])


def recommend_retirement_strategy(age, model_retirement):
    '''
    Recommends an asset allocation during retirement in order to maximize the probability of pension fund solvency.

    Inputs:     age: int (has to be above retirement_age and below end_age)
                model_retirement: RandomForestClassifier obj
    Outputs:    stock_ratio: float

    '''

    year = end_age - age

    stock_ratio_grid = np.linspace(0.0, 1.0, 100)  

    combos = pd.DataFrame([(year, sr) for sr in stock_ratio_grid], columns=['year', 'stock_ratio'])

    combos['prop_solvency'] = model_retirement.predict_proba(combos)[:, 1]

    best_combo = combos.loc[combos['prop_solvency'].idxmax()]

    return best_combo['stock_ratio']


# Example Usage
print(recommend_investment_strategy(40, retirement_age, model_accumulation))
print(recommend_retirement_strategy(68, model_retirement))


