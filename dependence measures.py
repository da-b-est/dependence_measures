#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  1 10:53:14 2025

@author: wujiayi
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2, rankdata, norm, multivariate_normal
from copulas.bivariate import Clayton, Gumbel, Frank
from scipy.stats import ks_2samp

#%%#### Data processing
sp500 = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/sp500.csv')
euro50 = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/euro stoxx 50.csv')
msci_em = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/MSCI EM.csv')
us_10 = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/US 10 year yield.csv')
usd_eur = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/USD_EUR.csv')
gold = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/Gold Futures Historical Data.csv')
brent = pd.read_csv('/Users/wujiayi/desktop/FTD/Econometrie/data/Europe_Brent_Spot_Price_FOB.csv', header=4)

sp500['Date'] = pd.to_datetime(sp500["Date"], format="%m/%d/%Y")
euro50['Date'] = pd.to_datetime(euro50["Date"], format="%m/%d/%Y")
msci_em['Date'] = pd.to_datetime(msci_em["Date"], format="%Y-%m-%d")
us_10['observation_date'] = pd.to_datetime(us_10["observation_date"], format="%Y-%m-%d")
usd_eur['DATE'] = pd.to_datetime(usd_eur["DATE"], format="%Y/%m/%d")
gold['Date'] = pd.to_datetime(gold["Date"], format="%m/%d/%Y")
brent['Day'] = pd.to_datetime(brent["Day"], format="%m/%d/%Y")

column_mapping = {
    "brent": ("Day", "Europe Brent Spot Price FOB  Dollars per Barrel"),
    "euro50": ("Date", "Close/Last"),
    "gold": ("Date", "Price"),
    "msci_em": ("Date", "Value"),
    "sp500": ("Date", "Close/Last"),
    "us_10": ("observation_date", "DGS10"),
    "usd_eur": ("DATE", "Euro/US dollar (EXR.D.USD.EUR.SP00.A)")
}


for df_name, (date_col, value_col) in column_mapping.items():
    globals()[df_name].rename(columns={date_col: "Date", value_col: df_name}, inplace=True)

data = brent
for df_name in ["euro50", "gold", "msci_em", "sp500", "us_10", "usd_eur"]:
    data = data.merge(globals()[df_name], on="Date", how="outer")

data.set_index('Date', inplace=True)
data_filtered= data[data.index >= "2000-01-01"]

log_transform_assets = ["brent", "euro50", "gold", "msci_em", "sp500"]
no_log_transform = ["usd_eur"]

data_log_returns = data_filtered[log_transform_assets].apply(lambda x: np.log(x) - np.log(x.shift(1)))
data_log_returns[no_log_transform] = data_filtered[no_log_transform].pct_change()

data_log_returns['us_10'] = data_filtered['us_10']
data_log_returns = data_log_returns.sort_index()

#%%#### Define the research period
gfc_period = ("2007-06-29", "2009-06-30")  # Global Financial Crisis
covid_period = ("2020-01-21", "2022-12-30")  # COVID-19 Crisis

gfc_data = data_log_returns.loc[gfc_period[0]:gfc_period[1]]
covid_data = data_log_returns.loc[covid_period[0]:covid_period[1]]

gfc_data = gfc_data[['brent', 'usd_eur', 'us_10']]
covid_data = covid_data[['brent','sp500', 'euro50', 'us_10', 'usd_eur']]

gfc_data.ffill(inplace=True)
covid_data.ffill(inplace=True)

#%%#### Compute dependency measures
# Function to compute dependency measures
def compute_dependency_measures(data, period_name):
    pearson_corr = data.corr(method='pearson')
    spearman_corr = data.corr(method='spearman')
    kendall_corr = data.corr(method='kendall')


# Compute for each crisis period
compute_dependency_measures(gfc_data, "GFC (2008)")
compute_dependency_measures(covid_data, "COVID-19 (2020)")


# Plot heatmaps for correlation matrices
def plot_heatmap(corr_matrix, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

# Pearson correlation heatmaps
plot_heatmap(gfc_data.corr(method='pearson'), "Pearson Correlation - GFC (2008)")
plot_heatmap(covid_data.corr(method='pearson'), "Pearson Correlation - COVID-19 (2020)")

# Spearman correlation heatmaps
plot_heatmap(gfc_data.corr(method='spearman'), "Spearman Correlation - GFC (2008)")
plot_heatmap(covid_data.corr(method='spearman'), "Spearman Correlation - COVID-19 (2020)")

# Kendall correlation heatmaps
plot_heatmap(gfc_data.corr(method='kendall'), "Kendall Correlation - GFC (2008)")
plot_heatmap(covid_data.corr(method='kendall'), "Kendall Correlation - COVID-19 (2020)")


#%%#### Trandformation

# Function to transform data to Uniform [0,1] using ECDF
def transform_to_uniform(data):
    return data.apply(lambda x: (rankdata(x) - 0.5) / len(x))

# Transform data for both crisis periods
gfc_uniform = transform_to_uniform(gfc_data)
covid_uniform = transform_to_uniform(covid_data)


#%%#### Fit copulas using Chi-Square fit test

# Custom Gaussian Copula Implementation
class GaussianCopula:
    def __init__(self):
        self.rho = None

    def fit(self, data):
        """Estimate the correlation parameter from data."""
        self.rho = np.corrcoef(data, rowvar=False)[0, 1]

    def cumulative_distribution(self, u):
        """
        Compute CDF of the Gaussian Copula.
        Expects u as a 2D array with shape (1,2).
        """
        if self.rho is None:
            raise ValueError("Copula not fitted yet.")
        cov_matrix = [[1, self.rho], [self.rho, 1]]
        return multivariate_normal.cdf(u, mean=[0, 0], cov=cov_matrix)

# Function to transform data into uniform margins
def transform_to_uniform(data):
    return data.apply(lambda x: (rankdata(x) - 0.5) / len(x))

# Function to compute empirical copula
def compute_empirical_copula(data, K=10):
    """Discretizes the uniform data into K bins and computes empirical copula probabilities."""
    bins = np.linspace(0, 1, K+1)
    empirical_table = np.histogram2d(data.iloc[:, 0], data.iloc[:, 1], bins=[bins, bins])[0]
    return empirical_table / empirical_table.sum()  # Normalize to get probabilities

# Function to compute theoretical copula probabilities
def compute_theoretical_copula(copula, data, K=10):
    """Uses the estimated copula to compute theoretical probabilities."""
    bins = np.linspace(0, 1, K+1)
    epsilon = 1e-6  # Avoid log(0) issues
    copula_probs = np.zeros((K, K))
    
    for i in range(K):
        for j in range(K):
            u1 = np.clip(bins[i+1], epsilon, 1-epsilon)
            v1 = np.clip(bins[j+1], epsilon, 1-epsilon)
            u0 = np.clip(bins[i], epsilon, 1-epsilon)
            v0 = np.clip(bins[j], epsilon, 1-epsilon)
            
            u1_v1 = np.array([[u1, v1]])
            u0_v1 = np.array([[u0, v1]])
            u1_v0 = np.array([[u1, v0]])
            u0_v0 = np.array([[u0, v0]])
            
            cdf_u1_v1 = np.atleast_1d(copula.cumulative_distribution(u1_v1))[0]
            cdf_u0_v1 = np.atleast_1d(copula.cumulative_distribution(u0_v1))[0]
            cdf_u1_v0 = np.atleast_1d(copula.cumulative_distribution(u1_v0))[0]
            cdf_u0_v0 = np.atleast_1d(copula.cumulative_distribution(u0_v0))[0]
            
            copula_probs[i, j] = cdf_u1_v1 - cdf_u0_v1 - cdf_u1_v0 + cdf_u0_v0
            
    return copula_probs

# Function to compute chi-square test statistic
def compute_chi_square(empirical, theoretical):
    mask = empirical > 0  # Avoid division by zero
    return np.sum((empirical[mask] - theoretical[mask])**2 / (theoretical[mask] + 1e-9))

# Function to find the best copula per asset pair
def select_best_copula(data_uniform, period_name, K=10):
    copulas = {
        "Gaussian": GaussianCopula(),
        "Clayton": Clayton(),
        "Gumbel": Gumbel(),
        "Frank": Frank()
    }

    best_copulas = {}  # Store best copula per asset pair
    chi2_results = {}

    # Generate asset pairs
    asset_pairs = [(i, j) for idx, i in enumerate(data_uniform.columns) for j in data_uniform.columns[idx+1:]]

    # Iterate over asset pairs
    for asset_1, asset_2 in asset_pairs:
        pair_data = data_uniform[[asset_1, asset_2]].dropna()
        empirical_copula = compute_empirical_copula(pair_data, K)

        best_copula = None
        best_chi2 = np.inf
        chi2_results[(asset_1, asset_2)] = {}

        # Iterate over copulas
        for name, copula in copulas.items():
            try:
                copula.fit(pair_data.to_numpy())
                theoretical_copula = compute_theoretical_copula(copula, pair_data, K)
                chi2_stat = compute_chi_square(empirical_copula, theoretical_copula)
                chi2_results[(asset_1, asset_2)][name] = chi2_stat

                # Select best copula based on lowest chi-square
                if chi2_stat < best_chi2:
                    best_chi2 = chi2_stat
                    best_copula = name

            except Exception as e:
                print(f"Error computing chi-square for {name} copula ({asset_1} & {asset_2}): {e}")

        best_copulas[(asset_1, asset_2)] = best_copula
        print(f"Best copula for {asset_1} & {asset_2} during {period_name}: {best_copula}")


# Transform data to uniform distribution
gfc_uniform = transform_to_uniform(gfc_data)
covid_uniform = transform_to_uniform(covid_data)

# Select the best copula for each asset pair within each period
select_best_copula(gfc_uniform, "GFC (2008)")
select_best_copula(covid_uniform, "COVID-19 (2020)")


#%%#### Visualization

# Gaussian Copula Implementation
class GaussianCopula:
    def __init__(self):
        self.rho = None

    def fit(self, data):
        """Estimate the correlation parameter from data."""
        self.rho = np.corrcoef(data, rowvar=False)[0, 1]

    def cumulative_distribution(self, u):
        """Compute CDF of the Gaussian Copula."""
        if self.rho is None:
            raise ValueError("Copula not fitted yet.")
        cov_matrix = [[1, self.rho], [self.rho, 1]]
        return multivariate_normal.cdf(u, mean=[0, 0], cov=cov_matrix)

    def sample(self, size=1000):
        """Generate random samples from the fitted Gaussian copula."""
        if self.rho is None:
            raise ValueError("Copula not fitted yet.")
        mean = [0, 0]
        cov = [[1, self.rho], [self.rho, 1]]
        samples = multivariate_normal.rvs(mean, cov, size=size)
        return norm.cdf(samples)  # Convert back to uniform margins

# Function to visualize the fitted copula vs. empirical data
def plot_copula_comparison(data_uniform, asset_1, asset_2, best_copula_name, period_name):
    copulas = {
        "Gaussian": GaussianCopula(),
        "Clayton": Clayton(),
        "Gumbel": Gumbel(),
        "Frank": Frank()
    }

    pair_data = data_uniform[[asset_1, asset_2]].dropna().to_numpy()

    # Fit the best copula
    best_copula = copulas[best_copula_name]
    best_copula.fit(pair_data)

    # Generate synthetic data from the fitted copula
    if best_copula_name == "Gaussian":
        copula_samples = best_copula.sample(size=1000)
    else:
        copula_samples = best_copula.sample(1000)

    # Scatter plot of empirical copula
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=pair_data[:, 0], y=pair_data[:, 1], alpha=0.5)
    plt.title(f"Empirical Copula: {asset_1} & {asset_2} ({period_name})")
    plt.xlabel(asset_1)
    plt.ylabel(asset_2)

    # Contour plot of the fitted copula
    plt.subplot(1, 2, 2)
    sns.kdeplot(x=copula_samples[:, 0], y=copula_samples[:, 1], cmap="coolwarm", fill=True)
    plt.title(f"Fitted {best_copula_name} Copula: {asset_1} & {asset_2} ({period_name})")
    plt.xlabel(asset_1)
    plt.ylabel(asset_2)

    plt.show()

# Example visualizations for selected pairs
plot_copula_comparison(gfc_uniform, "brent", "usd_eur", "Gumbel", "GFC (2008)")
plot_copula_comparison(gfc_uniform, "usd_eur", "us_10", "Clayton", "GFC (2008)")
plot_copula_comparison(covid_uniform, "sp500", "euro50", "Frank", "COVID-19 (2020)")
plot_copula_comparison(covid_uniform, "us_10", "usd_eur", "Gumbel", "COVID-19 (2020)")


#%%#### Robustness Checks for Copula Fits

# Custom function to compute AIC and BIC
def compute_aic_bic(log_likelihood, num_params, sample_size):
    """Computes AIC and BIC for a fitted copula."""
    aic = 2 * num_params - 2 * log_likelihood
    bic = num_params * np.log(sample_size) - 2 * log_likelihood
    return aic, bic

# Function to evaluate robustness of Copula Fit
def evaluate_copula_fit(data_uniform, asset_1, asset_2, best_copula_name, period_name):
    copulas = {
        "Gaussian": GaussianCopula(),
        "Clayton": Clayton(),
        "Gumbel": Gumbel(),
        "Frank": Frank()
    }

    pair_data = data_uniform[[asset_1, asset_2]].dropna().to_numpy()
    sample_size = len(pair_data)

    # Fit the best copula
    best_copula = copulas[best_copula_name]
    best_copula.fit(pair_data)

    # Compute log-likelihood
    log_likelihood = np.sum(np.log(best_copula.probability_density(pair_data)))

    # Compute AIC & BIC
    num_params = 1  # Archimedean Copulas have 1 parameter (theta), Gaussian has correlation
    aic, bic = compute_aic_bic(log_likelihood, num_params, sample_size)

    # Perform KS Test on fitted vs. empirical
    if best_copula_name == "Gaussian":
        synthetic_samples = best_copula.sample(size=sample_size)
    else:
        synthetic_samples = best_copula.sample(sample_size)

    ks_stat, ks_p_value = ks_2samp(pair_data.flatten(), synthetic_samples.flatten())

    # Create DataFrame for output
    results_df = pd.DataFrame({
        "Metric": ["Log-Likelihood", "AIC", "BIC", "KS-Statistic", "KS p-value"],
        "Value": [log_likelihood, aic, bic, ks_stat, ks_p_value]
    })

    # Display results in console
    print(f"\n**Copula Robustness for {asset_1} & {asset_2} ({period_name})**")
    print(results_df.to_string(index=False))  # Print formatted DataFrame

# Run Robustness Checks for some key asset pairs
evaluate_copula_fit(gfc_uniform, "brent", "usd_eur", "Gumbel", "GFC (2008)")
evaluate_copula_fit(gfc_uniform, "usd_eur", "us_10", "Clayton", "GFC (2008)")
evaluate_copula_fit(covid_uniform, "sp500", "euro50", "Frank", "COVID-19 (2020)")
evaluate_copula_fit(covid_uniform, "us_10", "usd_eur", "Gumbel", "COVID-19 (2020)")


















