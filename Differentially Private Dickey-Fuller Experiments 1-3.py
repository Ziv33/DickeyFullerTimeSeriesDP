import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import os

"""
Generates a synthetic time series of specified length with optional constant, trend, and autoregressive (AR) components.

Parameters:
----------
T : int
    Length of the time series.
phi : float
    Autoregressive (AR) coefficient, controlling the series' dependence on previous values.
constant : float, optional
    Constant term added at each time step, if specified.
trend_value : float, optional
    Linear trend slope to apply over the time series; default is 0 (no trend).
y0rand : bool, optional
    If True, initializes the first value in the series (`y[0]`) randomly between 0 and 1;
    otherwise, starts with `y[0]` set to zero. Default is False.

Returns:
-------
np.ndarray
    Array containing the generated time series values.

"""

def generate_time_series(T, phi, constant=None, trend_value=0, y0rand=False):
    y = np.zeros(T)
    trend = np.linspace(0, trend_value * (T-1), T) if trend_value != 0 else np.zeros(T)
    if y0rand:
        y[0] = np.random.random()
    else:
        y[0] = 0
    for t in range(1, T):
        if constant is not None:
            y[t] = constant + trend[t] + phi * y[t-1] + np.random.normal()
        else:
            y[t] = trend[t] + phi * y[t-1] + np.random.normal()
    return y

"""
Performs the Dickey-Fuller test on a time series `y`, with options for constant and linear trend components.

Parameters:
----------
y : np.ndarray
    Input time series to test for stationarity.
trend : str, optional
    Specifies the trend component to include in the test:
    - 'n' : No trend
    - 'c' : Constant (intercept) only
    - 'ct': Constant + linear trend

Returns:
-------
t_stat : float
    Test statistic for assessing stationarity of the time series.
p_value : float
    p-value associated with the test statistic.
constant : float
    Estimated constant term if specified in `trend`; otherwise, 0.
trend_coefficient : float
    Estimated trend coefficient if linear trend is included; otherwise, 0.
phi : float
    Coefficient for lagged values, indicating the presence of a unit root.
rho : float
    Value computed as 1 + phi, indicating the persistence level of the series.

"""

def df_test(y, trend='n'):
    T = len(y)
    y_diff = np.diff(y)
    y_lag = y[:-1]
    if trend == 'c':
        X = np.column_stack([np.ones(T-1), y_lag])
    elif trend == 'ct':
        X = np.column_stack([np.ones(T-1), np.arange(1, T), y_lag])
    else:
        X = np.column_stack([y_lag])
    y_diff = y_diff[:T-1]
    beta = np.linalg.inv(X.T @ X) @ X.T @ y_diff
    residuals = y_diff - X @ beta
    sigma2 = np.sum(residuals ** 2) / (T - X.shape[1])
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    t_stat = beta[-1] / np.sqrt(var_beta[-1, -1])
    p_value = 2 * t.cdf(-abs(t_stat), df=T-X.shape[1])
    constant = beta[0] if trend != 'n' else 0
    trend_coefficient = beta[1] if trend == 'ct' else 0
    phi = beta[-1]
    rho = 1 + phi
    return t_stat, p_value, constant, trend_coefficient, phi, rho

"""
Generates multiple time series and performs the Dickey-Fuller test on each, assessing stationarity.

Parameters:
----------
n : int
    Number of time series to generate and test.
T : int
    Length of each generated time series.
phi : float
    Autoregressive coefficient, determining the series' dependence on prior values.
constant : float
    Constant term added to each step in the time series.
trend_value : float
    Linear trend slope applied over the time series.
model : str
    Type of trend included in the Dickey-Fuller test:
    - 'n' : No trend
    - 'c' : Constant trend
    - 'ct': Constant + linear trend
alpha : float, optional
    Significance level for the Dickey-Fuller test; default is 0.05.

Returns:
-------
pd.DataFrame
    DataFrame with each test's p-value and the result of the stationarity test (True if accepted).
float
    Mean p-value from all tests.
float
    Mean t-statistic from all tests.
float
    Mean rho (1 + phi) from all tests.
list
    List of p-values for each test.
list
    List of t-statistics for each test.
list
    List of rho values for each test.

"""

def generate_and_test_series(n, T, phi, constant, trend_value, model, alpha=0.05):
    results = []
    p_values = []
    t_values = []
    rhos = []
    for i in range(n):
        ts = generate_time_series(T, phi, constant, trend_value)
        t_stat, p_value, const, trend_coeff, phi_value, rho = df_test(ts, trend=model)
        accepted = p_value > alpha
        results.append({'P-Value': p_value, 'Accepted': accepted})
        p_values.append(p_value)
        t_values.append(t_stat)
        rhos.append(rho)
    return pd.DataFrame(results), np.mean(p_values), np.mean(t_values), np.mean(rhos), p_values, t_values, rhos

"""
Calculates a differentially private mean of `data` using Laplace noise addition for privacy.

Parameters:
----------
data : array-like
    List or array of numerical values to compute the mean on.
epsilon : float
    Privacy parameter, controlling the noise level; lower values increase privacy but add more noise.
lower_bound : float
    Minimum possible value in the data, used to scale noise sensitivity and clip the output.
upper_bound : float
    Maximum possible value in the data, also used to scale noise sensitivity and clip the output.

Returns:
-------
float
    Differentially private mean of `data`, clipped to the range [lower_bound, upper_bound].
    
"""

def dp_mean(data, epsilon, lower_bound, upper_bound):
    sensitivity = (upper_bound - lower_bound) / len(data)
    noisy_sum = sum(data)/len(data) + np.random.laplace(0, sensitivity / epsilon)
    return np.clip(noisy_sum , lower_bound, upper_bound)

# Parameters for Experiments
T = 100                                                                        # Length of each individual's time series
alpha = 0.05                                                                   # Significance level for the Dickey-Fuller test
rho_values = [0.8, 0.82, 0.85, 0.88, 0.9, 0.92, 0.95, 0.99, 1, 1.02, 1.05]     # Coefficient of Y_(t-1) of the autoregressive model
n_values = [20, 25, 30, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]   # Number of individuals in each experiment
epsilon_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.4]               # Grid of the privacy budgets (or, privacy loss)
N_Repeats = 50                                                                 # Number of repetitions of each experiment

# List of directories for storing the experiments' results:
'''
First And Third Experiment - All Three Models With The Correct Autoregressive Model's Dataset.
Second Experiment Part i (i=1,2,3) - The i-th Autoregressive Model's Dataset as Input For Every Model - 
                                     Checking How Each Model Behaves When it Not Receiving Its Suitable Model (Itself as a Model).
'''
experiments_list = ['First And Third Experiment', 'Second Experiment Part 1', 'Second Experiment Part 2', 'Second Experiment Part 3']


def main():
    # Create directories if they don't exist:
    for Experiment in experiments_list:
        os.makedirs(Experiment, exist_ok=True)


        # Initialize MSE dictionaries
        mse_rejection = {model: {epsilon: [] for epsilon in epsilon_values} for model in ['n', 'c', 'ct']}
        mse_t_stat = {model: {epsilon: [] for epsilon in epsilon_values} for model in ['n', 'c', 'ct']}
        mse_rho = {model: {epsilon: [] for epsilon in epsilon_values} for model in ['n', 'c', 'ct']}

        # Plot results
        for n in n_values:
            for epsilon in epsilon_values:
                fig, axs = plt.subplots(3, 3, figsize=(18, 12)) # 3x3 grid for three models
                fig.suptitle(f'Results for n={n}, Epsilon={epsilon}', fontsize=16)

                for model_idx, model in enumerate(['n', 'c', 'ct']):
                    all_rejection_rates = []
                    all_dp_rejection_rates = []
                    all_t_stats = []
                    all_dp_t_stats = []
                    all_rhos = []
                    all_dp_rhos = []

                    for rho in rho_values:
                        rejection_rates = []
                        dp_rejection_rates = []
                        t_stats = []
                        dp_t_stats = []
                        calculated_rhos = []
                        dp_calculated_rhos = []

                        for repeat in range(N_Repeats):
                            if model == 'c':
                                const = 0.5
                                trend = 0
                            elif model == 'ct':
                                const = 0.5
                                trend = 0.2
                            else:
                                const = None
                                trend = 0

                            if Experiment == 'First And Third Experiment':
                                results_series, mean_p, mean_t, mean_rho, p_values, t_values, rhos = generate_and_test_series(n, T, rho, constant=const, trend_value=trend, model=model, alpha=alpha)
                            elif Experiment == 'Second Experiment Part 1':
                                results_series, mean_p, mean_t, mean_rho, p_values, t_values, rhos = generate_and_test_series(n, T, rho, constant=None, trend_value=0, model=model, alpha=alpha)
                            elif Experiment == 'Second Experiment Part 2':
                                results_series, mean_p, mean_t, mean_rho, p_values, t_values, rhos = generate_and_test_series(n, T, rho, constant=0.5, trend_value=0, model=model, alpha=alpha)
                            elif Experiment == 'Second Experiment Part 3':
                                results_series, mean_p, mean_t, mean_rho, p_values, t_values, rhos = generate_and_test_series(n, T, rho, constant=0.5, trend_value=0.2, model=model, alpha=alpha)


                            rejection_rate = 1 - results_series['Accepted'].mean()
                            dp_rejection_rate = 1 - dp_mean(results_series['Accepted'].values, epsilon, 0, 1)

                            rejection_rates.append(rejection_rate)
                            dp_rejection_rates.append(dp_rejection_rate)
                            t_stats.append(mean_t)
                            dp_t_stats.append(dp_mean(np.array(t_values), epsilon, -20, 20))
                            calculated_rhos.append(mean_rho)
                            dp_calculated_rhos.append(dp_mean(np.array(rhos), epsilon, 0, 1.3))

                        all_rejection_rates.append(np.mean(rejection_rates))
                        all_dp_rejection_rates.append(np.mean(dp_rejection_rates))
                        all_t_stats.append(np.mean(t_stats))
                        all_dp_t_stats.append(np.mean(dp_t_stats))
                        all_rhos.append(np.mean(calculated_rhos))
                        all_dp_rhos.append(np.mean(dp_calculated_rhos))

                    # Calculate MSE for this n
                    mse_rejection[model][epsilon].append(np.mean((np.array(all_rejection_rates) - np.array(all_dp_rejection_rates))**2))
                    mask = np.array(rho_values) <= 1
                    mse_t_stat[model][epsilon].append(np.mean((np.array(all_t_stats)[mask] - np.array(all_dp_t_stats)[mask])**2))
                    mse_rho[model][epsilon].append(np.mean((np.array(all_rhos) - np.array(all_dp_rhos))**2))

                    # Plot rejection rates
                    std_rej = np.std(all_rejection_rates)
                    std_dp_rej = np.std(all_dp_rejection_rates)
                    axs[model_idx, 0].plot(rho_values, all_rejection_rates, marker='o', linestyle='--', color='blue', label='Rejection Rate')
                    axs[model_idx, 0].plot(rho_values, all_dp_rejection_rates, marker='o', linestyle='-', color='red', label='DP Rejection Rate')
                    axs[model_idx, 0].fill_between(rho_values, np.array(all_rejection_rates) - std_rej, 
                                                np.array(all_rejection_rates) + std_rej, alpha=0.2, color='blue')
                    axs[model_idx, 0].fill_between(rho_values, np.array(all_dp_rejection_rates) - std_dp_rej, 
                                                np.array(all_dp_rejection_rates) + std_dp_rej, alpha=0.2, color='red')
                    axs[model_idx, 0].set_title(f'Rejection Rates (Model {model})')
                    axs[model_idx, 0].set_xlabel('Rho')
                    axs[model_idx, 0].set_ylabel('Rejection Rate')
                    axs[model_idx, 0].legend()
                    axs[model_idx, 0].grid(True)
                    axs[model_idx, 0].annotate(f'STD: {std_rej:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color='blue')
                    axs[model_idx, 0].annotate(f'STD: {std_dp_rej:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=8, color='red')

                    # Plot t-stats
                    std_t = np.std(np.array(all_t_stats)[mask])
                    std_dpt = np.std(np.array(all_dp_t_stats)[mask])
                    axs[model_idx, 1].plot(np.array(rho_values)[mask], np.array(all_t_stats)[mask], marker='o', linestyle='--', color='green', label='T-Stat')
                    axs[model_idx, 1].plot(np.array(rho_values)[mask], np.array(all_dp_t_stats)[mask], marker='o', linestyle='-', color='orange', label='DP T-Stat')
                    axs[model_idx, 1].fill_between(np.array(rho_values)[mask], np.array(all_t_stats)[mask] - std_t, 
                                                np.array(all_t_stats)[mask] + std_t, alpha=0.2, color='green')
                    axs[model_idx, 1].fill_between(np.array(rho_values)[mask], np.array(all_dp_t_stats)[mask] - std_dpt, 
                                                np.array(all_dp_t_stats)[mask] + std_dpt, alpha=0.2, color='orange')
                    axs[model_idx, 1].set_title(f'T-Stats (Model {model})')
                    axs[model_idx, 1].set_xlabel('Rho')
                    axs[model_idx, 1].set_ylabel('T-Stat')
                    axs[model_idx, 1].legend()
                    axs[model_idx, 1].grid(True)
                    axs[model_idx, 1].annotate(f'STD: {std_t:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color='green')
                    axs[model_idx, 1].annotate(f'STD: {std_dpt:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=8, color='orange')

                    # Plot rhos
                    std_rho = np.std(all_rhos)
                    std_dp_rho = np.std(all_dp_rhos)
                    axs[model_idx, 2].plot(rho_values, all_rhos, marker='o', linestyle='--', color='purple', label='Rho')
                    axs[model_idx, 2].plot(rho_values, all_dp_rhos, marker='o', linestyle='-', color='brown', label='DP Rho')
                    axs[model_idx, 2].fill_between(rho_values, np.array(all_rhos) - std_rho, 
                                                np.array(all_rhos) + std_rho, alpha=0.2, color='purple')
                    axs[model_idx, 2].fill_between(rho_values, np.array(all_dp_rhos) - std_dp_rho, 
                                                np.array(all_dp_rhos) + std_dp_rho, alpha=0.2, color='brown')
                    axs[model_idx, 2].set_title(f'Rhos (Model {model})')
                    axs[model_idx, 2].set_xlabel('Rho')
                    axs[model_idx, 2].set_ylabel('Calculated Rho')
                    axs[model_idx, 2].legend()
                    axs[model_idx, 2].grid(True)
                    axs[model_idx, 2].annotate(f'STD: {std_rho:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, color='purple')
                    axs[model_idx, 2].annotate(f'STD: {std_dp_rho:.2f}', xy=(0.05, 0.90), xycoords='axes fraction', fontsize=8, color='brown')

                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                plt.savefig(f'{Experiment}/results_n{n}_epsilon{epsilon}.png')
                plt.close()



        # Plot MSE as a function of n_values
        for model in ['n', 'c', 'ct']:
            fig, axs = plt.subplots(3, 1, figsize=(12, 18))
            fig.suptitle(f'MSE as a function of n (Model = {model})', fontsize=16)

            for ax_idx, (ax, mse_data, title) in enumerate(zip(axs, 
                                                            [mse_rejection[model], 
                                                                mse_t_stat[model], 
                                                                mse_rho[model]],
                                                            ['MSE of Rejection Rates', 
                                                                'MSE of T-Stats', 
                                                                'MSE of Calculated Rhos'])):
                for epsilon in epsilon_values:
                    line, = ax.plot(n_values, mse_data[epsilon], marker='o', linestyle='-', label=f'Epsilon {epsilon}')
                    color = line.get_color()
                    
                    for i, (n, mse) in enumerate(zip(n_values, mse_data[epsilon])):
                        ax.annotate(f'{mse:.2e}', 
                                    (n, mse), 
                                    xytext=(5, 5), 
                                    textcoords='offset points', 
                                    fontsize=8,
                                    color=color)
                
                ax.set_title(title)
                ax.set_xlabel('n')
                ax.set_ylabel('MSE')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{Experiment}/mse_n_model_{model}.png', dpi=300)
            plt.close()

            # Save MSE tables as images with improved resolution
            mse_table = pd.DataFrame(mse_data, index=n_values)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis('tight')
            ax.axis('off')
            mse_table_rounded = mse_table.round(7)
            table = ax.table(cellText=(mse_table_rounded.values), colLabels=mse_table.columns, rowLabels=mse_table.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 1.5)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2) # Adjust layout to prevent cutting
            plt.savefig(f'{Experiment}/mse_table_model_{model}.png', dpi=300)
            plt.close()



        # Plot abs(log(MSE)) as a function of n_values
        for model in ['n', 'c', 'ct']:
            fig, axs = plt.subplots(3, 1, figsize=(12, 18))
            fig.suptitle(f'abs(log(MSE)) as a function of n (Model = {model})', fontsize=16)

            for ax_idx, (ax, mse_data, title) in enumerate(zip(axs, 
                                                            [mse_rejection[model], 
                                                                mse_t_stat[model], 
                                                                mse_rho[model]],
                                                            ['MSE of Rejection Rates', 
                                                                'MSE of T-Stats', 
                                                                'MSE of Calculated Rhos'])):
                for epsilon in epsilon_values:
                    line, = ax.plot(n_values, np.abs(np.log(mse_data[epsilon])), marker='o', linestyle='-', label=f'Epsilon {epsilon}')
                    color = line.get_color()
                    
                    for i, (n, mse) in enumerate(zip(n_values, np.abs(np.log(mse_data[epsilon])))):
                        ax.annotate(f'{mse:.2e}', 
                                    (n, mse), 
                                    xytext=(5, 5), 
                                    textcoords='offset points', 
                                    fontsize=8,
                                    color=color)
                
                ax.set_title(title)
                ax.set_xlabel('n')
                ax.set_ylabel('abs(log(MSE))')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{Experiment}/abs(log(MSE))_n_model_{model}.png', dpi=300)
            plt.close()

            # Save abs(log(MSE)) tables as images with improved resolution
            mse_table = pd.DataFrame(mse_data, index=n_values)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.axis('tight')
            ax.axis('off')
            abs_log_mse_table_rounded = np.abs(np.log(mse_table)).round(7)
            table = ax.table(cellText=(abs_log_mse_table_rounded.values), colLabels=mse_table.columns, rowLabels=mse_table.index, cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.5, 1.5)
            plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2) # Adjust layout to prevent cutting
            plt.savefig(f'{Experiment}/abs(log(MSE))_table_model_{model}.png', dpi=300)
            plt.close()

        # Plot log scale MSE as a function of n_values
        for model in ['n', 'c', 'ct']:
            fig, axs = plt.subplots(3, 1, figsize=(12, 18))
            fig.suptitle(f'log scale MSE as a function of n (Model = {model})', fontsize=16)

            for ax_idx, (ax, mse_data, title) in enumerate(zip(axs, 
                                                            [mse_rejection[model], 
                                                                mse_t_stat[model], 
                                                                mse_rho[model]],
                                                            ['MSE of Rejection Rates', 
                                                                'MSE of T-Stats', 
                                                                'MSE of Calculated Rhos'])):
                for epsilon in epsilon_values:
                    line, = ax.semilogy(n_values, mse_data[epsilon], marker='o', linestyle='-', label=f'Epsilon {epsilon}')
                    color = line.get_color()
                    
                    for i, (n, mse) in enumerate(zip(n_values, mse_data[epsilon])):
                        ax.annotate(f'{mse:.2e}', 
                                    (n, mse), 
                                    xytext=(5, 5), 
                                    textcoords='offset points', 
                                    fontsize=8,
                                    color=color)
                
                ax.set_title(title)
                ax.set_xlabel('n')
                ax.set_ylabel('log scale MSE')
                ax.legend()
                ax.grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{Experiment}/log_scale_MSE_n_model_{model}.png', dpi=300)
            plt.close()
        print(f"All plots and tables have been generated and saved in the '{Experiment}' directory.")

    print('All experiment was finished successfully.')



if __name__ == '__main__':
    main()