
import numpy as np
import pandas as pd
from scipy.stats import t


# Cleans a CSV file by skipping the first 3 lines, removing an unwanted column, and dropping rows with missing values.
def preprocessingClearingOriginalFile(inputFile, outputFile):   
    # Load the CSV file into a DataFrame, skipping the first 3 lines
    input_file = inputFile  # The CSV file name
    output_file = outputFile  # The file to save the cleaned data

    # Read the CSV file, skipping the first 3 lines and specifying a custom delimiter if needed
    df = pd.read_csv(input_file, sep=',', skiprows=3, on_bad_lines='warn', engine='python')

    df = df.drop(df.columns[-1], axis=1) # No need the 'Unnamed: 68' column

    # Attempt to create the columns to check based on the actual DataFrame columns
    cols_to_check = list(map(str, range(1, 82)))  # Checking for columns "1" to "81"

    # Verify which of the expected columns actually exist in the DataFrame
    existing_cols_to_check = [col for col in df.columns]

    # If there are no valid columns to check
    if not existing_cols_to_check:
        print("No valid columns to check for missing values. Please verify the column names in the CSV.")
    else:
        # Drop rows with missing values in the specified existing columns
        df_cleaned = df.dropna(subset=existing_cols_to_check)

        # Save the cleaned data to a new CSV file
        df_cleaned.to_csv(output_file, index=False)
        df_cleaned = pd.read_csv(output_file, sep=',', on_bad_lines='warn', engine='python')

        print(f"Cleaned data saved to {output_file}")

    return df_cleaned   # Updated Data Frame


# Generates a time series from GDP data for multiple countries and performs the Dickey-Fuller test to estimate persistence (rho).
def generate_and_test_series(df, model, alpha=0.05):
    rhos = []
    # Initialize the time series
    T = 64 # Number of years from 1960 to 2023 (inclusive)
    y = np.zeros((len(df), T))  # Shape: (number of countries, number of years)

    # Extracting the series of numbers for each country
    for index, row in df.iterrows():
        # Extract the GDP values for years 1960 to 2023
        gdp_values = row[4:].values  # This will get all the GDP values
        y[index] = gdp_values  # Store them in the time series array

    for i in range(len(df)):
        ts = y[i]
        rho = df_test(ts, trend=model)
        rhos.append(rho)
    return rhos, np.mean(rhos)


# Performs the Dickey-Fuller test on a time series `y` to estimate the persistence parameter (rho) based on the specified trend.
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
    phi = beta[-1]
    rho = 1 + phi
    return rho


# Computes a differentially private mean of `data` by adding Laplace noise for privacy protection.
def dp_mean(data, epsilon, lower_bound, upper_bound):
    sensitivity = (upper_bound - lower_bound) / len(data)
    noisy_sum = sum(data)/len(data) + np.random.laplace(0, sensitivity / epsilon)
    return np.clip(noisy_sum , lower_bound, upper_bound)





def main():
    inputFile = 'OriginalGDPData.csv'
    df = None

    ######################
    ### Pre-Processing ###
    ######################
    outputFile = 'CleanedGDPData.csv'
    df = preprocessingClearingOriginalFile(inputFile, outputFile)
    #print('Countries in the experiment:', countriesNames)

    #########################################################################################
    ### Generate Multiple Time series, and Perform Regular and DP Dickey-Fuller Algorithm ###
    #########################################################################################
    epsilon = 0.40
    model = 'ct'
    rhos, mean_rho = generate_and_test_series(df, model)

    # mean_rho
    dp_mean_rho = dp_mean(np.array(rhos), epsilon, 0, 1.3)

    print(f"\n\nRegular Dickey-Fuller Test Metrics for Model {model}:\nRho Estimator = {mean_rho}")
    print(f"\n\nDifferentially Private Dickey-Fuller Test Metrics for Model {model}:\nRho Estimator = {dp_mean_rho}")


if __name__ == '__main__':
    main()