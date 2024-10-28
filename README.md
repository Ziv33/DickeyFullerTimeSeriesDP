# Differentially Private Statistical Hypothesis Testing
Existing research has primarily explored differential privacy, in contexts where individuals contribute single data points. Our work extends this scope by examining challenging scenarios in which each individual provides an entire time series, such as financial or economic data. This study addresses the challenge of conducting differentially private statistical hypothesis testing on time series data, with a particular focus on the Dickey-Fuller unit root test. We develop a differentially private mechanism tailored for the Dickey-Fuller test, ensuring that privacy is maintained without compromising the accuracy of statistical inference.

Through a series of experiments, we compare the performance of the differentially private Dickey-Fuller test with its non-private counterpart across various autoregressive models. Our findings demonstrate that the differentially private algorithm can produce results closely aligned with the non-private version, particularly when the privacy budget is optimized and the sample size is sufficiently large. We also find that certain autoregressive models are more robust to the effects of differential privacy. Additionally, we apply our method to a real-world dataset, demonstrating its practical utility in preserving privacy. 

Our research contributes to the growing field of privacy-preserving data analysis, offering a novel approach to hypothesis testing in time series data. The results indicate that differential privacy can be effectively integrated into time series analysis, allowing for the protection of individual privacy in sensitive datasets without significantly sacrificing statistical validity.

## Overview
In this project, we are applying a differentially private method to the Dickey-Fuller test to ensure data privacy while performing this important statistical analysis.

<p align="center">
  <img src="https://github.com/user-attachments/assets/e0397471-85b3-432d-bd38-c5174161f394" alt="Algorithm Overview">
  <br>
  <em>Our Algorithm Overview</em>
</p>

We conducted four experiments, each with its own specific objective. 

The first three experiments, which appears in ```Differentially Private Dickey-Fuller Experiments 1-3.py```, are performed over synthetic data.

The fourth experiments, which appears in ```Differentially Private Dickey-Fuller Experiments 4.py```, is perfomed over a real-world dataset, which is attached to this repository. This experiment aims to evaluate the algorithm's performance in practical applications, ensuring robustness and applicability to real data.

## Authors
* Ziv Chaba
* Hila Cohen
* _Supervisor_: Dr. Or Sheffet

## How To Run The Programs
* Download the files or clone the repository to your local computer.
* Open command line at their location.
* Run the command by ```python3 <Experiment>.py```

Alternatively, you can run this python project on every python IDE.

