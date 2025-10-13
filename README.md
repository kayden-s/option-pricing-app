# Derivio

Derivio is a web platform for pricing financial options using three quantitative methods: 

1. Black-Scholes model
2. Monte Carlo simulation
3. Binomial model

The app fetches data for 105,000+ stocks and provides an interactive interface for analysis and comparison.

---

## Overview
Derivio allows users to compute theoretical option prices for both call and put options under various assumptions and market conditions.

---

## Implemented Models

### 1. Black-Scholes Model
A closed-form solution for European options, based on stock price, strike price, time to maturity, risk-free rate, and volatility.

### 2. Monte Carlo Simulation
A stochastic simulation technique that generates thousands of possible price paths to estimate an option’s expected payoff.

### 3. Binomial Model
A discrete-time framework that models the underlying price as a binomial tree, calculating option values step by step until expiration.

---

## Features
- Fetches live market data for over 105,000 stocks via the Yahoo Finance API (pandas-datareader)
- Caches API calls using requests-cache for faster performance  
- Customizable user inputs:
  - Strike price  
  - Risk-free rate (%)  
  - Volatility (σ, %)  
  - Expiration date  
- Computes prices across all three models  
- Visualizes outputs in real time through an interactive dashboard  
