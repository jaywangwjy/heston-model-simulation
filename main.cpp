
#include <iostream>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>
#include <boost/math/tools/minima.hpp>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;

double N(double x) {
	return 0.5 * (1 + erf(x / sqrt(2)));
}

double black_scholes_price(double S, double K, double T, double r, double sigma, const string& option_type) {
	double d1 = (log(S/K) + (r + sigma*sigma/2)*T) / (sigma * sqrt(T));
	double d2 = d1 - sigma * sqrt(T);
	
	if (option_type == "call") {
		return S * N(d1) - K * exp(-r * T) * N(d2);
	} else if (option_type == "put") {
		return K * exp(-r * T) * N(-d2) - S * N(-d1);
	} else {
		throw invalid_argument("Invalid option type");
	}
}

double implied_volatility(double market_price, double S, double K, double T, double r, const string& option_type) {
	auto objective_function = [&](double sigma) {
		return abs(black_scholes_price(S, K, T, r, sigma, option_type) - market_price);
	};

	double lower_bound = 0.01;
	double upper_bound = 6.0;

	std::uintmax_t max_iter = 100;
	double result = boost::math::tools::brent_find_minima(objective_function, lower_bound, upper_bound, max_iter).first;

	return result;
}

vector<vector<double>> generate_heston_paths(double S, double T, double r, double kappa, double theta, double v_0, double rho, double xi, int steps, int Npaths, bool return_vol = false) {
	double dt = T / steps;
	vector<vector<double>> prices(Npaths, vector<double>(steps));
	vector<vector<double>> sigs(Npaths, vector<double>(steps));
	double S_t = S;
	double v_t = v_0;
	//risk modelling intern
	//developed and calibrated statistical model to forecast credit parameters on consumer auto loan portfolios to support strategic decision and regulatory stress testing (ICAAP and CCAR)
	//conducted scenario analysis to simulate the evolution of portfolio credit quality in different macroeconomic and market environment (interest rate hikes etc)
	
	//stock price sentiment analysis
	//calibrated LSTM and GRU model to predict stock movements using both quantitative stock data and qualitative news and tweets data
	//conducted sentiment analysis on news and tweets using (find a tool that does this) to quantify the psychological impact on stock prices
	//achieved 65% accuracy, outperforming vanilla models without incorporating the qualitative data
	
	//hull white model calibration
	//derived analytical closed form solution of interest short rate formula by solving SDE of HW1F model
	//worked on zero coupon bond with call and put options to evaluate the price of cap/floors on stocks
	//calibrated the model and analysed the sensitivity (duration/convexity)
	
	default_random_engine generator;
	normal_distribution<double> normal_dist(0, 1);
	
	for (int t = 0; t < steps; ++t) {
		vector<double> WT(2);
		WT[0] = normal_dist(generator);
		WT[1] = normal_dist(generator);
		
		S_t *= exp((r - 0.5 * v_t) * dt + sqrt(v_t) * WT[0]);
		v_t = abs(v_t + kappa * (theta - v_t) * dt + xi * sqrt(v_t) * WT[1]);
		
		for (int i = 0; i < Npaths; ++i) {
			prices[i][t] = S_t;
			sigs[i][t] = v_t;
		}
	}
	
	if (return_vol) {
		return sigs;
	}
	
	return prices;
}

int main() {
	// Heston model parameters for analysis
	double kappa = 3, theta = 0.04, v_0 = 0.04, xi = 0.6, r = 0.05;
	double S = 100;
	int paths = 3, steps = 10000;
	double T = 1;
	double rho = -0.8;
	
	// Generate Heston paths
	vector<vector<double>> paths_data = generate_heston_paths(S, T, r, kappa, theta, v_0, rho, xi, steps, paths);
	
	// Create a vector of time steps for x-axis
		vector<double> time_steps(steps);
		iota(time_steps.begin(), time_steps.end(), 0);

		// Create a plot for each path
		for (int i = 0; i < paths; ++i) {
			plt::plot(time_steps, paths_data[i]);
		}

		plt::title("Heston Price Paths Simulation");
		plt::xlabel("Time Steps");
		plt::ylabel("Stock Price");
		plt::show();
	
	// Print Heston price paths simulation
	cout << "Heston Price Paths Simulation:" << endl;
	for (int i = 0; i < paths; ++i) {
		cout << "Path " << i + 1 << ": ";
		for (int t = 0; t < steps; ++t) {
			cout << paths_data[i][t] << " ";
		}
		cout << endl;
	}
	
	// Print Heston stochastic volatility simulation
	cout << "Heston Stochastic Vol Simulation:" << endl;
	for (int i = 0; i < paths; ++i) {
		cout << "Path " << i + 1 << ": ";
		for (int t = 0; t < steps; ++t) {
			cout << sqrt(paths_data[i][t]) << " ";
		}
		cout << endl;
	}
	
	
	
	return 0;
}
