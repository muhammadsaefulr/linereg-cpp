#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>

namespace base::metrics {

// Mean Squared Error Base Function
inline double mse(const std::vector<double>& y_true, const std::vector<double> y_pred) {

	if(y_true.size() != y_pred.size())
		throw std::invalid_argument("mse: size missmatch");
	
	const std::size_t n = y_true.size();
	double s = 0.0;
	
	for(std::size_t i = 0; i < n; ++i){
		double d = y_true[i] - y_pred[i];
		s += d * d;
	}

	return s / static_cast<double>(n);
}

// Mean Absolute Error Base Function
inline double mae(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
	if (y_true.size() != y_pred.size()) 
		throw std::invalid_argument("mse: size missmatch");
	
	const std::size_t n = y_true.size();
	double s = 0.0;
	
	for(std::size_t i = 0; i < n; ++i){
		s += std::fabs(y_true[i] - y_pred[i]);
	}

	return s / static_cast<double>(n);
}

// R Squared Linear
inline double r2_score(const std::vector<double>& y_true, const std::vector<double>& y_pred) {
	if (y_true.size() != y_pred.size())
		throw std::invalid_argument("r2_score: size missmatch");
	
	const std::size_t n = y_true.size();
	double mean = 0.0;
	
	for(double v : y_true) mean += v;
	mean /= static_cast<double>(n);

	double ss_res = 0.0;
	double ss_tot = 0.0;

	for(std::size_t i = 0; i < n; ++i) {
		double diff_res = y_true[i] - y_pred[i];
		double diff_tot = y_true[i] - mean;

		ss_res += diff_res * diff_res;
		ss_tot += diff_tot * diff_tot;
	}

	if (ss_tot == 0.0) return 1.0;
	return 1.0 - ss_res / ss_tot;
	}

}
	

