#include "base/linear_regression.h"
#include <stdexcept>

namespace base {

void LinearRegression1D::fit(const std::vector<double>& x, const std::vector<double>& y) {
	if (x.size() != y.size() || x.size() < 2){
		throw std::invalid_argument("fit: invalid input sizes (need >= 2 and same size");
	}

	const std::size_t n = x.size();
	double sum_x = 0.0, sum_y = 0.0;
	
	for(std::size_t i = 0; i < n; ++i) {
		sum_x += x[i];
		sum_y += y[i];
	}

	const double xbar = sum_x / static_cast<double>(n);
	const double ybar = sum_y / static_cast<double>(n);

	double num = 0.0; // Σ (xi - x̄)(yi - ȳ)
	double den = 0.0; // Σ (xi - x̄)^2

	for (std::size_t i = 0; i < n; ++i) {
		const double dx = x[i] - xbar;
		const double dy = y[i] - ybar;
		num += dx * dy;
		den += dx * dx;	
	}

	if (den == 0.0) {
		throw std::runtime_error("fit: all X are identical, cannot fit a line");
	}

	b_ = num / den;
	a_ = ybar - b_ * xbar;
	fitted_ = true;

}

double LinearRegression1D::predict(double x) const {
	if(!fitted_) throw std::runtime_error("predict: model not fitted");
	return a_ + b_ * x;
}

std::vector<double> LinearRegression1D::predict(const std::vector<double>& xs) const {
	if(!fitted_) throw std::runtime_error("predict: model not fitted");
	std::vector<double> out;
	out.reserve(xs.size());
	
	for(double v : xs) out.push_back(predict(v));
	return out;
	}

}
