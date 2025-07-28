#pragma once
#include <vector>

namespace base {

class LinearRegression1D {

public: 
	
	// Fit Model Y = a + bX
	void fit(const std::vector<double>& x, const std::vector<double>& y);
	
	double predict(double x) const;

	std::vector<double> predict(const std::vector<double>& xs) const;

	// Getter Slope Intercept
	double a() const { return a_; }
	double b() const { return b_; }
	bool fitted() const { return fitted_; }
	
private: 

	double a_ {0.0};
	double b_ {0.0};
	bool fitted_ {false};
	};
}
