#include <iostream>
#include <vector>

#include <base/linear_regression.h>
#include <base/metrics.h>

int main() {

	// Example base inputs ( X -> Y )
	std::vector<double> X = {1, 2, 2, 3, 4, 5, 5};
    	std::vector<double> Y = {120, 135, 137, 150, 164, 182, 186};

	try {
		base::LinearRegression1D lr;
		lr.fit(X,Y);

		std::cout << "Model: Y = " << lr.a() << " + " << lr.b() << "X\n";

		auto Y_hat = lr.predict(X);

		double mse = base::metrics::mse(Y, Y_hat);
		double r2 = base::metrics::r2_score(Y, Y_hat);

		std::cout << "MSE = " << mse << "\n";
        	std::cout << "R^2 = " << r2  << "\n";

		double x_new = 6.0;
		std::cout << "Predictions for X=" << x_new << " -> " << lr.predict(x_new) << "\n";
		}

		catch(const std::exception& ex) {
			std::cerr << "Error: " << ex.what() << "\n";
        		return 1;
		}

	return 0;
}
