
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>

#include <boost/random.hpp>

#include "RBFNetworkRegression.h"

#define BOOST_TEST_MODULE(RBFNetworkRegressionTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Simple)
{
	// Randomly set the data observations
	boost::mt19937 rgen(18902);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Generate data set, 2D observations
	std::vector<std::vector<double> > X;
	std::vector<double> Y;
	std::vector<double> x_cur(2);
	unsigned int sample_count = 750;
	for (unsigned int n = 0; n < sample_count; ++n) {
		x_cur[0] = 16.0*(randu()-0.5);
		x_cur[1] = 16.0*(randu()-0.5);
		X.push_back(x_cur);
		Y.push_back(std::cos(0.3*x_cur[0] - 0.5*x_cur[1]));
	}

	unsigned int rbf_k = 50;
	Grante::RBFNetworkRegression reg(rbf_k, 2);
	reg.FixBeta(0.0);
	double l2err = reg.Fit(X, Y, 1.0e-5, 2000);
	double l2err_mean = l2err / static_cast<double>(sample_count);
	std::cout << "L2 mean error " << l2err_mean << std::endl;
	for (unsigned int n = 0; n < 8; ++n) {
		std::cout << "   sample " << n << ", y_truth " << Y[n]
			<< ", y_pred " << reg.Evaluate(X[n]) << std::endl;
	}
	BOOST_CHECK_LT(l2err_mean, 0.05);
}

BOOST_AUTO_TEST_CASE(SimpleFixedCenters)
{
	// Randomly set the data observations
	boost::mt19937 rgen(18902);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Generate data set, 2D observations
	std::vector<std::vector<double> > X;
	std::vector<double> Y;
	std::vector<double> x_cur(2);
	unsigned int sample_count = 500;
	for (unsigned int n = 0; n < sample_count; ++n) {
		x_cur[0] = 16.0*(randu()-0.5);
		x_cur[1] = 16.0*(randu()-0.5);
		X.push_back(x_cur);
		Y.push_back(std::cos(0.3*x_cur[0] - 0.5*x_cur[1]));
	}

	unsigned int rbf_k = 50;
	std::vector<std::vector<double> > Xproto;
	for (unsigned int n = 0; n < rbf_k; ++n) {
		x_cur[0] = 16.0*(randu()-0.5);
		x_cur[1] = 16.0*(randu()-0.5);
		Xproto.push_back(x_cur);
	}

	Grante::RBFNetworkRegression reg(Xproto);
	reg.FixBeta(-1.0);
	std::cout << "Training" << std::endl;
	double l2err = reg.Fit(X, Y, 1.0e-5, 1000);
	double l2err_mean = l2err / static_cast<double>(sample_count);
	std::cout << "L2 mean error " << l2err_mean << std::endl;
	for (unsigned int n = 0; n < 8; ++n) {
		std::cout << "   sample " << n << ", y_truth " << Y[n]
			<< ", y_pred " << reg.Evaluate(X[n]) << std::endl;
	}
	BOOST_CHECK_LT(l2err_mean, 0.10);
}

