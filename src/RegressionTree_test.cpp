
#include <vector>
#include <iostream>
#include <ctime>
#include <cmath>

#include <boost/random.hpp>

#include "RegressionTree.h"
#include "RegressionTreeBuilder.h"

#define BOOST_TEST_MODULE(RegressionTreeTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Simple)
{
	// Randomly set the data observations
	boost::mt19937 rgen(18902);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::uniform_int<unsigned int> rdestd(0, 1);	// {0, 1}
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);
	boost::mt19937 rgen2(18903);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > randd(rgen2, rdestd);

	// Generate data set, 2D observations
	unsigned int sample_count = 750;
	std::vector<Grante::RegressionTreeBuilder::data_pair_t> X;
	std::vector<std::vector<double> > XC(sample_count);
	std::vector<double> x_cur_c(1);
	std::vector<std::vector<unsigned int> > XD(sample_count);
	std::vector<unsigned int> x_cur_d(1);
	std::vector<double> Y;
	for (unsigned int n = 0; n < sample_count; ++n) {
		x_cur_c[0] = 16.0*(randu()-0.5);
		x_cur_d[0] = randd();
		XC[n] = x_cur_c;
		XD[n] = x_cur_d;
		X.push_back(Grante::RegressionTreeBuilder::data_pair_t(
			&XC[n], &XD[n]));
		Y.push_back(std::cos(0.3*x_cur_c[0] - 0.5*x_cur_d[0]));
	}

	Grante::RegressionTreeBuilder builder(/*max_depth*/ 8, /*min_sample*/ 10);
	Grante::RegressionTree* tree = builder.Build(X, Y);

	double l2_err = 0.0;
	for (unsigned int n = 0; n < Y.size(); ++n) {
		double y_pred = tree->Evaluate(*X[n].first, *X[n].second);
#if 0
		std::cout << "n " << n << ", true " << Y[n] << ", pred "
			<< y_pred << std::endl;
#endif
		double cur_err = (y_pred - Y[n])*(y_pred - Y[n]);
		l2_err += cur_err;
	}
	l2_err = std::sqrt(l2_err) / static_cast<double>(sample_count);
	std::cout << "Mean L2 prediction error: " << l2_err << std::endl;
	BOOST_CHECK_LT(l2_err, 0.01);

	std::vector<unsigned int> leaf_to_node;
	tree->ComputeLeafToNodeIndex(leaf_to_node);
	std::cout << leaf_to_node.size() << " leaf nodes" << std::endl;
}

