
#include <vector>
#include <iostream>
#include <cmath>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TestModels.h"
#include "SAMCInference.h"

#define BOOST_TEST_MODULE(SAMCTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(Ising2D)
{
	// Get a 2D Ising model
	Grante::FactorGraphModel* model;
	Grante::FactorGraph* fg;
	Grante::TestModels::Ising2D(16, 0.8, &model, &fg);

	// Perform stochastic approximation Monte Carlo
	Grante::SAMCInference samc(fg);
	samc.SetSamplingParameters(15, 5.0, 0.5, 100, 2500);
	samc.PerformInference();

	const std::vector<unsigned int>& hist = samc.TemperatureHistogram();
	const std::vector<double>& theta = samc.LogPartitionEstimates();
	for (unsigned int li = 0; li < hist.size(); ++li) {
		std::cout << "level " << li << " visits: " << hist[li]
			<< ", theta " << theta[li] << std::endl;
	}
	std::cout << std::endl;

	const std::vector<double>& marg = samc.Marginal(0);
	for (unsigned int mi = 0; mi < marg.size(); ++mi)
		std::cout << "marg " << mi << ", prob: " << marg[mi] << std::endl;

	BOOST_CHECK_CLOSE_ABS(0.5, marg[0], 0.2);
	BOOST_CHECK_CLOSE_ABS(0.5, marg[3], 0.2);
	BOOST_CHECK(marg[1] <= 0.2);
	BOOST_CHECK(marg[2] <= 0.2);

	delete fg;
	delete model;
}

BOOST_AUTO_TEST_CASE(IsingSamples)
{
	// Get a 2D Ising model
	Grante::FactorGraphModel* model;
	Grante::FactorGraph* fg;
	unsigned int N = 24;
	Grante::TestModels::Ising2D(N, 1.8, &model, &fg);

	// Perform stochastic approximation Monte Carlo
	Grante::SAMCInference samc(fg);
	samc.SetSamplingParameters(15, 10.0, 0.5, 2500, 2500);

	std::vector<std::vector<unsigned int> > states;
	samc.Sample(states, 100);

	for (unsigned int si = 0; si < 2; ++si) {
		std::cout << "Sample " << (si+1) << std::endl;

		unsigned int si1 = (si == 0 ? 0 : states.size()-1);
		for (unsigned int row = 0; row < N; ++row) {
			for (unsigned int col = 0; col < N; ++col) {
				std::cout << (states[si1][row*N+col] == 0 ? "." : "#");
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
	delete fg;
	delete model;
}

