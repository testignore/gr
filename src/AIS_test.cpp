
#include <vector>
#include <limits>
#include <iostream>
#include <ctime>
#include <cmath>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "AISInference.h"

#define BOOST_TEST_MODULE(AISTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(MiniAIS)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4);
	w[0] = 0.8;
	w[1] = 0.5;
	w[2] = 2.0;
	w[3] = 0.6;
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt2 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::AISInference aisinf(&fg);
	aisinf.SetSamplingParameters(50, 1, 1000);
	aisinf.PerformInference();

	// Ground truth from prototype/mfield.m
	double log_z = aisinf.LogPartitionFunction();
	std::cout << "AIS log_z " << log_z << ", exact 0.55390" << std::endl;
	BOOST_CHECK_CLOSE_ABS(0.55390, log_z, 0.01);
}

