
#include <vector>
#include <iostream>

#include "SubFactorGraph.h"
#include "FactorGraphModel.h"
#include "RandomFactorGraphGenerator.h"

#define BOOST_TEST_MODULE(SubFactorGraphTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(Simple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);
	var_index[0] = 0;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt, var_index, data));

	// Compute the forward map
	fg.ForwardMap();

	// Compute the backward map for some marginal vector
	std::vector<double> marg(4);
	marg[0] = 0.25;
	marg[1] = 0.4;
	marg[2] = 0.1;
	marg[3] = 0.25;
	std::vector<double> pargrad(4, 0.0);
	fac1->BackwardMap(marg, pargrad);
	for (unsigned int pi = 0; pi < pargrad.size(); ++pi) {
		BOOST_CHECK_CLOSE_ABS(marg[pi], pargrad[pi], 1.0e-5);
	}

	// Decompose the factor graph into two disjoint subgraphs
	std::vector<unsigned int> f_set1(2);
	f_set1[0] = 0;
	f_set1[1] = 1;
	std::vector<double> f_scale1(2);
	f_scale1[0] = 1.0;
	f_scale1[1] = 0.5;
	Grante::SubFactorGraph sfg1(&fg, f_set1, f_scale1);

	std::vector<unsigned int> f_set2(2);
	f_set2[0] = 2;
	f_set2[1] = 1;
	std::vector<double> f_scale2(2);
	f_scale2[0] = 1.0;
	f_scale2[1] = 0.5;
	Grante::SubFactorGraph sfg2(&fg, f_set2, f_scale2);

	sfg1.ForwardMap();
	sfg2.ForwardMap();

	std::vector<unsigned int> state(3);
	for (unsigned int s1 = 0; s1 < 2; ++s1) {
		state[0] = s1;
		for (unsigned int s2 = 0; s2 < 2; ++s2) {
			state[1] = s2;
			for (unsigned int s3 = 0; s3 < 2; ++s3) {
				state[2] = s3;
				double orig_E = fg.EvaluateEnergy(state);
				double sfg1_E = sfg1.FG()->EvaluateEnergy(state);
				double sfg2_E = sfg2.FG()->EvaluateEnergy(state);
				BOOST_CHECK_CLOSE_ABS(orig_E, sfg1_E + sfg2_E, 1.0e-5);
			}
		}
	}

	// TODO: how to check backwardmap?
}

