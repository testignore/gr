
#include <iostream>
#include <cmath>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "Factor.h"
#include "FactorGraphModel.h"
#include "FactorConditioningTable.h"
#include "FactorGraphPartialObservation.h"
#include "Conditioning.h"
#include "TreeInference.h"

#define BOOST_TEST_MODULE(ConditioningTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(SimpleConditioning)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);	// (0,0)
	w.push_back(0.3);	// (1,0)
	w.push_back(0.2);	// (0,1)
	w.push_back(0.0);	// (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(2);
	state[0] = 0;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(0.4, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 0;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(0.9, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 1;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 1;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1e-5);

	// Condition the factor graph
	Grante::FactorConditioningTable ftab;
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_var_state;
	// Condition on state[1] = 0
	cond_var_set.push_back(1);
	cond_var_state.push_back(0);
	std::vector<unsigned int> var_new_to_orig;

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_var_state);
	Grante::FactorGraph* fg_cond = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_cond->ForwardMap();
	//fg_cond->Print();

	// Perform inference
	Grante::TreeInference tinf_uncond(&fg);
	Grante::TreeInference tinf_cond(fg_cond);
	tinf_uncond.PerformInference();
	tinf_cond.PerformInference();
	double logZ_uncond = tinf_uncond.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.48363, logZ_uncond, 1.0e-5);
	double logZ_cond = tinf_cond.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.24115, logZ_cond, 1.0e-5);

	std::vector<unsigned int> state_cond(1);
	state_cond[0] = 0;
	BOOST_CHECK_CLOSE_ABS(0.71095, std::exp(-fg_cond->EvaluateEnergy(state_cond))
		/ std::exp(logZ_cond), 1.0e-5);
	state_cond[0] = 1;
	BOOST_CHECK_CLOSE_ABS(0.28905, std::exp(-fg_cond->EvaluateEnergy(state_cond))
		/ std::exp(logZ_cond), 1.0e-5);

	state_cond[0] = 0;
	double energy_0 = fg_cond->EvaluateEnergy(state_cond);
	BOOST_CHECK_CLOSE_ABS(0.4-0.3, energy_0, 1.0e-5);

	state_cond[0] = 1;
	double energy_1 = fg_cond->EvaluateEnergy(state_cond);
	BOOST_CHECK_CLOSE_ABS(1.3-0.3, energy_1, 1.0e-5);

	// TODO: compare probabilities, not energies.  Energies change with
	// constant bias, probabilities should not change
	delete fg_cond;
}

BOOST_AUTO_TEST_CASE(ConditioningHO)
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

	// Add higher order factor type
	card.push_back(2);
	w.push_back(1.0);
	w.push_back(2.0);
	w.push_back(0.4);
	w.push_back(0.7);
	Grante::FactorType* factortype3 = new Grante::FactorType("tripple", card, w);
	model.AddFactorType(factortype3);

	// Create a factor graph from the model: 5 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors:
	//
	// (0) --[0]-- (1)     (2) --[1]-- (4)
	//              |       |
	//               --[3]--
	//                  |
	//                 (3)
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	const Grante::FactorType* pt3 = model.FindFactorType("tripple");
	BOOST_REQUIRE(pt != 0);
	BOOST_REQUIRE(pt3 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 2;
	var_index[1] = 4;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	var_index[0] = 1;
	var_index[1] = 2;
	var_index.push_back(3);	// tripple
	Grante::Factor* fac3 = new Grante::Factor(pt3, var_index, data);
	fg.AddFactor(fac3);
	fg.ForwardMap();

	// Condition the factor graph by observing (1)=1 and (3)=0
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_var_state;
	// Condition on state[1] = 1, state[3] = 0
	cond_var_set.push_back(1);
	cond_var_state.push_back(1);
	cond_var_set.push_back(3);
	cond_var_state.push_back(0);
	std::vector<unsigned int> var_new_to_orig;

	Grante::FactorConditioningTable ftab;
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_var_state);
	Grante::FactorGraph* fg_cond = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_cond->ForwardMap();

	// Print unconditional base energies
	std::vector<unsigned int> state(5);
	for (unsigned int s0 = 0; s0 < 2; ++s0) {
		state[0] = s0;
		for (unsigned int s1 = 0; s1 < 2; ++s1) {
			state[1] = s1;
			for (unsigned int s2 = 0; s2 < 2; ++s2) {
				state[2] = s2;
				for (unsigned int s3 = 0; s3 < 2; ++s3) {
					state[3] = s3;
					for (unsigned int s4 = 0; s4 < 2; ++s4) {
						state[4] = s4;
						double energy = fg.EvaluateEnergy(state);
						std::cout << "(" << state[0] << "," << state[1] << ","
							<< state[2] << "," << state[3] << "," << state[4]
							<< "): " << energy << std::endl;
					}
				}
			}
		}
	}

	// Test conditioned energies
	std::vector<unsigned int> state_cond(3);
	std::vector<unsigned int> state_uncond(5);
	state_uncond[1] = 1;
	state_uncond[3] = 0;
	for (unsigned int s0 = 0; s0 < 2; ++s0) {
		state_cond[0] = s0;	// 0
		state_uncond[0] = s0;
		for (unsigned int s2 = 0; s2 < 2; ++s2) {
			state_cond[1] = s2;	// 2
			state_uncond[2] = s2;
			for (unsigned int s4 = 0; s4 < 2; ++s4) {
				state_cond[2] = s4;	// 4
				state_uncond[4] = s4;
				double energy_uncond = fg.EvaluateEnergy(state_uncond);
				double energy_cond = fg_cond->EvaluateEnergy(state_cond);
				BOOST_CHECK_CLOSE_ABS(energy_uncond, energy_cond, 1.0e-5);
			}
		}
	}
	// TODO: compare probabilities, not energies.  Energies change with
	// constant bias, probabilities should not change
	//fg_cond->Print();
	delete fg_cond;
}

BOOST_AUTO_TEST_CASE(SimpleConditioningExpect)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);	// (0,0)
	w.push_back(0.3);	// (1,0)
	w.push_back(0.2);	// (0,1)
	w.push_back(0.0);	// (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(2);
	state[0] = 0;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(0.4, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 0;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(0.9, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 1;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1e-5);
	state[0] = 1;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1e-5);

	// Condition the factor graph
	Grante::FactorConditioningTable ftab;
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_fac_set;
	std::vector<std::vector<double> > cond_obs_expect;
	cond_var_set.push_back(1);

	cond_fac_set.push_back(0);
	std::vector<double> fac0_condvi1_marg;
	fac0_condvi1_marg.push_back(1.0);
	fac0_condvi1_marg.push_back(0.0);
	cond_obs_expect.push_back(fac0_condvi1_marg);

	cond_fac_set.push_back(2);
	std::vector<double> fac2_condvi1_marg;
	fac2_condvi1_marg.push_back(1.0);
	fac2_condvi1_marg.push_back(0.0);
	cond_obs_expect.push_back(fac2_condvi1_marg);

	std::vector<unsigned int> var_new_to_orig;

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_fac_set,
		cond_obs_expect);
	Grante::FactorGraph* fg_cond = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_cond->ForwardMap();

	// Perform inference
	Grante::TreeInference tinf_uncond(&fg);
	Grante::TreeInference tinf_cond(fg_cond);
	tinf_uncond.PerformInference();
	tinf_cond.PerformInference();
	double logZ_uncond = tinf_uncond.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.48363, logZ_uncond, 1.0e-5);
	double logZ_cond = tinf_cond.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.24115, logZ_cond, 1.0e-5);

	std::vector<unsigned int> state_cond(1);
	state_cond[0] = 0;
	BOOST_CHECK_CLOSE_ABS(0.71095, std::exp(-fg_cond->EvaluateEnergy(state_cond))
		/ std::exp(logZ_cond), 1.0e-5);
	state_cond[0] = 1;
	BOOST_CHECK_CLOSE_ABS(0.28905, std::exp(-fg_cond->EvaluateEnergy(state_cond))
		/ std::exp(logZ_cond), 1.0e-5);

	state_cond[0] = 0;
	double energy_0 = fg_cond->EvaluateEnergy(state_cond);
	BOOST_CHECK_CLOSE_ABS(0.4-0.3, energy_0, 1.0e-5);

	state_cond[0] = 1;
	double energy_1 = fg_cond->EvaluateEnergy(state_cond);
	BOOST_CHECK_CLOSE_ABS(1.3-0.3, energy_1, 1.0e-5);

	delete fg_cond;
}

BOOST_AUTO_TEST_CASE(SimpleConditioningExpectHO)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);	// (0,0,0)
	w.push_back(0.3);	// (1,0,0)
	w.push_back(0.2);	// (0,1,0)
	w.push_back(0.1);	// (1,1,0)
	w.push_back(0.4);	// (0,0,1)
	w.push_back(0.5);	// (1,0,1)
	w.push_back(0.9);	// (0,1,1)
	w.push_back(0.6);	// (1,1,1)
	Grante::FactorType* factortype = new Grante::FactorType("tripple", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt3 = model.FindFactorType("tripple");
	BOOST_REQUIRE(pt3 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(3);
	var_index[0] = 0;
	var_index[1] = 1;
	var_index[2] = 2;
	Grante::Factor* fac1 = new Grante::Factor(pt3, var_index, data);
	fg.AddFactor(fac1);

	// Compute the forward map
	fg.ForwardMap();

	// Condition the factor graph
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_fac_set;
	std::vector<std::vector<double> > cond_obs_expect;

	// Condition variable 1, affecting factor 0
	cond_var_set.push_back(1);
	cond_fac_set.push_back(0);
	std::vector<double> fac0_condvi1_marg;
	fac0_condvi1_marg.push_back(0.3);
	fac0_condvi1_marg.push_back(0.7);
	cond_obs_expect.push_back(fac0_condvi1_marg);
	std::vector<unsigned int> var_new_to_orig;

	// Test conditioned energies
	Grante::FactorConditioningTable ftab;
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_fac_set,
		cond_obs_expect);
	Grante::FactorGraph* fg_cond = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_cond->ForwardMap();

	// Perform inference on conditioned model
	Grante::TreeInference tinf_cond(fg_cond);
	tinf_cond.PerformInference();
	const std::vector<double>& c_fac0 = tinf_cond.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.31505, c_fac0[0], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.30882, c_fac0[1], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.17118, c_fac0[2], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.20495, c_fac0[3], 1.0e-5);

	delete fg_cond;
}

BOOST_AUTO_TEST_CASE(SimpleSequentialConditioning)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);	// (0,0)
	w.push_back(0.3);	// (1,0)
	w.push_back(0.2);	// (0,1)
	w.push_back(0.0);	// (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	w1[0] = 0.4;
	w1[1] = 0.9;
	Grante::FactorType* factortype1c = new Grante::FactorType("unary3", card1, w1);
	model.AddFactorType(factortype1c);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	const Grante::FactorType* pt1c = model.FindFactorType("unary3");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	BOOST_REQUIRE(pt1c != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	fg.AddFactor(new Grante::Factor(pt2, var_index, data));
	var_index[0] = 1;
	var_index[1] = 2;
	fg.AddFactor(new Grante::Factor(pt2, var_index, data));

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	fg.AddFactor(new Grante::Factor(pt1a, var_index1, data));
	var_index1[0] = 1;
	fg.AddFactor(new Grante::Factor(pt1b, var_index1, data));
	var_index1[0] = 2;
	fg.AddFactor(new Grante::Factor(pt1c, var_index1, data));

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(3);
	state[0] = 0;
	state[1] = 0;
	state[2] = 0;
	BOOST_CHECK_CLOSE_ABS(0.8, fg.EvaluateEnergy(state), 1.0e-6);
	state[0] = 0;
	state[1] = 1;
	state[2] = 0;
	BOOST_CHECK_CLOSE_ABS(1.6, fg.EvaluateEnergy(state), 1.0e-6);
	state[0] = 1;
	state[1] = 1;
	state[2] = 1;
	BOOST_CHECK_CLOSE_ABS(2.2, fg.EvaluateEnergy(state), 1.0e-6);

	// Condition the factor graph
	Grante::FactorConditioningTable ftab;
	std::vector<unsigned int> cond_var_set;
	std::vector<unsigned int> cond_var_state;
	// Condition on state[0] = 1
	cond_var_set.push_back(0);
	cond_var_state.push_back(1);
	std::vector<unsigned int> var_new_to_orig;

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobs(cond_var_set, cond_var_state);
	Grante::FactorGraph* fg_cond1 = Grante::Conditioning::ConditionFactorGraph(
		&ftab, &fg, &pobs, var_new_to_orig);
	fg_cond1->ForwardMap();

	BOOST_CHECK(var_new_to_orig[0] == 1);
	BOOST_CHECK(var_new_to_orig[1] == 2);

	std::vector<unsigned int> state1(2);
	state1[0] = 0;
	state1[1] = 0;
	BOOST_CHECK_CLOSE_ABS(1.7-0.7, fg_cond1->EvaluateEnergy(state1), 1.0e-6);
	state1[0] = 1;
	state1[1] = 0;
	BOOST_CHECK_CLOSE_ABS(2.0-0.7, fg_cond1->EvaluateEnergy(state1), 1.0e-6);
	state1[0] = 1;
	state1[1] = 1;
	BOOST_CHECK_CLOSE_ABS(2.2-0.7, fg_cond1->EvaluateEnergy(state1), 1.0e-6);

	// Condition a second time
	std::vector<unsigned int> cond_var_set2;
	std::vector<unsigned int> cond_var_state2;
	// Condition on state[2] = 1
	cond_var_set2.push_back(1);
	cond_var_state2.push_back(1);
	std::vector<unsigned int> var_new_to_orig2;

	// Test conditioned energies
	Grante::FactorGraphPartialObservation pobs2(cond_var_set2, cond_var_state2);
	Grante::FactorGraph* fg_cond2 = Grante::Conditioning::ConditionFactorGraph(
		&ftab, fg_cond1, &pobs2, var_new_to_orig2);
	fg_cond2->ForwardMap();

	BOOST_CHECK(var_new_to_orig2[0] == 0);
	std::vector<unsigned int> state2(1);
	state2[0] = 0;
	BOOST_CHECK_CLOSE_ABS(2.4-0.7-0.9, fg_cond2->EvaluateEnergy(state2), 1.0e-6);
	state2[0] = 1;
	BOOST_CHECK_CLOSE_ABS(2.2-0.7-0.9, fg_cond2->EvaluateEnergy(state2), 1.0e-6);

	Grante::TreeInference tinf_cond1(fg_cond1);
	tinf_cond1.PerformInference();
	Grante::TreeInference tinf_cond2(fg_cond2);
	tinf_cond2.PerformInference();

	delete fg_cond2;
	delete fg_cond1;
}

// TODO: test conditioning-on-expectation

