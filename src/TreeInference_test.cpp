
#include <vector>
#include <limits>
#include <iostream>
#include <ctime>
#include <cmath>

#include <boost/timer.hpp>
#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "BruteForceExactInference.h"
#include "TreeInference.h"
#include "GibbsSampler.h"

#define BOOST_TEST_MODULE(TreeInferenceTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(SimpleEntropy)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 1.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1(2, 1.0);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
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
	Grante::Factor* fac1b = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	double log_z = tinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(std::log(4.0*std::exp(-3.0)), log_z, 1.0e-5);

	// Entropy should be 2 bits
	double entropy = tinf.Entropy();
	std::cout << "entropy " << entropy << std::endl;
	BOOST_CHECK_CLOSE_ABS(-4.0*0.25*std::log(0.25), entropy, 1.0e-5);

	double base_entropy = tinf.InferenceMethod::Entropy();
	std::cout << "base_entropy " << base_entropy << std::endl;
	BOOST_CHECK_CLOSE_ABS(entropy, base_entropy, 1.0e-5);
}

BOOST_AUTO_TEST_CASE(EnergyMinimizationStress)
{
	for (unsigned int random_start = 0; random_start < 10000; ++random_start) {
		// Randomly set the data observations
		boost::mt19937 rgen(static_cast<const boost::uint32_t>(random_start));
		boost::uniform_real<double> rdestu;	// range [0,1]
		boost::variate_generator<boost::mt19937,
			boost::uniform_real<double> > randu(rgen, rdestu);

		Grante::FactorGraphModel model;

		// Create one simple parametrized, data-independent pairwise factor type
		std::vector<unsigned int> card;
		card.push_back(2);
		std::vector<double> w;

		Grante::FactorType* factortype_u = new Grante::FactorType("unary", card, w);
		model.AddFactorType(factortype_u);

		card.push_back(2);
		Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
		model.AddFactorType(factortype);

		// Create a N-by-N grid-structured model
		unsigned int N = 2;

		// Create a factor graph for the model
		std::vector<unsigned int> vc(N*N, 2);
		Grante::FactorGraph fg(&model, vc);

		// Add unary factors
		Grante::FactorType* pt_u = model.FindFactorType("unary");
		std::vector<double> data_u(2);
		std::vector<unsigned int> var_index_u(1);
		for (unsigned int y = 0; y < N; ++y) {
			for (unsigned int x = 0; x < N; ++x) {
				var_index_u[0] = y*N + x;
				for (unsigned int di = 0; di < data_u.size(); ++di)
					data_u[di] = randu();
				Grante::Factor* fac = new Grante::Factor(pt_u, var_index_u, data_u);
				fg.AddFactor(fac);
			}
		}

		// Add pairwise factors
		Grante::FactorType* pt = model.FindFactorType("pairwise");
		BOOST_REQUIRE(pt != 0);
		std::vector<double> data(4);
		std::vector<unsigned int> var_index(2);
		for (unsigned int y = 0; y < N; ++y) {
			for (unsigned int x = 1; x < N; ++x) {
				// Horizontal edge
				var_index[0] = y*N + x - 1;
				var_index[1] = y*N + x;

				for (unsigned int di = 0; di < data.size(); ++di)
					data[di] = randu();
				Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
				fg.AddFactor(fac);
			}
		}
		for (unsigned int y = 1; y < N; ++y) {
			unsigned int x = 0;

			// Vertical edge
			var_index[0] = (y-1)*N + x;
			var_index[1] = y*N + x;

			for (unsigned int di = 0; di < data.size(); ++di)
				data[di] = randu();
			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}

		// fg is now a N-by-N tree-structured factor graph.  Minimize energy.
		fg.ForwardMap();
		Grante::TreeInference tinf(&fg);
		std::vector<unsigned int> tinf_state;
		double tinf_energy = tinf.MinimizeEnergy(tinf_state);

		// Find minimum energy state by exhaustive search
		std::vector<unsigned int> test_var(4);
		std::vector<unsigned int> min_var(4);
		double min_var_energy = std::numeric_limits<double>::infinity();
		for (unsigned int v0 = 0; v0 < 2; ++v0) {
			test_var[0] = v0;
			for (unsigned int v1 = 0; v1 < 2; ++v1) {
				test_var[1] = v1;
				for (unsigned int v2 = 0; v2 < 2; ++v2) {
					test_var[2] = v2;
					for (unsigned int v3 = 0; v3 < 2; ++v3) {
						test_var[3] = v3;

						double orig_e = fg.EvaluateEnergy(test_var);
						if (orig_e < min_var_energy) {
							min_var = test_var;
							min_var_energy = orig_e;
						}
					}
				}
			}
		}
		BOOST_CHECK_CLOSE_ABS(min_var_energy, tinf_energy, 1.0e-6);
	}
}

BOOST_AUTO_TEST_CASE(Simple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);
	w.push_back(0.3);
	w.push_back(0.2);
	w.push_back(0.0);
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

	// Test inference results
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	double log_z = tinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.4836311, log_z, 1.0e-6);

	const std::vector<double>& m_fac0 = tinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.4132795, m_fac0[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[1], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.2506666, m_fac0[2], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[3], 1.0e-6);

	const std::vector<double>& m_fac1 = tinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.6639461, m_fac1[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3360538, m_fac1[1], 1.0e-6);

	const std::vector<double>& m_fac2 = tinf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5813064, m_fac2[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.4186935, m_fac2[1], 1.0e-6);

	// Brute-force check
	Grante::BruteForceExactInference bfinf(&fg);
	bfinf.PerformInference();

	log_z = bfinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.4836311, log_z, 1.0e-6);

	const std::vector<double>& m2_fac0 = bfinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.4132795, m2_fac0[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m2_fac0[1], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.2506666, m2_fac0[2], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m2_fac0[3], 1.0e-6);

	const std::vector<double>& m2_fac1 = bfinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.6639461, m2_fac1[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.3360538, m2_fac1[1], 1.0e-6);

	const std::vector<double>& m2_fac2 = bfinf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5813064, m2_fac2[0], 1.0e-6);
	BOOST_CHECK_CLOSE_ABS(0.4186935, m2_fac2[1], 1.0e-6);
}

BOOST_AUTO_TEST_CASE(SimpleDisconnected)
{
	Grante::FactorGraphModel model;

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
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	double log_z = tinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.591843, log_z, 1.0e-5);

	const std::vector<double>& m_fac1 = tinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.64566, m_fac1[0], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.354346, m_fac1[1], 1.0e-4);

	const std::vector<double>& m_fac2 = tinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.57444, m_fac2[0], 1.0e-4);
	BOOST_CHECK_CLOSE_ABS(0.42556, m_fac2[1], 1.0e-4);
}

BOOST_AUTO_TEST_CASE(SimpleLarge)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);
	w.push_back(0.3);
	w.push_back(0.2);
	w.push_back(0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	// Create a factor graph from the model: n binary variables
	std::vector<unsigned int> vcs;
	vcs.push_back(100);
	vcs.push_back(1000);
	vcs.push_back(2500);
	vcs.push_back(5000);
	vcs.push_back(7500);
	vcs.push_back(10000);
	vcs.push_back(25000);
	vcs.push_back(50000);
	vcs.push_back(75000);
	vcs.push_back(100000);
	for (std::vector<unsigned int>::const_iterator vcsi = vcs.begin();
		vcsi != vcs.end(); ++vcsi) {
		unsigned int var_count = *vcsi;
		std::vector<unsigned int> vc(var_count, 2);
		Grante::FactorGraph fg(&model, vc);

		// Add factors
		const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
		const Grante::FactorType* pt1a = model.FindFactorType("unary1");
		BOOST_REQUIRE(pt2 != 0);
		BOOST_REQUIRE(pt1a != 0);
		std::vector<double> data;
		std::vector<unsigned int> var_index(2);

		// Link each variable to its parent
		boost::timer setup_timer;
		for (unsigned int vi = 0; vi < var_count; ++vi) {
			// Root node has no parent
			if (vi != 0) {
				var_index[0] = vi;
				var_index[1] = (vi - 1) / 2;	// parent in binary heap
				Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
				fg.AddFactor(fac1);
			}

			std::vector<unsigned int> var_index1(1);
			var_index1[0] = vi;
			Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
			fg.AddFactor(fac1a);
		}
		double setup_time = setup_timer.elapsed();

		// Compute the forward map
		boost::timer forward_timer;
		fg.ForwardMap();
		double forward_time = forward_timer.elapsed();

		// Test inference results
		boost::timer inf_setup_timer;
		Grante::TreeInference tinf(&fg);
		double inf_setup_time = inf_setup_timer.elapsed();
		boost::timer inf_timer;
		tinf.PerformInference();
		double inf_time = inf_timer.elapsed();
		std::cout << ", " << var_count << ": " << setup_time << " "
			<< forward_time << " " << inf_setup_time << " "
			<< inf_time;
	}
}

BOOST_AUTO_TEST_CASE(SamplingSimple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);
	w.push_back(0.3);
	w.push_back(0.2);
	w.push_back(0.0);
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

	// Test inference results
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	// Exact inference results
	const std::vector<double>& m_fac0 = tinf.Marginal(0);
	const std::vector<double>& m_fac1 = tinf.Marginal(1);
	const std::vector<double>& m_fac2 = tinf.Marginal(2);

	// Produce samples: exact sampling on tree
	unsigned int sample_count = 30000;
	std::vector<std::vector<unsigned int> > states(sample_count);
	tinf.Sample(states, sample_count);

	std::vector<double> s_fac0(4, 0.0);
	std::vector<double> s_fac1(4, 0.0);
	std::vector<double> s_fac2(4, 0.0);
	BOOST_REQUIRE(states.size() == sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		const std::vector<unsigned int>& state = states[si];
		s_fac0[state[0]+2*state[1]] += 1.0;
		s_fac1[state[0]] += 1.0;
		s_fac2[state[1]] += 1.0;
	}
	for (unsigned int fi = 0; fi < s_fac0.size(); ++fi)
		s_fac0[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac1.size(); ++fi)
		s_fac1[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac2.size(); ++fi)
		s_fac2[fi] /= static_cast<double>(sample_count);

	BOOST_CHECK_CLOSE_ABS(m_fac0[0], s_fac0[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[1], s_fac0[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[2], s_fac0[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[3], s_fac0[3], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[0], s_fac1[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[1], s_fac1[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[0], s_fac2[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[1], s_fac2[1], 0.025);

	// Produce samples: Gibbs sampler
	Grante::GibbsSampler gs(&fg);
	gs.Sweep(100);
	for (unsigned int si = 0; si < sample_count; ++si) {
		states[si] = gs.State();
		gs.Sweep(10);
	}
	std::fill(s_fac0.begin(), s_fac0.end(), 0.0);
	std::fill(s_fac1.begin(), s_fac1.end(), 0.0);
	std::fill(s_fac2.begin(), s_fac2.end(), 0.0);

	BOOST_REQUIRE(states.size() == sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		const std::vector<unsigned int>& state = states[si];
		s_fac0[state[0]+2*state[1]] += 1.0;
		s_fac1[state[0]] += 1.0;
		s_fac2[state[1]] += 1.0;
	}
	for (unsigned int fi = 0; fi < s_fac0.size(); ++fi)
		s_fac0[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac1.size(); ++fi)
		s_fac1[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac2.size(); ++fi)
		s_fac2[fi] /= static_cast<double>(sample_count);

	BOOST_CHECK_CLOSE_ABS(m_fac0[0], s_fac0[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[1], s_fac0[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[2], s_fac0[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[3], s_fac0[3], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[0], s_fac1[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[1], s_fac1[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[0], s_fac2[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[1], s_fac2[1], 0.025);
}

BOOST_AUTO_TEST_CASE(SamplingHO)
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

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
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
	var_index.push_back(3);
	Grante::Factor* fac3 = new Grante::Factor(pt3, var_index, data);
	fg.AddFactor(fac3);

	// Visualize tree order
#if 0
	std::vector<Grante::FactorGraphStructurizer::OrderStep> order;
	Grante::FactorGraphStructurizer::ComputeTreeOrder(&fg, order);
	Grante::FactorGraphStructurizer::PrintOrder(order);
#endif

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	// Exact inference results
	const std::vector<double>& m_fac0 = tinf.Marginal(0);
	const std::vector<double>& m_fac1 = tinf.Marginal(1);
	const std::vector<double>& m_fac2 = tinf.Marginal(2);

	// Produce samples: Gibbs sampling
	unsigned int sample_count = 30000;
	std::vector<std::vector<unsigned int> > states(sample_count);
	Grante::GibbsSampler gs(&fg);
	gs.Sweep(100);
	for (unsigned int si = 0; si < sample_count; ++si) {
		states[si] = gs.State();
		gs.Sweep(10);
	}

	std::vector<double> s_fac0(4, 0.0);
	std::vector<double> s_fac1(4, 0.0);
	std::vector<double> s_fac2(8, 0.0);
	BOOST_REQUIRE(states.size() == sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		const std::vector<unsigned int>& state = states[si];
		s_fac0[state[0]+2*state[1]] += 1.0;
		s_fac1[state[2]+2*state[4]] += 1.0;
		s_fac2[state[1]+2*state[2]+4*state[3]] += 1.0;
	}
	for (unsigned int fi = 0; fi < s_fac0.size(); ++fi)
		s_fac0[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac1.size(); ++fi)
		s_fac1[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac2.size(); ++fi)
		s_fac2[fi] /= static_cast<double>(sample_count);

	BOOST_CHECK_CLOSE_ABS(m_fac0[0], s_fac0[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[1], s_fac0[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[2], s_fac0[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[3], s_fac0[3], 0.025);

	BOOST_CHECK_CLOSE_ABS(m_fac1[0], s_fac1[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[1], s_fac1[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[2], s_fac1[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[3], s_fac1[3], 0.025);

	BOOST_CHECK_CLOSE_ABS(m_fac2[0], s_fac2[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[1], s_fac2[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[2], s_fac2[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[3], s_fac2[3], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[4], s_fac2[4], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[5], s_fac2[5], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[6], s_fac2[6], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[7], s_fac2[7], 0.025);

	// Produce samples: exact sampling
	tinf.Sample(states, sample_count);

	std::fill(s_fac0.begin(), s_fac0.end(), 0.0);
	std::fill(s_fac1.begin(), s_fac1.end(), 0.0);
	std::fill(s_fac2.begin(), s_fac2.end(), 0.0);
	BOOST_REQUIRE(states.size() == sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		const std::vector<unsigned int>& state = states[si];
		s_fac0[state[0]+2*state[1]] += 1.0;
		s_fac1[state[2]+2*state[4]] += 1.0;
		s_fac2[state[1]+2*state[2]+4*state[3]] += 1.0;
	}
	for (unsigned int fi = 0; fi < s_fac0.size(); ++fi)
		s_fac0[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac1.size(); ++fi)
		s_fac1[fi] /= static_cast<double>(sample_count);
	for (unsigned int fi = 0; fi < s_fac2.size(); ++fi)
		s_fac2[fi] /= static_cast<double>(sample_count);

	BOOST_CHECK_CLOSE_ABS(m_fac0[0], s_fac0[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[1], s_fac0[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[2], s_fac0[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac0[3], s_fac0[3], 0.025);

	BOOST_CHECK_CLOSE_ABS(m_fac2[0], s_fac2[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[1], s_fac2[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[2], s_fac2[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[3], s_fac2[3], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[4], s_fac2[4], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[5], s_fac2[5], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[6], s_fac2[6], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac2[7], s_fac2[7], 0.025);

	BOOST_CHECK_CLOSE_ABS(m_fac1[0], s_fac1[0], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[1], s_fac1[1], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[2], s_fac1[2], 0.025);
	BOOST_CHECK_CLOSE_ABS(m_fac1[3], s_fac1[3], 0.025);
}

BOOST_AUTO_TEST_CASE(SimpleEnergyMinimization)
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

	// Test inference results
	Grante::TreeInference tinf(&fg);
	std::vector<unsigned int> min_state(2);
	double min_energy = tinf.MinimizeEnergy(min_state);

	BOOST_CHECK_CLOSE_ABS(0.4, min_energy, 1.0e-5);
	BOOST_CHECK(min_state[0] == 0 && min_state[1] == 0);
}

