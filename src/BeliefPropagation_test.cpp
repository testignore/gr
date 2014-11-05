
#include <vector>
#include <limits>
#include <iostream>
#include <ctime>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "BeliefPropagation.h"
#include "GibbsSampler.h"
#include "GibbsInference.h"
#include "AISInference.h"

#define BOOST_TEST_MODULE(BeliefPropagationTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(EnergyMinimizationStress)
{
	for (unsigned int random_start = 0; random_start < 100; ++random_start) {
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
		Grante::BeliefPropagation bpinf(&fg);
		std::vector<unsigned int> bpinf_state;
		double bpinf_energy = bpinf.MinimizeEnergy(bpinf_state);

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
		BOOST_CHECK_CLOSE_ABS(min_var_energy, bpinf_energy, 1.0e-6);
	}
}

BOOST_AUTO_TEST_CASE(EnergyMinimizationGrid)
{
	// PRNG
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 10;

	// Create a factor graph from the model (binary variables)
	std::vector<unsigned int> vc(N*N, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
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
		for (unsigned int x = 0; x < N; ++x) {
			// Vertical edge
			var_index[0] = (y-1)*N + x;
			var_index[1] = y*N + x;
			for (unsigned int di = 0; di < data.size(); ++di)
				data[di] = randu();

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}
	fg.ForwardMap();

	// Perform inference
	Grante::BeliefPropagation bpinf(&fg);
	std::vector<unsigned int> bp_min_state;
	double bp_min_energy = bpinf.MinimizeEnergy(bp_min_state);

	// Compare against Gibbs sampling
	Grante::GibbsInference ginf(&fg);
	std::vector<std::vector<unsigned int> > states;
	ginf.Sample(states, 100);
	for (unsigned int si = 0; si < states.size(); ++si) {
#if 0
		std::cout << "BP energy " << bp_min_energy
			<< ", sample energy " << fg.EvaluateEnergy(states[si])
			<< std::endl;
#endif
		BOOST_CHECK_LT(bp_min_energy, fg.EvaluateEnergy(states[si]));
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
	Grante::BeliefPropagation bpinf(&fg, Grante::BeliefPropagation::Sequential);
	bpinf.PerformInference();

	double log_z = bpinf.LogPartitionFunction();
	std::cout << "log_z " << log_z << std::endl;
	BOOST_CHECK_CLOSE_ABS(0.4836311, log_z, 1e-3);

	const std::vector<double>& m_fac0 = bpinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.4132795, m_fac0[0], 1e-3);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[1], 1e-3);
	BOOST_CHECK_CLOSE_ABS(0.2506666, m_fac0[2], 1e-3);
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[3], 1e-3);

	const std::vector<double>& m_fac1 = bpinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.6639461, m_fac1[0], 1e-3);
	BOOST_CHECK_CLOSE_ABS(0.3360538, m_fac1[1], 1e-3);

	const std::vector<double>& m_fac2 = bpinf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5813064, m_fac2[0], 1e-3);
	BOOST_CHECK_CLOSE_ABS(0.4186935, m_fac2[1], 1e-3);

	// Check Gibbs sampler
	Grante::GibbsInference ginf(&fg);
	ginf.SetSamplingParameters(10000, 10, 100000);
	ginf.PerformInference();
	std::cout << "Gibbs log_z " << ginf.LogPartitionFunction() << std::endl;

	const std::vector<double>& g_fac0 = ginf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.4132795, g_fac0[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.1680269, g_fac0[1], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.2506666, g_fac0[2], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.1680269, g_fac0[3], 0.01);

	const std::vector<double>& g_fac1 = ginf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.6639461, g_fac1[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.3360538, g_fac1[1], 0.01);

	const std::vector<double>& g_fac2 = ginf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5813064, g_fac2[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.4186935, g_fac2[1], 0.01);
}

BOOST_AUTO_TEST_CASE(SimpleRandomGrid)
{
	// PRNG
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 5;

	// Create a factor graph from the model (binary variables)
	std::vector<unsigned int> vc(N*N, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
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
		for (unsigned int x = 0; x < N; ++x) {
			// Vertical edge
			var_index[0] = (y-1)*N + x;
			var_index[1] = y*N + x;
			for (unsigned int di = 0; di < data.size(); ++di)
				data[di] = randu();

			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}
	fg.ForwardMap();

	// Perform inference
	Grante::BeliefPropagation bpinf(&fg, Grante::BeliefPropagation::Sequential);
	bpinf.SetParameters(false, 100, 1.0e-6);
	bpinf.PerformInference();
	double log_z_seq = bpinf.LogPartitionFunction();
	std::cout << "log_z(seq) " << log_z_seq << std::endl;

	Grante::BeliefPropagation bpinf_par(&fg,
		Grante::BeliefPropagation::ParallelSync);
	bpinf_par.SetParameters(false, 100, 1.0e-6);
	bpinf_par.PerformInference();
	double log_z_par = bpinf_par.LogPartitionFunction();
	std::cout << "log_z(par) " << log_z_par << std::endl;

	// Compare partition function against AIS
	Grante::AISInference aisinf(&fg);
	aisinf.SetSamplingParameters(40, 1, 1000);
	aisinf.PerformInference();
	double ais_log_z = aisinf.LogPartitionFunction();
	std::cout << "AIS log_z " << ais_log_z << std::endl;

	// Compare against extensive Gibbs inference
	Grante::GibbsInference ginf(&fg);
	ginf.SetSamplingParameters(10000, 1, 200000);
	ginf.PerformInference();
//	std::cout << "Gibbs log_z " << ginf.LogPartitionFunction() << std::endl;
	const std::vector<double>& gibbs200k_fac0 = ginf.Marginal(0);

	// Compare against Gibbs sampling inference
	Grante::GibbsSampler gibbs(&fg);

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 2500;
	states.reserve(sample_count);
	gibbs.Sweep(1000);
	std::vector<double> gibbs_fac0(4, 0.0);
	for (unsigned int si = 0; si < sample_count; ++si) {
		gibbs.Sweep(100);
		gibbs_fac0[gibbs.State()[0] + 2*gibbs.State()[1]] +=
			static_cast<double>(1.0 / sample_count);
	}
	// Compare marginals of first factor (var 0 and var 1)
	const std::vector<double>& marg_fac0 = bpinf.Marginal(0);
	for (unsigned int s0 = 0; s0 < 2; ++s0) {
		for (unsigned int s1 = 0; s1 < 2; ++s1) {
			std::cout << "fac0 (" << s0 << "," << s1 << "): "
				<< "gibbs(2.5k) " << gibbs_fac0[s0+2*s1]
				<< ", gibbs(200k) " << gibbs200k_fac0[s0+2*s1]
				<< ", lbp " << marg_fac0[s0+2*s1] << std::endl;
			BOOST_CHECK_CLOSE_ABS(gibbs_fac0[s0+2*s1], marg_fac0[s0+2*s1], 0.05);
		}
	}
}

