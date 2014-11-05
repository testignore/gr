
#include <vector>
#include <limits>
#include <iostream>
#include <ctime>
#include <cmath>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "NaiveMeanFieldInference.h"
#include "StructuredMeanFieldInference.h"
#include "BeliefPropagation.h"
#include "GibbsInference.h"
#include "MultichainGibbsInference.h"
#include "AISInference.h"
#include "ParallelTemperingInference.h"
#include "SwendsenWangInference.h"
#include "FactorConditioningTable.h"

#define BOOST_TEST_MODULE(MeanfieldTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(MiniNaiveMeanfield)
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
	Grante::FactorConditioningTable fcond_tab;
	std::vector<bool> factor_is_removed(1, true);
	Grante::StructuredMeanFieldInference mfinf(&fg, &fcond_tab,
		factor_is_removed);
	mfinf.PerformInference();

	// Ground truth from prototype/mfield.m
	double log_z = mfinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.52342, log_z, 1.0e-4);
	BOOST_CHECK_LT(log_z, 0.55390);	// exact log partition function

	const std::vector<double>& m_fac0 = mfinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.1996, m_fac0[0], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.4131, m_fac0[1], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.1262, m_fac0[2], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.2611, m_fac0[3], 1.0e-2);

	// Naive mean field, should give the same results
	Grante::NaiveMeanFieldInference nmfinf(&fg);
	nmfinf.PerformInference();

	log_z = nmfinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(0.52342, log_z, 1.0e-4);
	BOOST_CHECK_LT(log_z, 0.55390);	// exact log partition function

	const std::vector<double>& m2_fac0 = nmfinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.1996, m2_fac0[0], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.4131, m2_fac0[1], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.1262, m2_fac0[2], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.2611, m2_fac0[3], 1.0e-2);
}

// Higher-order factor test case
BOOST_AUTO_TEST_CASE(MiniHONaiveMeanfield)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(8);
	w[0] = 0.8;
	w[1] = 0.5;
	w[2] = 2.0;
	w[3] = 0.6;
	w[4] = 0.1;
	w[5] = 0.7;
	w[6] = 0.5;
	w[7] = 0.9;
	Grante::FactorType* factortype = new Grante::FactorType("tripple", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 2 binary variables
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

	// Test inference results
	Grante::FactorConditioningTable fcond_tab;
	std::vector<bool> factor_is_removed(1, true);
	Grante::StructuredMeanFieldInference mfinf(&fg, &fcond_tab,
		factor_is_removed);
	mfinf.PerformInference();

	// Ground truth from prototype/mfield2.m
	double log_z = mfinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(1.3634, log_z, 1.0e-4);
	BOOST_CHECK_LT(log_z, 1.4242);	// exact log partition function

	const std::vector<double>& m_fac0 = mfinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.1238, m_fac0[0], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.1218, m_fac0[1], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.0794, m_fac0[2], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.0782, m_fac0[3], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.1832, m_fac0[4], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.1803, m_fac0[5], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.1176, m_fac0[6], 1.0e-3);
	BOOST_CHECK_CLOSE_ABS(0.1157, m_fac0[7], 1.0e-3);
}

#if 1
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
	Grante::FactorConditioningTable ftab;
	Grante::StructuredMeanFieldInference mfinf(&fg, &ftab);
	mfinf.PerformInference();
	Grante::StructuredMeanFieldInference mfinf_u(&fg, &ftab,
		Grante::StructuredMeanFieldInference::DecompositionType::UniformFactorWeights);
	mfinf_u.PerformInference();
	double log_z = mfinf.LogPartitionFunction();
	double log_z_u = mfinf_u.LogPartitionFunction();
	std::cout << "MF log_z(tc) " << log_z << ", log_z(u) " << log_z_u << std::endl;
	const std::vector<double>& mf_fac0 = mfinf.Marginal(0);

	// Multichain Gibbs sampling
	Grante::MultichainGibbsInference mcginf(&fg);
	mcginf.SetSamplingParameters(15, 1.01, 1, 5000);
	mcginf.PerformInference();
	const std::vector<double>& mcg_fac0 = mcginf.Marginal(0);

	// Compare against belief propagation
	Grante::BeliefPropagation bpinf(&fg);
	bpinf.PerformInference();
	std::cout << "BP log_z " << bpinf.LogPartitionFunction() << std::endl;
	const std::vector<double>& bp_fac0 = bpinf.Marginal(0);

	// AIS
	Grante::AISInference aisinf(&fg);
	aisinf.SetSamplingParameters(80, 1, 500);
	aisinf.PerformInference();
	double ais_log_z = aisinf.LogPartitionFunction();
	std::cout << "AIS log_z " << ais_log_z << std::endl;
	const std::vector<double>& ais_fac0 = aisinf.Marginal(0);

	// PT
	Grante::ParallelTemperingInference ptinf(&fg);
	ptinf.SetSamplingParameters(10, 10.0, 0.5, 100, 2000);
	ptinf.PerformInference();
	const std::vector<double>& pt_fac0 = ptinf.Marginal(0);
	const std::vector<double>& pt_accept_prob = ptinf.AcceptanceProbabilities();
	for (unsigned int li = 0; li < pt_accept_prob.size(); ++li) {
		std::cout << "PT accept in (" << li << "," << (li+1) << "): "
			<< pt_accept_prob[li] << std::endl;
	}

	// SW
	Grante::SwendsenWangInference swinf(&fg, ptinf.Marginals());
	swinf.SetSamplingParameters(true, 50, 1, 1000);
	swinf.PerformInference();
	const std::vector<double>& sw_fac0 = swinf.Marginal(0);

	// Compare against Gibbs sampling
	Grante::GibbsInference ginf(&fg);
	ginf.SetSamplingParameters(100, 0, 5000);
	ginf.PerformInference();
	const std::vector<double>& g_fac0 = ginf.Marginal(0);

	for (unsigned int state = 0; state < mf_fac0.size(); ++state) {
		std::cout << "fac0 (" << state << ")"
			<< ", mf " << mf_fac0[state]
			<< ", gibbs " << g_fac0[state]
			<< ", lbp " << bp_fac0[state]
			<< ", ais " << ais_fac0[state]
			<< ", sw " << sw_fac0[state]
			<< ", pt " << pt_fac0[state]
			<< ", mcgibbs " << mcg_fac0[state]
			<< std::endl;
	}

#if 0
	std::vector<unsigned int> state_v(N*N);
	double exact_log_z = 0.0;
	for (int si = 0; si < (1<<(N*N)); ++si) {
		for (unsigned int vi = 0; vi < (N*N); ++vi)
			state_v[vi] = (si >> vi) & 0x01;
		double energy = fg.EvaluateEnergy(state_v);
		//std::cout << "si " << si << ", energy " << energy << std::endl;
		exact_log_z += std::exp(-energy);
	}
	exact_log_z = std::log(exact_log_z);
	std::cout << "Exact log_z " << exact_log_z << std::endl;
#endif
}
#endif

