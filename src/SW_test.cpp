
#include <vector>
#include <iostream>
#include <cmath>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "SwendsenWangSampler.h"

#define BOOST_TEST_MODULE(SwendsenWangTest)
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
	w.push_back(0.0);
	w.push_back(0.3);
	w.push_back(0.2);
	w.push_back(0.0);
	Grante::FactorType* factortype =
		new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a =
		new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b =
		new Grante::FactorType("unary2", card1, w1);
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
	BOOST_CHECK_CLOSE_ABS(0.4132795, m_fac0[0], 1.0e-5);	// 0 0
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[1], 1.0e-5);	// 1 0
	BOOST_CHECK_CLOSE_ABS(0.2506666, m_fac0[2], 1.0e-5);	// 0 1
	BOOST_CHECK_CLOSE_ABS(0.1680269, m_fac0[3], 1.0e-5);	// 1 1

	const std::vector<double>& m_fac1 = tinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.6639461, m_fac1[0], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.3360538, m_fac1[1], 1.0e-5);

	const std::vector<double>& m_fac2 = tinf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5813064, m_fac2[0], 1.0e-5);
	BOOST_CHECK_CLOSE_ABS(0.4186935, m_fac2[1], 1.0e-5);

	// Swendsen-Wang
	std::vector<double> qf(3, 0.0);
	qf[0] = 0.5;
	Grante::SwendsenWangSampler sw(&fg, qf);

	sw.SampleSite(0);
}

BOOST_AUTO_TEST_CASE(NetworkReliabilitySimple)
{
	// (0) --[0]-- (1)
	//  |        /
	// [1]   [2]
	//  |  /
	// (2)
	//  |
	// [3]
	//  |
	// (3)
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype =
		new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 4 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
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

	var_index[0] = 0;
	var_index[1] = 2;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	var_index[0] = 1;
	var_index[1] = 2;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	var_index[0] = 2;
	var_index[1] = 3;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	// All factors appear with 50/50 chance
	std::vector<double> qf(4, 0.5);
	std::vector<double> qf_out(4, 0.0);
	Grante::SwendsenWangSampler::ComputeNetworkReliability(
		&fg, qf, qf_out, 100000);
	for (unsigned int fi = 0; fi < qf_out.size(); ++fi) {
		std::cout << "NR fi " << fi << ": " << qf_out[fi] << std::endl;
	}
	BOOST_CHECK_CLOSE_ABS(0.5 + 0.5*0.5*0.5, qf_out[0], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(qf_out[0], qf_out[1], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(qf_out[0], qf_out[2], 1.0e-2);
	BOOST_CHECK_CLOSE_ABS(0.5, qf_out[3], 1.0e-2);
}

BOOST_AUTO_TEST_CASE(NetworkReliabilityChain)
{
	// (0) --[s]-- (1)
	//    \         |
	//  |    [s]   [r]
	//           \  |
	//            (2)
	//    \         |
	//       [s]   [r]
	//           \  |
	//             (3)
	//              |
	//     ...     ...
	Grante::FactorGraphModel model;
	unsigned int K = 10;	// Number of nodes in the right chain
	double r = 0.9;
	double s = 0.1;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype =
		new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 4 binary variables
	std::vector<unsigned int> vc(1+K,2);
	Grante::FactorGraph fg(&model, vc);

	std::vector<double> qf;

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt2 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (unsigned int k = 0; k < K; ++k) {
		var_index[0] = 0;
		var_index[1] = 1 + k;
		Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
		fg.AddFactor(fac1);
		qf.push_back(s);

		if (k >= 1) {
			var_index[0] = k;
			var_index[1] = 1 + k;
			fac1 = new Grante::Factor(pt2, var_index, data);
			fg.AddFactor(fac1);
			qf.push_back(r);
		}
	}

	// Estimate edge appearance probabilities
	std::vector<double> edgeprob_out(qf.size(), 0.0);
	std::vector<double> qf_actual_cc(qf.size(), 0.0);
	Grante::SwendsenWangSampler::AdjustFactorProbStochastic(
		&fg, qf, edgeprob_out, qf_actual_cc, 2000);
	for (unsigned int fi = 0; fi < edgeprob_out.size(); ++fi) {
		std::cout << "NR fi " << fi << ": cc_desired " << qf[fi]
			<< ", cc_actual " << qf_actual_cc[fi]
			<< ", edge appearance " << edgeprob_out[fi] << std::endl;
	}
}

BOOST_AUTO_TEST_CASE(NetworkReliabilityContradiction)
{
	// (0) --[s]-- (1)
	//  |           |
	// [r]         [r]
	//  |           |
	// (2) --[r]-- (3)
	//
	// where r is a high (0.9) co-occurence probability and s is a small (0.1)
	// one.
	//
	// There is no edge appearance probability solution in this case.
	Grante::FactorGraphModel model;
	double r = 0.9;
	double s = 0.3;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype =
		new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 4 binary variables
	std::vector<unsigned int> vc(4,2);
	Grante::FactorGraph fg(&model, vc);

	std::vector<double> qf;

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt2 != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);

	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);
	qf.push_back(s);

	var_index[0] = 0;
	var_index[1] = 2;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);
	qf.push_back(r);

	var_index[0] = 1;
	var_index[1] = 3;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);
	qf.push_back(r);

	var_index[0] = 2;
	var_index[1] = 3;
	fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);
	qf.push_back(r);

	// Optimize edge appearance probabilities
	std::vector<double> edgeprob_out(qf.size(), 0.0);
	std::vector<double> qf_actual_cc(qf.size(), 0.0);
	double obj = Grante::SwendsenWangSampler::AdjustFactorProbStochastic(
		&fg, qf, edgeprob_out, qf_actual_cc, 2000);

	std::cout << std::endl;
	std::cout << "Infeasible network:" << std::endl;
	for (unsigned int fi = 0; fi < edgeprob_out.size(); ++fi) {
		std::cout << "NR fi " << fi << ": q_desired_cc " << qf[fi]
			<< ", edge appearance " << edgeprob_out[fi]
			<< ", qf_actual_cc " << qf_actual_cc[fi]
			<< std::endl;
	}
	std::cout << "Achieved objective: " << obj << std::endl;
	std::cout << std::endl;
}

// This test case reproduces the classic application of the Swendsen-Wang
// algorithm to sampling at subcritical temperatures of the 2D lattice Ising
// model.
BOOST_AUTO_TEST_CASE(ClassicIsing)
{
	Grante::FactorGraphModel model;

	unsigned int N = 32;	// Lattice size

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> wp(4, 0.0);
	Grante::FactorType* factortype =
		new Grante::FactorType("ising", card, wp);
	model.AddFactorType(factortype);

	// Create a 2D N-by-N lattice model, with periodic boundary conditions
	std::vector<unsigned int> vc(N*N, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("ising");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (int y = 0; y < static_cast<int>(N); ++y) {
		for (int x = 0; x < static_cast<int>(N); ++x) {
			// Horizontal edge
			var_index[0] = (N*N + y*N + x - 1) % (N*N);
			var_index[1] = y*N + x;
			Grante::Factor* fac_h = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac_h);

			// Vertical edge
			var_index[0] = (N*N + (y-1)*N + x) % (N*N);
			var_index[1] = y*N + x;
			Grante::Factor* fac_v = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac_v);
		}
	}
	std::vector<double>& w = factortype->Weights();

	std::cout << "Inv-Temp | BondProb | AVG SW  | Spont. magn. | CC(sample)"
		<< std::endl;
	std::cout << "---------+----------+---------+--------------+------------"
		<< std::endl;

	// 2D lattice critical inverse temp
	double K0 = 0.5 * std::log(1.0 + std::sqrt(2.0));
	std::vector<unsigned int> state;
	std::vector<unsigned int> state_crit;
	for (double Kf = 0.5; Kf <= 1.3; Kf += 0.1) {
		double K = Kf * K0;	// inverse temp

		// Ising energies.  Here state 0 is "-1 spin" and 1 is "+1 spin".
		w[0] = -K;
		w[1] = K;
		w[2] = K;
		w[3] = -K;
		fg.ForwardMap();

		// Swendsen-Wang, classic edge appearance probabilities
		unsigned int fac_count = fg.Factors().size();
		double qe_ising = 1.0 - std::exp(-2.0*K);
		std::vector<double> qf(fac_count, qe_ising);

#if 0
		// Optimize edge appearance probabilities
		std::vector<double> qf_out(qf.size(), 0.0);
		std::vector<double> qf_actual_cc(qf.size(), 0.0);
		double max_diff = Grante::SwendsenWangSampler::AdjustFactorProb(
			&fg, qf, qf_out, qf_actual_cc, 10, 1000);
		std::cout << "qf_adjusted = " << qf_out[0] << std::endl;
#endif
#if 0
		std::vector<double> qf(fac_count, 0.0);
		Grante::SwendsenWangSampler::ComputeFactorProb(&fg, qf, 2.0);
#endif
		Grante::SwendsenWangSampler sw(&fg, qf);

		double avg_size = 0.0;
		for (unsigned int vi = 0; vi < N*N; ++vi) {
			unsigned int sw_size = sw.SampleSite(vi);
			avg_size += sw_size;
		}
		avg_size /= static_cast<double>(N*N);

		// Output state
		state = sw.State();
		if (std::fabs(Kf - 1.0) <= 1.0e-5)
			state_crit = state;

		double cc_empirical = 0.0;
		double cc_empirical_count = 0.0;
		for (unsigned int y = 0; y < N; ++y) {
			for (unsigned int x = 1; x < N; ++x) {
				bool same_h = state[(N*N + y*N + x - 1) % (N*N)] == state[y*N + x];
				bool same_v = state[(N*N + (y-1)*N + x) % (N*N)] == state[y*N + x];
				cc_empirical += same_h ? 1.0 : 0.0;
				cc_empirical += same_v ? 1.0 : 0.0;
				cc_empirical_count += 2.0;
			}
		}
		cc_empirical /= cc_empirical_count;

		double M = 0.0;
		for (unsigned int y = 0; y < N; ++y) {
			for (unsigned int x = 1; x < N; ++x) {
				unsigned int vi = y*N + x;
				M += (state[vi] == 0) ? -1.0 : 1.0;
			}
		}
		M /= static_cast<double>(N*N);
		M = std::fabs(M);
		std::cout << K << " | " << (1.0 - std::exp(-2.0*K))
			<< " | " << avg_size << " | " << M
			<< " | " << cc_empirical
			<< std::endl;
	}
	std::cout << "---------+----------+---------+--------------------------"
		<< std::endl;
	std::cout << "Ising critical inv temperature is "
		<< K0 << std::endl;

	// Show a sample
	std::cout << std::endl;
	std::cout << "Sample at critical inv temp " << K0 << std::endl;
	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 1; x < N; ++x) {
			unsigned int vi = y*N + x;
			std::cout << (state_crit[vi] == 0 ? "." : "@");
		}
		std::cout << std::endl;
	}
}

