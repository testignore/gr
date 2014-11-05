
#include <iostream>

#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "SimulatedAnnealingInference.h"
#include "DiffusionInference.h"

#define BOOST_TEST_MODULE(DiffusionTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

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
	Grante::DiffusionInference msdinf(&fg);
	std::vector<unsigned int> min_state(2);
	double min_energy = msdinf.MinimizeEnergy(min_state);

	BOOST_CHECK_CLOSE_ABS(0.4, min_energy, 1.0e-5);
	BOOST_CHECK(min_state[0] == 0 && min_state[1] == 0);
}

BOOST_AUTO_TEST_CASE(SimpleEnergyMinimizationHO)
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
	w.push_back(0.05);	// (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	card1.push_back(2);
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	w1.push_back(0.3);
	w1.push_back(-0.3);
	w1.push_back(0.2);
	w1.push_back(0.5);
	w1.push_back(0.0);
	w1.push_back(-0.5);
	Grante::FactorType* factortype1a = new Grante::FactorType("tripple", card1, w1);
	model.AddFactorType(factortype1a);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("tripple");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(3);
	var_index1[0] = 0;
	var_index1[1] = 1;
	var_index1[2] = 2;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);

	// Compute the forward map
	fg.ForwardMap();

	// Test inference results
	Grante::DiffusionInference msdinf(&fg);
	msdinf.SetParameters(true, 100, 1.0e-5);
	std::vector<unsigned int> min_state(3);
	double min_energy = msdinf.MinimizeEnergy(min_state);

	BOOST_CHECK_CLOSE_ABS(-0.45, min_energy, 1.0e-5);
	BOOST_CHECK(min_state[0] == 1 && min_state[1] == 1 && min_state[2] == 1);
}

BOOST_AUTO_TEST_CASE(SimpleGrid)
{
	// Randomly set the data observations
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
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
	unsigned int N = 32;

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

	// fg is now a N-by-N factor graph.  Decompose it
	fg.ForwardMap();
	Grante::DiffusionInference msdmap(&fg);
	std::cout << "Minimizing energy..." << std::endl;
	std::vector<unsigned int> state;
	boost::timer msdmap_timer;
	double energy = msdmap.MinimizeEnergy(state);
	std::cout << "Energy MSD " << energy << " in " << msdmap_timer.elapsed()
		<< "s" << std::endl;

	Grante::SimulatedAnnealingInference sainf(&fg);
	boost::timer sainf_timer;
	sainf.SetParameters(1000, 10.0, 0.005);
	double energy_sa = sainf.MinimizeEnergy(state);
	std::cout << "Energy SA " << energy_sa << " in " << sainf_timer.elapsed()
		<< "s" << std::endl;
}

BOOST_AUTO_TEST_CASE(SimpleSumProduct)
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
	Grante::DiffusionInference msdinf(&fg);
	msdinf.SetParameters(true, 10, 1.0e-6);
	msdinf.PerformInference();

	double log_z = msdinf.LogPartitionFunction();
	BOOST_CHECK_CLOSE_ABS(1.8367, log_z, 1.0e-2);

	const std::vector<double>& m_fac0 = msdinf.Marginal(0);
	BOOST_CHECK_CLOSE_ABS(0.3422, m_fac0[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.1939, m_fac0[1], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.2399, m_fac0[2], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.2240, m_fac0[3], 0.01);

	const std::vector<double>& m_fac1 = msdinf.Marginal(1);
	BOOST_CHECK_CLOSE_ABS(0.5821, m_fac1[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.4179, m_fac1[1], 0.01);

	const std::vector<double>& m_fac2 = msdinf.Marginal(2);
	BOOST_CHECK_CLOSE_ABS(0.5361, m_fac2[0], 0.01);
	BOOST_CHECK_CLOSE_ABS(0.4639, m_fac2[1], 0.01);
}

