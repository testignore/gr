
#include <vector>
#include <iostream>
#include <ctime>

#include <boost/random.hpp>
#include <boost/timer.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "SimulatedAnnealingInference.h"
#include "LinearProgrammingMAPInference.h"

#define BOOST_TEST_MODULE(LPMAPTest)
#include <boost/test/unit_test.hpp>

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
	unsigned int N = 40;

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
	Grante::LinearProgrammingMAPInference lpmap(&fg);
	std::cout << "Minimizing energy..." << std::endl;
	std::vector<unsigned int> state;
	boost::timer lpmap_timer;
	double energy = lpmap.MinimizeEnergy(state);
	std::cout << "Energy " << energy << " in " << lpmap_timer.elapsed()
		<< "s" << std::endl;

	Grante::SimulatedAnnealingInference sainf(&fg);
	boost::timer sainf_timer;
	sainf.SetParameters(1000, 10.0, 0.005);
	double energy_sa = sainf.MinimizeEnergy(state);
	std::cout << "Energy SA " << energy_sa << " in " << sainf_timer.elapsed()
		<< "s" << std::endl;
}

