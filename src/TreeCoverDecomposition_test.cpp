
#include <iostream>
#include <vector>
#include <cassert>

#include "FactorGraphModel.h"
#include "FactorGraph.h"
#include "Factor.h"
#include "TreeCoverDecomposition.h"

#define BOOST_TEST_MODULE(TreeCoverTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(SimpleGrid)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4);
	// Random uniform pairwise energies
	for (unsigned int di = 0; di < w.size(); ++di)
		w[di] = static_cast<double>(di) * 0.1;

	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a N-by-N grid-structured model
	unsigned int N = 4;

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc(N*N, 2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (unsigned int y = 0; y < N; ++y) {
		for (unsigned int x = 1; x < N; ++x) {
			// Horizontal edge
			var_index[0] = y*N + x - 1;
			var_index[1] = y*N + x;

			std::cout << fg.Factors().size() << ": ("
				<< y << "," << (x-1) << ") to ("
				<< y << "," << x << ")" << std::endl;
			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}
	for (unsigned int y = 1; y < N; ++y) {
		for (unsigned int x = 0; x < N; ++x) {
			// Vertical edge
			var_index[0] = (y-1)*N + x;
			var_index[1] = y*N + x;

			std::cout << fg.Factors().size() << ": ("
				<< (y-1) << "," << x << ") to ("
				<< y << "," << x << ")" << std::endl;
			Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
			fg.AddFactor(fac);
		}
	}

	// fg is now a N-by-N factor graph.  Decompose it
	Grante::TreeCoverDecomposition decomp(&fg);
	std::vector<std::vector<unsigned int> > tree_factor_indices;
	std::vector<unsigned int> factor_cover_count;
	decomp.ComputeDecompositionGreedy(tree_factor_indices, factor_cover_count);
	std::cout << "Decomposed into " << tree_factor_indices.size() << " trees."
		<< std::endl;
	unsigned int total_factors = 0;
	for (unsigned int ti = 0; ti < tree_factor_indices.size(); ++ti) {
		std::cout << "Tree " << ti << ":" << std::endl;
		for (unsigned int fi = 0; fi < tree_factor_indices[ti].size(); ++fi) {
			std::cout << tree_factor_indices[ti][fi] << " ";
			total_factors += 1;
		}
		std::cout << std::endl;
	}
	unsigned int total_cover = 0;
	for (unsigned int fi = 0; fi < factor_cover_count.size(); ++fi) {
		std::cout << "Factor " << fi << " covered "
			<< factor_cover_count[fi] << " times" << std::endl;
		total_cover += factor_cover_count[fi];
	}
	BOOST_CHECK(tree_factor_indices.size() == 2);
	BOOST_CHECK(total_cover == total_factors);
}

