
#include <vector>
#include <iostream>

#include "FactorGraphStructurizer.h"

#define BOOST_TEST_MODULE(EulerianTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(Simple)
{
	unsigned int vertex_count = 6;
	std::vector<std::pair<unsigned int, unsigned int> > arcs;
	arcs.push_back(std::pair<unsigned int, unsigned int>(0, 1));	// A->B
	arcs.push_back(std::pair<unsigned int, unsigned int>(1, 0));	// B->A
	arcs.push_back(std::pair<unsigned int, unsigned int>(1, 2));	// B->C
	arcs.push_back(std::pair<unsigned int, unsigned int>(2, 2));	// C->C
	arcs.push_back(std::pair<unsigned int, unsigned int>(3, 1));	// D->B
	arcs.push_back(std::pair<unsigned int, unsigned int>(2, 3));	// C->D
	arcs.push_back(std::pair<unsigned int, unsigned int>(0, 3));	// A->D
	arcs.push_back(std::pair<unsigned int, unsigned int>(5, 0));	// F->A
	arcs.push_back(std::pair<unsigned int, unsigned int>(3, 4));	// D->E
	arcs.push_back(std::pair<unsigned int, unsigned int>(4, 5));	// E->F
	arcs.push_back(std::pair<unsigned int, unsigned int>(5, 4));	// F->E
	arcs.push_back(std::pair<unsigned int, unsigned int>(4, 5));	// E->F

	std::vector<unsigned int> trail;
	Grante::FactorGraphStructurizer::ComputeEulerianTrail(vertex_count,
		arcs, trail);
	unsigned int e_start = 0;
	unsigned int ei = e_start;
	unsigned int v_end = arcs[ei].first;
	std::vector<bool> traversed_edge(arcs.size(), false);
	do {
		// Check we traverse it exactly once
		BOOST_CHECK(traversed_edge[ei] == false);
		traversed_edge[ei] = true;

		// Check that it is a continuous path
		BOOST_CHECK(arcs[ei].first == v_end);
		v_end = arcs[ei].second;

		std::cout << "Edge " << ei << ": " << arcs[ei].first
			<< " -> " << arcs[ei].second << std::endl;
		ei = trail[ei];
	} while (ei != e_start);
}

