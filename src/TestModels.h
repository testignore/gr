
#ifndef GRANTE_TESTMODELS_H
#define GRANTE_TESTMODELS_H

#include "FactorGraphModel.h"
#include "FactorGraph.h"

namespace Grante {

class TestModels {
public:
	// Create a N-by-N 2D lattice Ising model with periodic boundary
	// conditions.
	//
	// The energies in the pairwise factors are [-K,K; K,-K].
	static void Ising2D(unsigned int N, double K,
		FactorGraphModel** out_model, FactorGraph** out_fg);
};

}

#endif

