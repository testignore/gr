
#include <vector>

#include "FactorType.h"
#include "TestModels.h"

namespace Grante {

void TestModels::Ising2D(unsigned int N, double K,
	FactorGraphModel** out_model, FactorGraph** out_fg) {
	*out_model = new FactorGraphModel();
	FactorGraphModel* model = *out_model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> wp(4, 0.0);
	// Ising energies.  Here state 0 is "-1 spin" and 1 is "+1 spin".
	wp[0] = -K;
	wp[1] = K;
	wp[2] = K;
	wp[3] = -K;
	FactorType* factortype = new FactorType("ising", card, wp);
	model->AddFactorType(factortype);

	// Create a 2D N-by-N lattice model, with periodic boundary conditions
	std::vector<unsigned int> vc(N*N, 2);
	*out_fg = new FactorGraph(model, vc);
	FactorGraph* fg = *out_fg;

	// Add factors
	const FactorType* pt = model->FindFactorType("ising");
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	for (int y = 0; y < static_cast<int>(N); ++y) {
		for (int x = 0; x < static_cast<int>(N); ++x) {
			// Horizontal edge
			var_index[0] = (N*N + y*N + x - 1) % (N*N);
			var_index[1] = y*N + x;
			Factor* fac_h = new Factor(pt, var_index, data);
			fg->AddFactor(fac_h);

			// Vertical edge
			var_index[0] = (N*N + (y-1)*N + x) % (N*N);
			var_index[1] = y*N + x;
			Factor* fac_v = new Factor(pt, var_index, data);
			fg->AddFactor(fac_v);
		}
	}
	fg->ForwardMap();
}

}


