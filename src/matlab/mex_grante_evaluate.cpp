
#include <vector>
#include <string>
#include <iostream>
#include <cassert>

#include <mex.h>

#include "Factor.h"
#include "FactorType.h"
#include "FactorGraph.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "InferenceMethod.h"

#include "matlab_helpers.h"

// [E] = grante_evaluate(model, factor_graph, states_or_marginals);
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	MatlabCPPInitialize();
	if (nrhs != 3 && nlhs != 1) {
		mexErrMsgTxt("Wrong number of arguments (must have nrhs=3, nlhs=1).\n");
		MatlabCPPExit();
		return;
	}

	// Master model
	Grante::FactorGraphModel model;
	if (matlab_parse_factorgraphmodel(prhs[0], model) == false) {
		MatlabCPPExit();
		return;
	}

	// Parse factor graph
	std::vector<Grante::FactorGraph*> FG;
	bool fgs_parsed = matlab_parse_factorgraphs(model, prhs[1], FG);
	if (fgs_parsed == false) {
		MatlabCPPExit();
		return;
	}

	size_t num_fgs = FG.size();
	if (num_fgs != 1) {
		mexErrMsgTxt("mex_grante_evaluate supports only one "
			"factor graph at a time.\n");
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
			delete (FG[fgi]);

		MatlabCPPExit();
		return;
	}
	// Compute energies
	Grante::FactorGraph* fg = FG[0];
	fg->ForwardMap();

	if (mxIsDouble(prhs[2])) {
		// states: (V,S) double, V nodes, S samples
		size_t V = mxGetM(prhs[2]);
		size_t S = mxGetN(prhs[2]);
		unsigned int var_count =
			static_cast<unsigned int>(fg->Cardinalities().size());
		const std::vector<unsigned int>& card = fg->Cardinalities();
		if (V != var_count) {
			mexErrMsgTxt("Number of model variables disagrees with "
				"number of rows in 'states' matrix.\n");
			delete fg;

			MatlabCPPExit();
			return;
		}
		double* states_p = mxGetPr(prhs[2]);

		// Create sample matrix
		plhs[0] = mxCreateNumericMatrix(1, S, mxDOUBLE_CLASS, mxREAL);
		double* E_p = mxGetPr(plhs[0]);

		// Evaluate energies
		std::vector<unsigned int> state(V);
		for (unsigned int si = 0; si < S; ++si) {
			for (unsigned int vi = 0; vi < V; ++vi) {
				// Index correction for Matlab
				state[vi] = static_cast<unsigned int>(states_p[si*V+vi]-1.0);

				// Check state is in bounds
				if (state[vi] >= card[vi]) {
					mexErrMsgTxt("State out of bounds.\n");
					delete fg;
					MatlabCPPExit();
					return;
				}
			}
			E_p[si] = fg->EvaluateEnergy(state);
		}
	} else if (mxIsCell(prhs[2])) {
		// marginals
		size_t marg_fac_count = mxGetNumberOfElements(prhs[2]);
		const std::vector<Grante::Factor*>& facs = fg->Factors();
		if (marg_fac_count != facs.size()) {
			mexErrMsgTxt("The given marginals must have as many elements as "
				"there are factors in the model.\n");
			delete fg;
			MatlabCPPExit();
			return;
		}

		double E_energy = 0.0;
		for (size_t fi = 0; fi < marg_fac_count; ++fi) {
			const mxArray* fac_marg_a = mxGetCell(prhs[2], fi);
			size_t melem_count = mxGetNumberOfElements(fac_marg_a);
			const Grante::FactorType* ftype = facs[fi]->Type();
			if (melem_count != ftype->ProdCardinalities()) {
				mexErrMsgTxt("The given marginal vectors do not have the correct size.\n");
				delete fg;
				MatlabCPPExit();
				return;
			}

			const double* fac_marg_p = mxGetPr(fac_marg_a);
			const std::vector<double>& E_fi = facs[fi]->Energies();
			for (size_t yi = 0; yi < melem_count; ++yi)
				E_energy += fac_marg_p[yi] * E_fi[yi];
		}
		plhs[0] = mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
		double* E_p = mxGetPr(plhs[0]);
		E_p[0] = E_energy;	// mean energy
	} else {
		mexErrMsgTxt("states_or_marginals must be a (V,sample_count) double "
			"array or a (1,F) cellarray.\n");
		delete fg;

		MatlabCPPExit();
		return;
	}

	delete (fg);
	MatlabCPPExit();
}


