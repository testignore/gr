
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cassert>

#include <mex.h>

#include "Factor.h"
#include "FactorType.h"
#include "FactorGraph.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "VAcyclicDecomposition.h"

#include "matlab_helpers.h"

// [edge_is_removed] = mex_vac(vertex_count, edge_list);
//
// Input
//    vertec_count: double (1,1), >0 number of vertices in the graph.
//    edge_list: double (EN,2) edge list with indices >=1.
//
// Output
//    edge_is_removed: (1,EN) with elements 0 or 1.
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	MatlabCPPInitialize(false);
	if (nlhs != 1 || nrhs != 2) {
		mexErrMsgTxt("Wrong number of arguments.\n");
		MatlabCPPExit();
		return;
	}

	unsigned int vertex_count = static_cast<unsigned int>(mxGetScalar(prhs[0]));
	assert(vertex_count > 0);

	// Setup model
	Grante::FactorGraphModel model;
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w(4, 0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Setup factor graph
	std::vector<unsigned int> vc(vertex_count, 2);
	Grante::FactorGraph fg(&model, vc);

	const double* e_ptr = mxGetPr(prhs[1]);
	unsigned int edge_count = static_cast<unsigned int>(mxGetM(prhs[1]));
	assert(mxGetN(prhs[1]) == 2);
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	for (unsigned int ei = 0; ei < edge_count; ++ei) {
		std::vector<double> data;
		std::vector<unsigned int> var_index(2);
		var_index[0] = static_cast<unsigned int>(e_ptr[ei]) - 1;
		assert(var_index[0] < vertex_count);
		var_index[1] = static_cast<unsigned int>(e_ptr[ei+edge_count]) - 1;
		assert(var_index[1] < vertex_count);
		Grante::Factor* fac = new Grante::Factor(pt, var_index, data);
		fg.AddFactor(fac);
	}

	std::vector<bool> factor_is_removed(edge_count, false);
	std::vector<double> factor_weights(edge_count, 1.0);
	Grante::VAcyclicDecomposition vac(&fg);
	vac.ComputeDecompositionGreedy(factor_weights, factor_is_removed);

	plhs[0] = mxCreateNumericMatrix(1, edge_count, mxDOUBLE_CLASS, mxREAL);
	double* er_ptr = mxGetPr(plhs[0]);
	for (unsigned int ei = 0; ei < edge_count; ++ei)
		er_ptr[ei] = factor_is_removed[ei] ? 1 : 0;
}

