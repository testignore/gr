
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <functional>
#include <cassert>

#include <mex.h>

#include "Factor.h"
#include "FactorType.h"
#include "FactorGraph.h"
#include "FactorGraphModel.h"
#include "FactorGraphStructurizer.h"
#include "InferenceMethod.h"
#include "TreeInference.h"
#include "GibbsInference.h"
#include "MultichainGibbsInference.h"

#include "matlab_helpers.h"

// [states] = grante_sample(model, fg, method, sample_count, options);
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	// Option structure
	const mxArray* opt_s = 0;
	if (nrhs >= 4 && mxIsEmpty(prhs[4]) == false)
		opt_s = prhs[4];

	MatlabCPPInitialize(GetScalarDefaultOption(opt_s, "verbose", 0) > 0);

	if (nrhs < 4 || nrhs > 5 || nlhs != 1) {
		mexErrMsgTxt("Wrong number of arguments.\n");
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
		mexErrMsgTxt("mex_grante_sample supports only one "
			"factor graph at a time.\n");
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
			delete (FG[fgi]);

		MatlabCPPExit();
		return;
	}
	// Compute energies
	Grante::FactorGraph* fg = FG[0];
	fg->ForwardMap();

	// Parse sampling method
	std::string method_name = GetMatlabString(prhs[2]);
	Grante::InferenceMethod* inf = 0;
	if (method_name == "treeinf") {
		if (Grante::FactorGraphStructurizer::IsForestStructured(fg) == false) {
			mexErrMsgTxt("Exact sampling is currently only "
				"possible for tree-structured factor graphs.\n");
			MatlabCPPExit();
			return;
		}
		inf = new Grante::TreeInference(fg);
	} else if (method_name == "gibbs") {
		Grante::GibbsInference* ginf = new Grante::GibbsInference(fg);
		ginf->SetSamplingParameters(
			GetIntegerDefaultOption(opt_s, "gibbs_burnin", 100),
			GetIntegerDefaultOption(opt_s, "gibbs_spacing", 0),
			GetIntegerDefaultOption(opt_s, "gibbs_samples", 1));
		inf = ginf;
	} else if (method_name == "mcgibbs") {
		Grante::MultichainGibbsInference* mcginf =
			new Grante::MultichainGibbsInference(fg);

		mcginf->SetSamplingParameters(
			GetIntegerDefaultOption(opt_s, "mcgibbs_chains", 5),
			GetScalarDefaultOption(opt_s, "mcgibbs_maxpsrf", 1.01),
			GetIntegerDefaultOption(opt_s, "mcgibbs_spacing", 0),
			GetIntegerDefaultOption(opt_s, "mcgibbs_samples", 1000));
		inf = mcginf;
	} else {
		mexErrMsgTxt("Unknown sampling method.  Use 'treeinf', "
			"'gibbs', or 'mcgibbs'.\n");
		MatlabCPPExit();
		return;
	}

	// Parse sample_count
	if (mxIsDouble(prhs[3]) == false || mxGetNumberOfElements(prhs[3]) != 1) {
		mexErrMsgTxt("sample_count must be a (1,1) double array.\n");
		MatlabCPPExit();
		return;
	}
	unsigned int sample_count = mxGetScalar(prhs[3]);
	assert(sample_count > 0);

	// Perform inference
	mexPrintf("[Grante] performing sampling using method: '%s'\n",
		method_name.c_str());

	unsigned int var_count = fg->Cardinalities().size();
	plhs[0] = mxCreateNumericMatrix(var_count, sample_count,
		mxDOUBLE_CLASS, mxREAL);
	double* sample_p = mxGetPr(plhs[0]);

	// Sample
	std::vector<std::vector<unsigned int> > states;
	inf->Sample(states, sample_count);
	assert(states.size() == sample_count);
	for (unsigned int si = 0; si < states.size(); ++si) {
		// Add 1 for Matlab indexing
		std::transform(states[si].begin(), states[si].end(),
			&sample_p[si * var_count],
			std::bind2nd(std::plus<double>(), 1.0));
	}
	delete (inf);
	delete (fg);
	MatlabCPPExit();
}

