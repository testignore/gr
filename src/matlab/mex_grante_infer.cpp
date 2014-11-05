
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
#include "InferenceMethod.h"
#include "TreeInference.h"
#include "GibbsInference.h"
#include "MultichainGibbsInference.h"
#include "AISInference.h"
#include "SAMCInference.h"
#include "SwendsenWangInference.h"
#include "BruteForceExactInference.h"
#include "LinearProgrammingMAPInference.h"
#include "DiffusionInference.h"
#include "SimulatedAnnealingInference.h"
#include "BeliefPropagation.h"
#include "FactorConditioningTable.h"
#include "StructuredMeanFieldInference.h"
#include "NaiveMeanFieldInference.h"

#include "matlab_helpers.h"


// [result, z_result] = grante_infer(model, factor_graphs, method, options);
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	// Option structure
	const mxArray* opt_s = 0;
	if (nrhs >= 4 && mxIsEmpty(prhs[3]) == false)
		opt_s = prhs[3];

	MatlabCPPInitialize(GetScalarDefaultOption(opt_s, "verbose", 0) > 0);
	if (nlhs < 1 || nlhs > 2 || nrhs < 3 || nrhs > 4) {
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
	for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
		FG[fgi]->ForwardMap();

	// Parse learning method
	std::string method_name = GetMatlabString(prhs[2]);
	std::vector<Grante::InferenceMethod*> inf(num_fgs);
	bool energy_minimization = false;
	Grante::FactorConditioningTable fcond_tab;
	if (method_name == "treeinf") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			if (Grante::FactorGraphStructurizer::IsForestStructured(FG[fgi])
				== false) {
				mexErrMsgTxt("Exact inference is currently only "
					"possible for tree-structured factor graphs.\n");
				MatlabCPPExit();
				return;
			}
			inf[fgi] = new Grante::TreeInference(FG[fgi]);
		}
	} else if (method_name == "gibbs") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::GibbsInference* ginf = new Grante::GibbsInference(FG[fgi]);
			ginf->SetSamplingParameters(
				GetIntegerDefaultOption(opt_s, "gibbs_burnin", 100),
				GetIntegerDefaultOption(opt_s, "gibbs_spacing", 0),
				GetIntegerDefaultOption(opt_s, "gibbs_samples", 1000));
			inf[fgi] = ginf;
		}
	} else if (method_name == "mcgibbs") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::MultichainGibbsInference* mcginf =
				new Grante::MultichainGibbsInference(FG[fgi]);

			mcginf->SetSamplingParameters(
				GetIntegerDefaultOption(opt_s, "mcgibbs_chains", 5),
				GetScalarDefaultOption(opt_s, "mcgibbs_maxpsrf", 1.01),
				GetIntegerDefaultOption(opt_s, "mcgibbs_spacing", 0),
				GetIntegerDefaultOption(opt_s, "mcgibbs_samples", 1000));
			inf[fgi] = mcginf;
		}
	} else if (method_name == "spdiff") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::DiffusionInference* spdiffinf =
				new Grante::DiffusionInference(FG[fgi]);
			spdiffinf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetIntegerDefaultOption(opt_s, "spdiff_max_iter", 100),
				GetScalarDefaultOption(opt_s, "spdiff_conv_tol", 1.0e-5));
			inf[fgi] = spdiffinf;
		}
	} else if (method_name == "ais") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::AISInference* ais = new Grante::AISInference(FG[fgi]);
			ais->SetSamplingParameters(
				GetIntegerDefaultOption(opt_s, "ais_k", 80),
				GetIntegerDefaultOption(opt_s, "ais_sweeps", 1),
				GetIntegerDefaultOption(opt_s, "ais_samples", 100));
			inf[fgi] = ais;
		}
	} else if (method_name == "samc") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::SAMCInference* samc = new Grante::SAMCInference(FG[fgi]);
			samc->SetSamplingParameters(
				GetIntegerDefaultOption(opt_s, "samc_k", 20),
				GetScalarDefaultOption(opt_s, "samc_high_temp", 20.0),
				GetScalarDefaultOption(opt_s, "samc_swap", 0.5),
				GetIntegerDefaultOption(opt_s, "samc_burnin", 1000),
				GetIntegerDefaultOption(opt_s, "samc_samples", 1000));
			inf[fgi] = samc;
		}
#if 0
	// FIXME: disabled so far
	} else if (method_name == "sw") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::SwendsenWangInference* sw =
				new Grante::SwendsenWangInference(FG[fgi],
					GetScalarDefaultOption(opt_s, "sw_qetemp", 1.0));
			sw->SetSamplingParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetIntegerDefaultOption(opt_s, "sw_burnin", 50),
				GetIntegerDefaultOption(opt_s, "sw_sweeps", 1),
				GetIntegerDefaultOption(opt_s, "sw_samples", 100));
			inf[fgi] = sw;
		}
#endif
	} else if (method_name == "bfexact") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
			inf[fgi] = new Grante::BruteForceExactInference(FG[fgi]);
	} else if (method_name == "bp") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::BeliefPropagation* bpinf =
				new Grante::BeliefPropagation(FG[fgi]);
			bpinf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetIntegerDefaultOption(opt_s, "bp_max_iter", 100),
				GetScalarDefaultOption(opt_s, "bp_conv_tol", 1.0e-5));
			inf[fgi] = bpinf;
		}
	} else if (method_name == "nmf") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::NaiveMeanFieldInference* nmf =
				new Grante::NaiveMeanFieldInference(FG[fgi]);
			nmf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetScalarDefaultOption(opt_s, "nmf_conv_tol", 1.0e-6),
				GetIntegerDefaultOption(opt_s, "nmf_max_iter", 50));
			inf[fgi] = nmf;
		}
	} else if (method_name == "smf") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::StructuredMeanFieldInference* smf =
				new Grante::StructuredMeanFieldInference(FG[fgi],
					&fcond_tab);
			smf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetScalarDefaultOption(opt_s, "smf_conv_tol", 1.0e-6),
				GetIntegerDefaultOption(opt_s, "smf_max_iter", 50));
			inf[fgi] = smf;
		}
	} else if (method_name == "mapbp") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::BeliefPropagation* bpinf =
				new Grante::BeliefPropagation(FG[fgi]);
			bpinf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetIntegerDefaultOption(opt_s, "bp_max_iter", 100),
				GetScalarDefaultOption(opt_s, "bp_conv_tol", 1.0e-5));
			inf[fgi] = bpinf;
		}
		energy_minimization = true;
	} else if (method_name == "maplp") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::LinearProgrammingMAPInference* lpinf =
				new Grante::LinearProgrammingMAPInference(FG[fgi],
					GetScalarDefaultOption(opt_s, "verbose", 0) > 0);
			lpinf->SetParameters(
				GetIntegerDefaultOption(opt_s, "lp_max_iter", 100),
				GetScalarDefaultOption(opt_s, "lp_conv_tol", 1.0e-6));
			inf[fgi] = lpinf;
		}
		energy_minimization = true;
	} else if (method_name == "mapsa") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::SimulatedAnnealingInference* sainf =
				new Grante::SimulatedAnnealingInference(FG[fgi],
					GetScalarDefaultOption(opt_s, "verbose", 0) > 0);
			sainf->SetParameters(
				GetIntegerDefaultOption(opt_s, "sa_steps", 100),
				GetScalarDefaultOption(opt_s, "sa_t0", 10.0),
				GetScalarDefaultOption(opt_s, "sa_tfinal", 0.05));
			inf[fgi] = sainf;
		}
		energy_minimization = true;
	} else if (method_name == "mapmsd") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			Grante::DiffusionInference* msdinf =
				new Grante::DiffusionInference(FG[fgi]);
			msdinf->SetParameters(
				GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
				GetIntegerDefaultOption(opt_s, "msd_max_iter", 100),
				GetScalarDefaultOption(opt_s, "msd_conv_tol", 1.0e-5));
			inf[fgi] = msdinf;
		}
		energy_minimization = true;
	} else if (method_name == "mapbfexact") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
			inf[fgi] = new Grante::BruteForceExactInference(FG[fgi]);
		energy_minimization = true;
	} else {
		mexErrMsgTxt("Unknown inference method.  Use 'treeinf', 'bp', "
			"'mapbp', 'nmf', 'smf', 'maplp', 'mapsa', 'mapmsd', 'mapbfexact', "
			"'spdiff', 'gibbs', 'mcgibbs', 'ais', 'samc', 'sw', or 'bfexact'.\n");
		MatlabCPPExit();
		return;
	}

	// Perform inference
	mexPrintf("[Grante] performing inference using method: '%s'\n",
		method_name.c_str());
	// Create output model and change learned weights
	plhs[0] = mxDuplicateArray(prhs[1]);
	double* logz_res_p = NULL;
	if (nlhs >= 2) {
		plhs[1] = mxCreateNumericMatrix(1, static_cast<int>(FG.size()),
			mxDOUBLE_CLASS, mxREAL);
		logz_res_p = mxGetPr(plhs[1]);
	}

	if (energy_minimization) {
		// Energy minimization
		mxAddField(plhs[0], "solution");
		std::vector<std::vector<unsigned int> > sol_out(FG.size());
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
			sol_out.resize(FG[fgi]->Cardinalities().size());

		int FG_size = static_cast<int>(FG.size());
		#pragma omp parallel for schedule(dynamic)
		for (int fgi = 0; fgi < FG_size; ++fgi) {
			double sol_energy = inf[fgi]->MinimizeEnergy(sol_out[fgi]);
			#pragma omp critical
			{
				if (logz_res_p != NULL)
					logz_res_p[fgi] = sol_energy;

				delete (inf[fgi]);
				delete (FG[fgi]);
			}
		}

		// Convert into Matlab array
		for (int fgi = 0; fgi < FG_size; ++fgi) {
			mxArray* fsol = mxCreateNumericMatrix(1, sol_out[fgi].size(),
				mxDOUBLE_CLASS, mxREAL);
			double* fsol_p = mxGetPr(fsol);
			for (unsigned int si = 0; si < sol_out[fgi].size(); ++si)
				fsol_p[si] = sol_out[fgi][si] + 1;	// index correction
			mxSetField(plhs[0], fgi, "solution", fsol);
		}
	} else {
		// Probabilistic inference
		mxAddField(plhs[0], "marginals");
		int FG_size = static_cast<int>(FG.size());
		#pragma omp parallel for schedule(dynamic)
		for (int fgi = 0; fgi < FG_size; ++fgi) {
			inf[fgi]->PerformInference();

			// Store computed marginals
			mwSize dim_0 = static_cast<mwSize>(FG[fgi]->Factors().size());
			mxArray* fmarg = 0;
			#pragma omp critical
			{
				fmarg = mxCreateCellArray(1, &dim_0);
				assert(fmarg != NULL);
			}

			const std::vector<Grante::Factor*>& factors = FG[fgi]->Factors();
			for (unsigned int fac_idx = 0; fac_idx < factors.size(); ++fac_idx) {
				const std::vector<unsigned int>& card_dim =
					factors[fac_idx]->Cardinalities();
				mwSize ndim = card_dim.size();
				#pragma omp critical
				{
					mwSize* dims = new mwSize[ndim];
					std::copy(card_dim.begin(), card_dim.end(), dims);
					mxArray* fac_arr = mxCreateNumericArray(ndim, dims,
						mxDOUBLE_CLASS, mxREAL);
					delete (dims);
					double* fac_arr_p = mxGetPr(fac_arr);
					assert(fac_arr_p != NULL);

					const std::vector<double>& m_fac = inf[fgi]->Marginal(fac_idx);
					assert(m_fac.size() == mxGetNumberOfElements(fac_arr));
					std::copy(m_fac.begin(), m_fac.end(), fac_arr_p);

					mxSetCell(fmarg, fac_idx, fac_arr);
				}
			}

			// fg_result(fgi).marginals
			#pragma omp critical
			{
				mxSetField(plhs[0], fgi, "marginals", fmarg);
			}

			// Log-Partition function
			if (logz_res_p != NULL)
				logz_res_p[fgi] = inf[fgi]->LogPartitionFunction();

			delete (inf[fgi]);
			delete (FG[fgi]);
		}
	}
	MatlabCPPExit();
}

