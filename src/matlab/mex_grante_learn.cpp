
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
#include "AISInference.h"
#include "BruteForceExactInference.h"
#include "BeliefPropagation.h"
#include "FactorConditioningTable.h"
#include "StructuredMeanFieldInference.h"
#include "NaiveMeanFieldInference.h"
#include "LinearProgrammingMAPInference.h"
#include "DiffusionInference.h"
#include "LaplacePrior.h"
#include "NormalPrior.h"
#include "StudentTPrior.h"
#include "Prior.h"
#include "MaximumLikelihood.h"
#include "MaximumPseudolikelihood.h"
#include "MaximumCompositeLikelihood.h"
#include "MaximumCrissCrossLikelihood.h"
#include "NaivePiecewiseTraining.h"
#include "ContrastiveDivergenceTraining.h"
#include "StructuredPerceptron.h"
#include "StructuredSVM.h"
#include "NonlinearRBFFactorType.h"
#include "ExpectationMaximization.h"

#include "matlab_helpers.h"

static void InitializeFactorTypes(
	Grante::FactorGraphModel* model,
	const std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>&
		training_data)
{
	const std::vector<Grante::FactorType*>& ftypes = model->FactorTypes();
	for (unsigned int fti = 0; fti < ftypes.size(); ++fti) {
		Grante::NonlinearRBFFactorType* rbf =
			dynamic_cast<Grante::NonlinearRBFFactorType*>(ftypes[fti]);
		if (rbf != 0) {
			rbf->InitializeUsingTrainingData(training_data);
			mexPrintf("Initialized RBF network for factor type '%s'.\n",
				rbf->Name().c_str());
		}
	}
}

// [fg_model_trained] = grante_learn(model, factor_graphs,
//                                   observations, priors, method, options);
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
	MatlabCPPInitialize();
	if (nlhs != 1 || nrhs < 5 || nrhs > 6) {
		mexErrMsgTxt("Wrong number of arguments.\n");
		MatlabCPPExit();
		return;
	}

	// Option structure
	const mxArray* opt_s = 0;
	if (nrhs >= 6 && mxIsEmpty(prhs[5]) == false)
		opt_s = prhs[5];

	// Other rhs parameters
	if (mxIsStruct(prhs[1]) == false) {
		mexErrMsgTxt("Second parameter must be a structure.\n");
		MatlabCPPExit();
		return;
	}
	if (mxIsStruct(prhs[2]) == false) {
		mexErrMsgTxt("Third parameter must be a structure.\n");
		MatlabCPPExit();
		return;
	}
	if (mxIsEmpty(prhs[3]) == false && mxIsStruct(prhs[3]) == false) {
		mexErrMsgTxt("Fourth parameter must be empty or a structure.\n");
		MatlabCPPExit();
		return;
	}
	if (mxIsChar(prhs[4]) == false) {
		mexErrMsgTxt("Fifth parameter must be a string.\n");
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
	mexPrintf("Successfully read %d factor graphs.\n", num_fgs);

	// Parse learning method
	std::string method_name = GetMatlabString(prhs[4]);
	std::string mle_infer_method_name = GetStringDefaultOption(
		opt_s, "mle_infer_method", "treeinf");
	if (method_name == "mle" || method_name == "em") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			if (mle_infer_method_name == "treeinf" &&
				Grante::FactorGraphStructurizer::IsForestStructured(FG[fgi])
				== false) {
				mexErrMsgTxt("Maximum likelihood learning and "
					"Expectation Maximization are currently only "
					"possible for tree-structured factor graphs.\n");
				MatlabCPPExit();
				return;
			}
		}
	} else if (method_name == "mple") {
	} else if (method_name == "mcle") {
	} else if (method_name == "mxxle") {
		for (unsigned int fgi = 0; fgi < FG.size(); ++fgi) {
			if (Grante::FactorGraphStructurizer::IsOrderedPairwiseGridStructured(FG[fgi])
				== false) {
				mexErrMsgTxt("Maximum criss-cross likelihood learning "
					"is only possible for 4-neighborhood grid factor "
					"graphs.\n");
				MatlabCPPExit();
				return;
			}
		}
	} else if (method_name == "npw") {
	} else if (method_name == "cd_obs") {
	} else if (method_name == "cd") {
	} else if (method_name == "perceptron") {
	} else if (method_name == "avg_perceptron") {
	} else if (method_name == "ssvm") {
	} else {
		mexErrMsgTxt("Unknown learning method.  Use 'mle', 'mple', 'mcle', "
			"'mxxle', 'npw', 'cd_obs', 'cd', 'perceptron', 'avg_perceptron', "
			"'ssvm', or 'em'.\n");
		MatlabCPPExit();
		return;
	}

	if (mle_infer_method_name == "treeinf") {
	} else if (mle_infer_method_name == "bp") {
	} else if (mle_infer_method_name == "nmf") {
	} else if (mle_infer_method_name == "smf") {
	} else if (mle_infer_method_name == "ais") {
	} else if (mle_infer_method_name == "bfexact") {
	} else {
		mexErrMsgTxt("Unknown mle_infer_method.  Use 'bp', 'nmf', 'smf', "
			"'ais', of 'bfexact'.\n");
		MatlabCPPExit();
		return;
	}

	// Parse observations
	size_t num_obs = mxGetNumberOfElements(prhs[2]);
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::ExpectationMaximization::partially_labeled_instance_type>
		training_data_em;

	if (method_name == "em" || method_name == "cd") {
		training_data_em.reserve(num_obs);
	} else {
		training_data.reserve(num_obs);
	}
	std::vector<Grante::InferenceMethod*> inference_methods;
	if (method_name == "mle" || method_name == "mcle" || method_name == "mxxle")
		inference_methods.reserve(num_obs);

	Grante::FactorConditioningTable fcond_tab;
	for (unsigned int oi = 0; oi < num_obs; ++oi) {
		const mxArray* obs_id = mxGetField(prhs[2], oi, "id");
		if (obs_id == NULL) {
			mexErrMsgTxt("Each element in observations must contain a .id "
				"field.\n");
			MatlabCPPExit();
			return;
		}

		const mxArray* obs_labels = mxGetField(prhs[2], oi, "labels");
		const mxArray* obs_expectations =
			mxGetField(prhs[2], oi, "expectations");
		if (obs_labels == NULL && obs_expectations == NULL) {
			mexErrMsgTxt("Each element in observations must contain a .labels "
				"or .expectations field.\n");
			MatlabCPPExit();
			return;
		}

		unsigned int fg_idx = static_cast<unsigned int>(mxGetScalar(obs_id)) - 1;
		if (fg_idx >= FG.size()) {
			mexErrMsgTxt("Observation id must be no larger than factor graph id.\n");
			MatlabCPPExit();
			return;
		}
		if (obs_labels != NULL) {
			if (method_name == "em" || method_name == "cd") {
				std::vector<unsigned int> var_subset;
				std::vector<unsigned int> var_state;
				GetPartialMatlabVector(obs_labels, var_subset, var_state);
				for (unsigned int vsi = 0; vsi < var_state.size(); ++vsi) {
					var_state[vsi] -= 1;
					if (var_subset[vsi] >= FG[fg_idx]->Cardinalities().size()) {
						mexErrMsgTxt("Label observation index exceeds number "
							"of variables.\n");
						MatlabCPPExit();
						return;
					}
					if (var_state[vsi] >= FG[fg_idx]->Cardinalities()[var_subset[vsi]]) {
						mexErrMsgTxt("Label observation exceeds variable "
							"cardinality.\n");
						MatlabCPPExit();
						return;
					}
				}
				// Add partially observed labels
				training_data_em.push_back(
					Grante::ParameterEstimationMethod::partially_labeled_instance_type(
						FG[fg_idx], new Grante::FactorGraphPartialObservation(
							var_subset, var_state)));
			} else {
				// Label observations
				std::vector<unsigned int> labels_vec;
				GetMatlabVector(obs_labels, labels_vec);
				for (unsigned int vi = 0; vi < labels_vec.size(); ++vi) {
					labels_vec[vi] -= 1;	// Matlab index correction

					// Check labels are now in correct cardinality range
					if (labels_vec[vi] >= FG[fg_idx]->Cardinalities()[vi]) {
						mexErrMsgTxt("Label observation exceeds variable "
							"cardinality.\n");
						MatlabCPPExit();
						return;
					}
				}
				training_data.push_back(
					Grante::ParameterEstimationMethod::labeled_instance_type(
						FG[fg_idx], new Grante::FactorGraphObservation(labels_vec)));
			}
		} else {
			// EM does not support expectation observations yet
			if (method_name == "em" || method_name == "cd" ||
				method_name == "cd_obs") {
				mexErrMsgTxt("Method does not support expectation observations yet.\n");
				MatlabCPPExit();
				return;
			}

			// Expectation observations
			assert(obs_expectations != NULL);
			if (mxIsCell(obs_expectations) == false) {
				mexErrMsgTxt("Expectation observation must be a cell array.\n");
				MatlabCPPExit();
				return;
			}
			const Grante::FactorGraph* cfg = FG[fg_idx];
			if (cfg->Factors().size() !=
				mxGetNumberOfElements(obs_expectations))
			{
				mexErrMsgTxt("Expectation observation size does not match "
					"factor count.\n");
				MatlabCPPExit();
				return;
			}

			const mxArray** obs_exp_p = reinterpret_cast<const mxArray**>(
				mxGetPr(obs_expectations));
			assert(obs_exp_p != NULL);
			std::vector<std::vector<double> > obs_exp_v(cfg->Factors().size());
			for (unsigned int fi = 0; fi < cfg->Factors().size(); ++fi) {
				GetMatlabVector(obs_exp_p[fi], obs_exp_v[fi]);
				if (obs_exp_v[fi].size() !=
					cfg->Factors()[fi]->Type()->ProdCardinalities()) {
					mexErrMsgTxt("Expectation observation marginal size is "
						" incorrect.\n");
					MatlabCPPExit();
					return;
				}
			}
			training_data.push_back(
				Grante::ParameterEstimationMethod::labeled_instance_type(
					FG[fg_idx], new Grante::FactorGraphObservation(obs_exp_v)));
		}

		if (method_name == "mle" || method_name == "em") {
			Grante::InferenceMethod* inf = 0;
			if (mle_infer_method_name == "treeinf") {
				inf = new Grante::TreeInference(FG[fg_idx]);
			} else if (mle_infer_method_name == "smf") {
				Grante::StructuredMeanFieldInference* smf =
					new Grante::StructuredMeanFieldInference(FG[fg_idx],
						&fcond_tab);
				smf->SetParameters(
					GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
					GetScalarDefaultOption(opt_s, "smf_conv_tol", 1.0e-6),
					GetIntegerDefaultOption(opt_s, "smf_max_iter", 50));
				inf = smf;
			} else if (mle_infer_method_name == "nmf") {
				Grante::NaiveMeanFieldInference* nmf =
					new Grante::NaiveMeanFieldInference(FG[fg_idx]);
				nmf->SetParameters(
					GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
					GetScalarDefaultOption(opt_s, "nmf_conv_tol", 1.0e-6),
					GetIntegerDefaultOption(opt_s, "nmf_max_iter", 50));
				inf = nmf;
			} else if (mle_infer_method_name == "ais") {
				Grante::AISInference* ais =
					new Grante::AISInference(FG[fg_idx]);
				ais->SetSamplingParameters(
					GetIntegerDefaultOption(opt_s, "ais_k", 80),
					GetIntegerDefaultOption(opt_s, "ais_sweeps", 1),
					GetIntegerDefaultOption(opt_s, "ais_samples", 100));
				inf = ais;
			} else if (mle_infer_method_name == "bp") {
				Grante::BeliefPropagation* bpinf =
					new Grante::BeliefPropagation(FG[fg_idx]);
				bpinf->SetParameters(
					GetScalarDefaultOption(opt_s, "verbose", 0) > 0,
					GetIntegerDefaultOption(opt_s, "bp_max_iter", 100),
					GetScalarDefaultOption(opt_s, "bp_conv_tol", 1.0e-5));
				inf = bpinf;
			} else if (mle_infer_method_name == "bfexact") {
				inf = new Grante::BruteForceExactInference(FG[fg_idx]);
			}
			assert(inf != 0);
			inference_methods.push_back(inf);
		} else if (method_name == "mcle" || method_name == "mxxle"
			|| method_name == "npw") {
			// For MCLE we instantiate a fresh tree inference object.  The
			// object will be copied when instantiating subgraphs.
			Grante::TreeInference* tinf = new Grante::TreeInference(0);
			inference_methods.push_back(tinf);
		} else if (method_name == "cd_obs") {
			// nothing to do
		} else if (method_name == "cd") {
			// nothing to do
#if 0
		} else if (method_name == "em") {
			// For EM we instantiate a tree inference object.  The object will
			// be copied when instantiating subgraphs for expectation
			// computations.
			Grante::TreeInference* tinf = new Grante::TreeInference(FG[fg_idx]);
			inference_methods.push_back(tinf);
#endif
		} else if (method_name == "perceptron" ||
			method_name == "avg_perceptron" ||
			method_name == "ssvm")
		{
			if (Grante::FactorGraphStructurizer::IsForestStructured(FG[fg_idx])) {
				Grante::TreeInference* tinf =
					new Grante::TreeInference(FG[fg_idx]);
				inference_methods.push_back(tinf);
			} else {
#if 0
				Grante::LinearProgrammingMAPInference* lpinf =
					new Grante::LinearProgrammingMAPInference(FG[fg_idx],
						false);	// no verbosity
#endif
				Grante::DiffusionInference* msdinf =
					new Grante::DiffusionInference(FG[fg_idx]);
				msdinf->SetParameters(false, 20, 1.0e-5);
				inference_methods.push_back(msdinf);
			}
		}
	}
	mexPrintf("Successfully read %d observations.\n", num_obs);

	// Setup the parameter estimation method, initialize training data
	// pe_method: fully observed data parameter estimation method,
	// ppe_method: partially observed data parameter estimation method.
	Grante::ParameterEstimationMethod* pe_method = 0;
	Grante::ExpectationMaximization* ppe_method = 0;

	std::string opt_method_s = GetStringDefaultOption(opt_s, "opt_method", "lbfgs");
	Grante::MaximumLikelihood::MLEOptimizationMethod opt_method;
	if (opt_method_s == "lbfgs") {
		opt_method = Grante::MaximumLikelihood::LBFGSMethod;
	} else if (opt_method_s == "gradient") {
		opt_method = Grante::MaximumLikelihood::SimpleGradientMethod;
	} else if (opt_method_s == "bb") {
		opt_method = Grante::MaximumLikelihood::BarzilaiBorweinMethod;
	} else if (opt_method_s == "fista") {
		opt_method = Grante::MaximumLikelihood::FISTAMethod;
	}

	if (method_name == "mle") {
		Grante::MaximumLikelihood* mle_method =
			new Grante::MaximumLikelihood(&model);
		InitializeFactorTypes(&model, training_data);
		mle_method->SetupTrainingData(training_data, inference_methods);
		mle_method->SetOptimizationMethod(opt_method);
		pe_method = mle_method;
	} else if (method_name == "mple") {
		Grante::MaximumPseudolikelihood* mple_method =
			new Grante::MaximumPseudolikelihood(&model);
		InitializeFactorTypes(&model, training_data);
		mple_method->SetupTrainingData(training_data, inference_methods);
		mple_method->SetOptimizationMethod(opt_method);
		pe_method = mple_method;
	} else if (method_name == "mcle") {
		Grante::MaximumCompositeLikelihood* mcle_method =
			new Grante::MaximumCompositeLikelihood(&model,
				GetIntegerDefaultOption(opt_s, "mcle_cover", 0));
		InitializeFactorTypes(&model, training_data);
		mcle_method->SetupTrainingData(training_data, inference_methods);
		mcle_method->SetOptimizationMethod(opt_method);
		pe_method = mcle_method;
	} else if (method_name == "mxxle") {
		Grante::MaximumCrissCrossLikelihood* mxxle_method =
			new Grante::MaximumCrissCrossLikelihood(&model);
		InitializeFactorTypes(&model, training_data);
		mxxle_method->SetupTrainingData(training_data, inference_methods);
		mxxle_method->SetOptimizationMethod(opt_method);
		pe_method = mxxle_method;
	} else if (method_name == "npw") {
		Grante::NaivePiecewiseTraining* npw_method =
			new Grante::NaivePiecewiseTraining(&model);
		InitializeFactorTypes(&model, training_data);
		npw_method->SetupTrainingData(training_data, inference_methods);
		npw_method->SetOptimizationMethod(opt_method);
		pe_method = npw_method;
	} else if (method_name == "cd_obs" || method_name == "cd") {
		Grante::ContrastiveDivergenceTraining* cd_method =
			new Grante::ContrastiveDivergenceTraining(&model,
				GetIntegerDefaultOption(opt_s, "cd_k", 1),
				GetIntegerDefaultOption(opt_s, "cd_minibatchsize", 10),
				GetScalarDefaultOption(opt_s, "cd_stepsize", 1.0e-2));
		InitializeFactorTypes(&model, training_data);

		if (method_name == "cd_obs") {
			cd_method->SetupTrainingData(training_data, inference_methods);
		} else {
			cd_method->SetupPartiallyObservedTrainingData(training_data_em);
		}
		pe_method = cd_method;
	} else if (method_name == "perceptron") {
		Grante::StructuredPerceptron* ptron_method =
			new Grante::StructuredPerceptron(&model, false);
		InitializeFactorTypes(&model, training_data);
		ptron_method->SetupTrainingData(training_data, inference_methods);
		pe_method = ptron_method;
	} else if (method_name == "avg_perceptron") {
		Grante::StructuredPerceptron* ptron_method =
			new Grante::StructuredPerceptron(&model, true);	// averaging
		InitializeFactorTypes(&model, training_data);
		ptron_method->SetupTrainingData(training_data, inference_methods);
		pe_method = ptron_method;
	} else if (method_name == "ssvm") {
		Grante::StructuredSVM* ssvm_method = new Grante::StructuredSVM(&model,
			GetScalarDefaultOption(opt_s, "ssvm_c", 1.0), "bmrm");
		InitializeFactorTypes(&model, training_data);
		ssvm_method->SetupTrainingData(training_data, inference_methods);
		pe_method = ssvm_method;
	} else if (method_name == "em") {
		ppe_method = new Grante::ExpectationMaximization(
			&model, new Grante::MaximumLikelihood(&model));
		ppe_method->SetupTrainingData(training_data_em, inference_methods,
			inference_methods);
	} else {
		mexErrMsgTxt("Invalid training method.\n");
		MatlabCPPExit();
		return;
	}
	mexPrintf("Successfully initialized training method \"%s\".\n",
		method_name.c_str());

	// Parse priors
	size_t num_priors = mxGetNumberOfElements(prhs[3]);
	for (unsigned int pi = 0; pi < num_priors; ++pi) {
		// Obtain factortype this prior refers to
		const mxArray* f_type = mxGetField(prhs[3], pi, "factor_type");
		if (f_type == 0) {
			mexErrMsgTxt("Prior specified without factor_type field.\n");
			MatlabCPPExit();
			return;
		}
		const Grante::FactorType* cur_ft = NULL;
		if (mxIsChar(f_type)) {
			std::string fname = GetMatlabString(f_type);
			cur_ft = model.FindFactorType(fname);
			if (cur_ft == NULL) {
				mexErrMsgTxt("Invalid factor specification in priors: "
					"cannot find factor type with given name.\n");
				MatlabCPPExit();
				return;
			}
		} else {
			// Matlab index correction
			unsigned int fidx =
				static_cast<unsigned int>(mxGetScalar(f_type)) - 1;
			if (fidx >= model.FactorTypes().size()) {
				mexErrMsgTxt("Invalid factor specification in priors: "
					"factor type index is too large.\n");
				MatlabCPPExit();
				return;
			}
			cur_ft = model.FactorTypes()[fidx];
		}

		// Obtain prior type and parameters
		const mxArray* p_name = mxGetField(prhs[3], pi, "prior_name");
		if (p_name == 0) {
			mexErrMsgTxt("Invalid prior specification: prior_name field "
				"missing.\n");
			MatlabCPPExit();
			return;
		}
		std::string prior_name = GetMatlabString(p_name);

		const mxArray* p_opt = mxGetField(prhs[3], pi, "prior_opt");
		if (p_opt == 0) {
			mexErrMsgTxt("Invalid prior specification: prior_opt field "
				"missing.\n");
			MatlabCPPExit();
			return;
		}
		double* p_opt_p = mxGetPr(p_opt);

		Grante::Prior* cur_prior = 0;
		if (prior_name == "normal") {
			if (mxGetNumberOfElements(p_opt) != 1) {
				mexErrMsgTxt("Prior 'normal' requires exactly one "
					"parameter.\n");
				MatlabCPPExit();
				return;
			}
			cur_prior = new Grante::NormalPrior(p_opt_p[0],
				cur_ft->WeightDimension());
		} else if (prior_name == "laplace") {
			if (mxGetNumberOfElements(p_opt) != 1) {
				mexErrMsgTxt("Prior 'laplace' requires exactly one "
					"parameter.\n");
				MatlabCPPExit();
				return;
			}
			cur_prior = new Grante::LaplacePrior(p_opt_p[0],
				cur_ft->WeightDimension());
		} else if (prior_name == "studentt") {
			if (mxGetNumberOfElements(p_opt) != 2) {
				mexErrMsgTxt("Prior 'studentt' requires exactly two "
					"parameters [dof, sigma].\n");
				MatlabCPPExit();
				return;
			}
			cur_prior = new Grante::StudentTPrior(p_opt_p[0], p_opt_p[1],
				cur_ft->WeightDimension());
		} else {
			mexErrMsgTxt("Unknown prior distribution type.\n");
			MatlabCPPExit();
			return;
		}
#if 0
		mexPrintf("  Adding prior for factor '%s': %s.\n",
			cur_ft->Name().c_str(), prior_name.c_str());
#endif
		if (pe_method != 0) {
			pe_method->AddPrior(cur_ft->Name(), cur_prior);
		} else {
			assert(ppe_method != 0);
			ppe_method->AddPrior(cur_ft->Name(), cur_prior);
		}
	}
	mexPrintf("Successfully read %d priors.\n", num_priors);

	// Training
	assert(pe_method != 0 || ppe_method != 0);
	mexPrintf("[Grante] training using method: '%s'\n", method_name.c_str());
	if (pe_method != 0) {
		pe_method->Train(GetScalarDefaultOption(opt_s, "conv_tol", 1.0e-5),
			static_cast<unsigned int>(
				GetScalarDefaultOption(opt_s, "max_iter", 1000)));
		mexPrintf("[Grante] finished training.\n");
		delete (pe_method);	// parameters are in the FactorGraphModel
	} else {
		ppe_method->Train(GetScalarDefaultOption(opt_s, "conv_tol", 1.0e-5),
			GetScalarDefaultOption(opt_s, "max_iter", 20),
			GetScalarDefaultOption(opt_s, "em_subtol", 1.0e-6),
			static_cast<unsigned int>(
				GetScalarDefaultOption(opt_s, "em_subiter", 100)));
		mexPrintf("[Grante] finished EM training.\n");
		delete (ppe_method);
	}
	mexPrintf("[Grante] deleted parameter learning object.\n");

	// Create output model and change learned weights
	plhs[0] = mxDuplicateArray(prhs[0]);
	int fn_ftypes = mxGetFieldNumber(plhs[0], "factor_types");
	mxArray* model_ftypes = mxGetFieldByNumber(plhs[0], 0, fn_ftypes);
	size_t num_ftypes = mxGetNumberOfElements(model_ftypes);
	for (size_t fti = 0; fti < num_ftypes; ++fti) {
		const mxArray* ft_name = mxGetField(model_ftypes, fti, "name");
		if (ft_name == NULL) {
			mexErrMsgTxt("model.factor_types must contain a .name field.\n");
			MatlabCPPExit();
			return;
		}

		// Might be NULL
		mxArray* ft_weights = mxGetField(model_ftypes, fti, "weights");
		if (ft_weights == NULL)
			continue;
		double* ft_weights_p = mxGetPr(ft_weights);

		// Non-NULL -> copy learned weights
		Grante::FactorType* cur_ft = NULL;
		std::string fname = GetMatlabString(ft_name);
		cur_ft = model.FindFactorType(fname);

		Grante::NonlinearRBFFactorType* rbf_ft =
			dynamic_cast<Grante::NonlinearRBFFactorType*>(cur_ft);
		if (rbf_ft != 0) {
			// XXX: Handle dynamically initialized parameters
			mxDestroyArray(const_cast<mxArray*>(ft_weights));
			ft_weights = mxCreateNumericMatrix(1, rbf_ft->Weights().size(),
				mxDOUBLE_CLASS, mxREAL);
			mxSetField(model_ftypes, fti, "weights", ft_weights);
			ft_weights_p = mxGetPr(ft_weights);
		}
		std::copy(cur_ft->Weights().begin(), cur_ft->Weights().end(),
			ft_weights_p);
	}
	mexPrintf("[Grante] produced fg_trained.\n");

	// FIXME: delete training_data, training_data_em?

	for (unsigned int fgi = 0; fgi < FG.size(); ++fgi)
		delete (FG[fgi]);
	for (unsigned int ii = 0; ii < inference_methods.size(); ++ii)
		delete (inference_methods[ii]);
	MatlabCPPExit();
}

