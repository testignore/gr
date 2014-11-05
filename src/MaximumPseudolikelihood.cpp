
#include <iostream>
#include <algorithm>
#include <limits>
#include <cassert>

#include "FunctionMinimization.h"
#include "CompositeMinimization.h"
#include "MaximumPseudolikelihood.h"
#include "Pseudolikelihood.h"

namespace Grante {

MaximumPseudolikelihood::MaximumPseudolikelihood(FactorGraphModel* fg_model)
	: MaximumLikelihood(fg_model), opt_method(MaximumLikelihood::LBFGSMethod)
{
}

MaximumPseudolikelihood::~MaximumPseudolikelihood() {
}

MaximumPseudolikelihood::MPLEProblem*
MaximumPseudolikelihood::GetLearnProblem() {
	return (new MPLEProblem(this));
}

void MaximumPseudolikelihood::SetOptimizationMethod(
	MaximumLikelihood::MLEOptimizationMethod opt_method) {
	this->opt_method = opt_method;
}

double MaximumPseudolikelihood::Train(double conv_tol, unsigned int max_iter) {
	// Minimize the negative log-pseudolikelihood objective
	MPLEProblem mple_prob(this);
	std::vector<double> x_opt;

	FunctionMinimization::CheckDerivative(mple_prob, 1.0, 4, 1e-8, 7.5e-3);

	PrintProblemStatistics();
	double obj = std::numeric_limits<double>::signaling_NaN();
	switch (opt_method) {
	case (LBFGSMethod):
		obj = FunctionMinimization::LimitedMemoryBFGSMinimize(
			mple_prob, x_opt, conv_tol, max_iter, true, 200);
		break;
	case (SimpleGradientMethod):
		obj = FunctionMinimization::GradientMethodMinimize(mple_prob,
			x_opt, conv_tol, max_iter, true);
		break;
	case (BarzilaiBorweinMethod):
		obj = FunctionMinimization::BarzilaiBorweinMinimize(
			mple_prob, x_opt, conv_tol, max_iter, true);
		break;
	case (FISTAMethod):
		obj = CompositeMinimization::FISTAMinimize(
			mple_prob, x_opt, conv_tol, max_iter, true);
		break;
	default:
		assert(0);
		break;
	}

	// Save optimal weights into model
	mple_prob.LinearToFactorWeights(x_opt);

	return (obj);
}

MaximumPseudolikelihood::MPLEProblem::MPLEProblem(
	MaximumPseudolikelihood* mple_base)
	: MLEProblem(mple_base), mple_base(mple_base) {
	// Initialize Gibbs sampler for computing single-site conditional
	// distributions
	fgu.resize(mple_base->training_data.size());
	for (unsigned int n = 0; n < mple_base->training_data.size(); ++n)
		fgu[n] = new FactorGraphUtility(mple_base->training_data[n].first);
}

MaximumPseudolikelihood::MPLEProblem::~MPLEProblem() {
	for (unsigned int n = 0; n < fgu.size(); ++n)
		delete(fgu[n]);
}

double MaximumPseudolikelihood::MPLEProblem::EvaluateLikelihoodGradient(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) {

	Pseudolikelihood plh(mple_base->fg_model);

	// For each sample: run forward map, compute gradient
	double nll = 0.0;
	for (unsigned int n = 0; n < mple_base->training_data.size(); ++n) {
		// Get sample
		FactorGraph* ts_fg = mple_base->training_data[n].first;
		const FactorGraphObservation* ts_obs =
			mple_base->training_data[n].second;

		// Compute forward map: parameters (changed) to energies
		ts_fg->ForwardMap();

		// Compute log-pseudolikelihood and gradient
		nll += plh.ComputeNegLogPseudolikelihood(ts_fg, fgu[n], ts_obs,
			parameter_gradient);

		// Conserve memory by destroying unused energies
		ts_fg->EnergiesRelease();
	}
	return (nll);
}

}

