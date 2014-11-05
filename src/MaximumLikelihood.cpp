
#include <algorithm>
#include <limits>
#include <tr1/unordered_map>
#include <cassert>

#include <boost/lambda/lambda.hpp>

#include "MaximumLikelihood.h"
#include "Likelihood.h"
#include "FunctionMinimization.h"
#include "CompositeMinimization.h"

using namespace boost::lambda;

namespace Grante {

MaximumLikelihood::MaximumLikelihood(FactorGraphModel* fg_model)
	: ParameterEstimationMethod(fg_model), opt_method(LBFGSMethod) {
}

MaximumLikelihood::~MaximumLikelihood() {
}

void MaximumLikelihood::SetOptimizationMethod(
	MLEOptimizationMethod opt_method) {
	this->opt_method = opt_method;
}

MaximumLikelihood::MLEProblem* MaximumLikelihood::GetLearnProblem() {
	return (new MLEProblem(this));
}

double MaximumLikelihood::Train(double conv_tol, unsigned int max_iter) {
	// Minimize the 1/N negative log-likelihood objective
	MLEProblem mle_prob(this);
	std::vector<double> x_opt;

	FunctionMinimization::CheckDerivative(mle_prob, 1.0e-2, 4, 1.0e-8, 1.0e-2);

	// Use rank-200 approximation to inverse Hessian.  Contrary to popular
	// belief ("ten to twenty vectors are enough for L-BFGS"), for likelihood
	// objectives of discrete models this really makes a big difference
	// because evaluating the objective is rather expensive.
	PrintProblemStatistics();
	double obj = std::numeric_limits<double>::signaling_NaN();
	switch (opt_method) {
	case (LBFGSMethod):
		obj = FunctionMinimization::LimitedMemoryBFGSMinimize(
			mle_prob, x_opt, conv_tol, max_iter, true, 200);
		break;
	case (SimpleGradientMethod):
		obj = FunctionMinimization::GradientMethodMinimize(mle_prob,
			x_opt, conv_tol, max_iter, true);
		break;
	case (BarzilaiBorweinMethod):
		obj = FunctionMinimization::BarzilaiBorweinMinimize(
			mle_prob, x_opt, conv_tol, max_iter, true);
		break;
	case (FISTAMethod):
		obj = CompositeMinimization::FISTAMinimize(
			mle_prob, x_opt, conv_tol, max_iter, true);
		break;
	default:
		assert(0);
		break;
	}

	// Save optimal weights into model
	mle_prob.LinearToFactorWeights(x_opt);

	return (obj);
}

// Function minimization part
MaximumLikelihood::MLEProblem::MLEProblem(MaximumLikelihood* mle_base)
	: mle_base(mle_base) {
	// Compute dimension once and setup factor type orders
	dim = 0;
	const std::vector<FactorType*>& factor_types =
		mle_base->fg_model->FactorTypes();
	for (std::vector<FactorType*>::const_iterator fti = factor_types.begin();
		fti != factor_types.end(); ++fti) {
		dim += (*fti)->WeightDimension();
		parameter_order.push_back((*fti)->Name());
	}
}

MaximumLikelihood::MLEProblem::~MLEProblem() {
}

double MaximumLikelihood::MLEProblem::EvalF(const std::vector<double>& x,
	std::vector<double>& grad) {
	assert(x.size() == dim);
	assert(grad.empty() || grad.size() == dim);

	// 1. Convert x into factor parameters
	LinearToFactorWeights(x);

	// 2. Setup parameter gradient
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	SetupParameterGradient(parameter_gradient);

	// 3. Compute likelihood related parameter gradient:
	//    \sum_{n=1}^N \nabla_w -log p(x_n;w)
	double nll = EvaluateLikelihoodGradient(parameter_gradient);

	// Scale by 1/N to have: 1/N (\sum_{n=1}^N \nabla_w - log p(x_n;w))
	double scale = 1.0 / static_cast<double>(mle_base->training_data.size());
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		std::transform(parameter_gradient[*ft_name].begin(),
			parameter_gradient[*ft_name].end(),
			parameter_gradient[*ft_name].begin(), scale * _1);
	}
	nll *= scale;

	// 4. Convert gradient into linear form
	if (grad.empty() == false)
		AddParameterGradient(parameter_gradient, grad);

	return (nll);
}

double MaximumLikelihood::MLEProblem::EvalG(const std::vector<double>& x,
	std::vector<double>& subgrad) {
	assert(x.size() == dim);
	assert(subgrad.empty() || subgrad.size() == dim);
	LinearToFactorWeights(x);

	// Scaling factor
	double scale = 1.0 / static_cast<double>(mle_base->training_data.size());

	// 1. Setup parameter gradient
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	SetupParameterGradient(parameter_gradient);

	// 2. Add regularizer
	double nll = 0.0;
	for (std::multimap<std::string, Prior*>::const_iterator
		prior = mle_base->priors.begin();
		prior != mle_base->priors.end(); ++prior)
	{
		FactorType* ft = mle_base->fg_model->FindFactorType(prior->first);
		nll += prior->second->EvaluateNegLogP(ft->Weights(),
			parameter_gradient[prior->first], scale);
	}

	// 2. Convert gradient into linear form
	if (subgrad.empty() == false)
		AddParameterGradient(parameter_gradient, subgrad);

	return (nll);
}

void MaximumLikelihood::MLEProblem::EvalGProximalOperator(
	const std::vector<double>& u, double L,
	std::vector<double>& wprox) const {
	assert(u.size() == dim);
	assert(wprox.size() == dim);

	// Hot-path: no prior, simply copy u
	wprox = u;
	if (mle_base->priors.empty())
		return;

	// Compute scaling factor used to adjust L for all proximal problems
	double scale = 1.0 / static_cast<double>(mle_base->training_data.size());
	L /= scale;

	// Setup vectors of appropriate size
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	SetupParameterGradient(parameter_gradient);

	// Proximal problem decomposes over priors, solve it separately
	size_t base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Find prior
		std::multimap<std::string, Prior*>::const_iterator pri =
			mle_base->priors.find(*ft_name);
		if (pri == mle_base->priors.end())
			continue;	// skip, no prior for this type

		// Obtain suitably sized vector
		std::tr1::unordered_map<std::string,
			std::vector<double> >::const_iterator pgi =
				parameter_gradient.find(*ft_name);
		assert(pgi != parameter_gradient.end());

		// Create a copy of the subvector of u
		std::vector<double> u_pgi(u.begin() + base_idx,
			u.begin() + base_idx + pgi->second.size());
		std::vector<double> wprox_pgi(u_pgi.size(), 0.0);

		pri->second->EvaluateProximalOperator(u_pgi, L, wprox_pgi);
		std::copy(wprox_pgi.begin(), wprox_pgi.end(),
			wprox.begin() + base_idx);
		base_idx += pgi->second.size();
	}
}

double MaximumLikelihood::MLEProblem::EvaluateLikelihoodGradient(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) {
	// For each sample: run forward map, run inference, compute gradient
	Likelihood lh(mle_base->fg_model);
	double nll = 0.0;
	int training_sample_count =
		static_cast<int>(mle_base->training_data.size());

	#pragma omp parallel for schedule(dynamic)
	for (int n = 0; n < training_sample_count; ++n) {
		// Get sample
		FactorGraph* ts_fg = mle_base->training_data[n].first;
		const FactorGraphObservation* ts_obs =
			mle_base->training_data[n].second;

		// Compute forward map: parameters (changed) to energies
		ts_fg->ForwardMap();

		// Compute marginals
		InferenceMethod* ts_inf = mle_base->inference_methods[n];
		ts_inf->ClearInferenceResult();
		ts_inf->PerformInference();
		const std::vector<std::vector<double> >& marginals =
			ts_inf->Marginals();
		double log_z = ts_inf->LogPartitionFunction();

		#pragma omp critical
		{
			// Compute likelihood and gradient
			nll += lh.ComputeNegLogLikelihood(ts_fg, ts_obs, marginals,
				log_z, parameter_gradient);
		}

		// Get rid of marginals
		ts_inf->ClearInferenceResult();
		ts_fg->EnergiesRelease();
	}
	return (nll);
}

void MaximumLikelihood::MLEProblem::SetupParameterGradient(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const {
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		FactorType* ft = mle_base->fg_model->FindFactorType(*ft_name);

		// Initialize to zero
		parameter_gradient[*ft_name] =
			std::vector<double>(ft->WeightDimension(), 0.0);
	}
}

void MaximumLikelihood::MLEProblem::AddParameterGradient(
	const std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, std::vector<double>& grad) const {
	size_t base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		std::tr1::unordered_map<std::string,
			std::vector<double> >::const_iterator pgi =
				parameter_gradient.find(*ft_name);
		assert(pgi != parameter_gradient.end());
		std::transform(pgi->second.begin(), pgi->second.end(),
			grad.begin() + base_idx, grad.begin() + base_idx,
			[](double pge, double ge) -> double { return (pge + ge); });
		base_idx += pgi->second.size();
	}
}

void MaximumLikelihood::MLEProblem::LinearToFactorWeights(
	const std::vector<double>& x) {
	unsigned int base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = mle_base->fg_model->FindFactorType(*ft_name);
		unsigned int ft_w_len = ft->WeightDimension();
		std::copy(x.begin() + base_idx, x.begin() + base_idx + ft_w_len,
			ft->Weights().begin());
		base_idx += ft_w_len;
	}
	assert(base_idx == x.size());
}

unsigned int MaximumLikelihood::MLEProblem::Dimensions() const {
	return (dim);
}

void MaximumLikelihood::MLEProblem::ProvideStartingPoint(
	std::vector<double>& x0) const {
	assert(x0.size() == dim);

	// Initial parameters are user-provided
	unsigned int base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = mle_base->fg_model->FindFactorType(*ft_name);
		unsigned int ft_w_len = ft->WeightDimension();
		std::copy(ft->Weights().begin(), ft->Weights().end(),
			x0.begin() + base_idx);
		base_idx += ft_w_len;
	}
	assert(base_idx == x0.size());
}

}

