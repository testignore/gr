
#include <algorithm>
#include <functional>
#include <numeric>
#include <limits>
#include <iostream>
#include <cmath>
#include <cassert>

#include <boost/timer.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "Likelihood.h"
#include "StructuredSVM.h"
#include "FunctionMinimization.h"
#include "StochasticFunctionMinimization.h"
#include "StructuredHammingLoss.h"

using namespace boost::lambda;

namespace Grante {

StructuredSVM::StructuredSVM(FactorGraphModel* fg_model, double ssvm_C,
	const std::string& opt_method)
	: ParameterEstimationMethod(fg_model), ssvm_C(ssvm_C),
		opt_method(opt_method) {
	assert(opt_method == "stochastic" || opt_method == "bmrm");
}

StructuredSVM::~StructuredSVM() {
	for (unsigned int tn = 0; tn < t_loss.size(); ++tn)
		delete (t_loss[tn]);
}

void StructuredSVM::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	t_instances.clear();
	t_loss.clear();
	t_instances.reserve(training_data.size());
	t_loss.reserve(training_data.size());
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		t_instances.push_back(training_data[n].first);
		t_loss.push_back(new StructuredHammingLoss(training_data[n].second));
	}
	SetupTrainingData(t_instances, t_loss, inference_methods);
	t_instances.clear();	// no longer needed
}

void StructuredSVM::SetupTrainingData(
	const std::vector<FactorGraph*>& instances,
	const std::vector<StructuredLossFunction*>& loss,
	const std::vector<InferenceMethod*>& inference_methods) {
	training_instances = instances;
	loss_functions = loss;
	this->inference_methods = inference_methods;
}

double StructuredSVM::Train(double conv_tol, unsigned int max_iter) {
	StructuredSVMProblem ssvm_prob(this);
	std::vector<double> x_opt;

	// Stochastic or BMRM training
	double obj = std::numeric_limits<double>::signaling_NaN();
	if (opt_method == "stochastic") {
		StochasticStructuredSVMProblem ssvm_sprob(&ssvm_prob);
		obj = StochasticFunctionMinimization::StochasticSubgradientMethodMinimize(
			ssvm_sprob, x_opt, conv_tol, max_iter, true);
	} else if (opt_method == "bmrm") {
		obj = TrainBMRM(&ssvm_prob, conv_tol, max_iter);
		x_opt.resize(ssvm_prob.Dimensions());
		std::fill(x_opt.begin(), x_opt.end(), 0.0);
		ssvm_prob.FactorWeightsToLinear(x_opt);
	} else {
		assert(0);
	}

	// Save optimal weights into model
	ssvm_prob.LinearToFactorWeights(x_opt);

	return (obj);
}

double StructuredSVM::TrainBMRM(StructuredSVMProblem* ssvm_prob,
	double conv_tol, unsigned int max_iter) {
	BMRM2StructuredSVMProblem bmrm(ssvm_prob);

	// Best (lowest) true feasible objective
	double J_best = std::numeric_limits<double>::infinity();
	double gap = std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; max_iter == 0 || iter <= max_iter; ++iter) {
		std::cout << "BMRM iter " << iter << ", J_best " << J_best
			<< ", gap " << gap << std::endl;

		// Compute: R_emp, subgradient of R_emp
		ssvm_prob->ClearParameterGradient();
		std::cout << "   Doing loss-augmented MAP inference." << std::endl;
		boost::timer inf_timer;
		double R_emp = ssvm_prob->EvaluateLossGradient();
		std::cout << "   " << inf_timer.elapsed()
			<< "s for loss-augmented MAP inference" << std::endl;

		// Enlarge cutting plane model
		bmrm.AddCurrentSubgradient(R_emp);

		// Compute exact objective and keep track of best feasible solution
		double J_t_exact = R_emp + ssvm_prob->AddRegularizer();
		if (J_t_exact < J_best)
			J_best = J_t_exact;

		// Minimize cutting plane model
		std::vector<double> w_opt;
		double pd_gap;
		double eg_conv_tol = 0.1*gap;
		std::vector<double> alpha_opt;
		boost::timer bmrm_sub_timer;
		double J_t_lowerbound = bmrm.OptimizeDual(w_opt, alpha_opt,
			eg_conv_tol, 20000, pd_gap, false);
		if (pd_gap > eg_conv_tol) {
			std::cout << "### WARNING: required BMRM subproblem accuracy "
				<< "not reached." << std::endl;
		}
		std::cout << "   " << bmrm_sub_timer.elapsed()
			<< "s for solving the BMRM sub problem" << std::endl;

		// Compute model gap
		// TODO: relax this condition using pd_gap
		if ((J_best + conv_tol) < J_t_lowerbound) {
			std::cout << "### WARNING: J_best " << J_best << " < "
				<< J_t_lowerbound << " J_t_lowerbound, when using exact "
				<< "inference this should not happen."
				<< std::endl;
		}
		std::cout << "J_best " << J_best << ", "
				<< "J_t_lowerbound " << J_t_lowerbound
				<< ", subproblem pd_gap " << pd_gap << std::endl;
		assert((J_best + conv_tol) >= J_t_lowerbound);
		gap = J_best - J_t_lowerbound;
		if (gap < conv_tol) {
			std::cout << "   * Converged with tolerance " << gap << " < "
				<< conv_tol << std::endl;
			break;
		}
	}
	return (J_best);
}

StructuredSVM::StructuredSVMProblem::StructuredSVMProblem(
	StructuredSVM* ssvm_base)
	: ssvm_base(ssvm_base) {
	// Compute dimension once and setup factor type orders
	dim = 0;
	const std::vector<FactorType*>& factor_types =
		ssvm_base->fg_model->FactorTypes();
	for (std::vector<FactorType*>::const_iterator fti = factor_types.begin();
		fti != factor_types.end(); ++fti) {
		dim += (*fti)->WeightDimension();
		parameter_order.push_back((*fti)->Name());
	}
}

StructuredSVM::StructuredSVMProblem::~StructuredSVMProblem() {
}

double StructuredSVM::StructuredSVMProblem::EvaluateLossGradient() {
	Likelihood lh(ssvm_base->fg_model);
	double obj = 0.0;
	int sample_count = static_cast<int>(ssvm_base->training_instances.size());
	#pragma omp parallel for schedule(dynamic)
	for (int n = 0; n < sample_count; ++n) {
		double obj_n = EvaluateLossGradient(lh, static_cast<unsigned int>(n));
		#pragma omp critical
		{
			obj += obj_n;
		}
	}

	return (obj);
}

double StructuredSVM::StructuredSVMProblem::EvaluateLossGradient(
	Likelihood& lh, unsigned int n) {
	// For each sample: run forward map, loss-augmentation, run MAP inference,
	// compute gradient
	double obj = 0.0;
	double reg_scale = ssvm_base->ssvm_C /
		static_cast<double>(ssvm_base->training_instances.size());

	// Get sample
	FactorGraph* ts_fg = ssvm_base->training_instances[n];
	const StructuredLossFunction* h_loss =
		ssvm_base->loss_functions[n];
	InferenceMethod* ts_inf = ssvm_base->inference_methods[n];

	// Compute forward map: parameters (changed) to energies
	ts_fg->ForwardMap();

	// Compute: -E(y_n;x_n,w)
	#pragma omp critical
	{
		double obj_e = lh.ComputeObservationEnergy(ts_fg,
			h_loss->Truth(), parameter_gradient, reg_scale);
		obj += obj_e;
	}

	// Compute: min_y [E(y;x_n,w)-Delta(y,y_n)]
	//   i) Find loss-augmented MAP state (y_star)
	std::vector<unsigned int> y_star;
	h_loss->PerformLossAugmentation(ts_fg, -1.0);	// +Delta(.,y_n)
	double obj_m_1 = -reg_scale * ts_inf->MinimizeEnergy(y_star);

	//  ii) Compute gradient (already of negative sign)
	#pragma omp critical
	{
		double obj_m = lh.ComputeObservationEnergy(ts_fg, y_star,
			parameter_gradient, -reg_scale);
		assert(std::fabs(obj_m_1 - obj_m) < 1e-8);
		obj += obj_m;
	}

	ts_inf->ClearInferenceResult();

	return (obj);
}

double StructuredSVM::StructuredSVMProblem::AddRegularizer(double scale) {
	double obj = 0.0;
	for (std::multimap<std::string, Prior*>::const_iterator
		prior = ssvm_base->priors.begin();
		prior != ssvm_base->priors.end(); ++prior)
	{
		FactorType* ft = ssvm_base->fg_model->FindFactorType(prior->first);
		obj += prior->second->EvaluateNegLogP(ft->Weights(),
			parameter_gradient[prior->first], scale);
	}
	return (obj);
}

double StructuredSVM::StructuredSVMProblem::EvaluateFenchelDual(void) {
	double obj = 0.0;
	for (std::multimap<std::string, Prior*>::const_iterator
		prior = ssvm_base->priors.begin();
		prior != ssvm_base->priors.end(); ++prior)
	{
		FactorType* ft = ssvm_base->fg_model->FindFactorType(prior->first);
		obj += prior->second->EvaluateFenchelDual(ft->Weights(),
			parameter_gradient[prior->first]);
	}
	return (obj);
}

void StructuredSVM::StructuredSVMProblem::ClearParameterGradient() {
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		FactorType* ft = ssvm_base->fg_model->FindFactorType(*ft_name);
		parameter_gradient[*ft_name] =
			std::vector<double>(ft->WeightDimension(), 0.0);
	}
}

void StructuredSVM::StructuredSVMProblem::LinearToFactorWeights(
	const std::vector<double>& x) {
	unsigned int base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = ssvm_base->fg_model->FindFactorType(*ft_name);
		unsigned int ft_w_len = ft->WeightDimension();
		std::copy(x.begin() + base_idx, x.begin() + base_idx + ft_w_len,
			ft->Weights().begin());
		base_idx += ft_w_len;
	}
	assert(base_idx == x.size());
}

void StructuredSVM::StructuredSVMProblem::FactorWeightsToLinear(
	std::vector<double>& x) {
	unsigned int base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = ssvm_base->fg_model->FindFactorType(*ft_name);
		unsigned int ft_w_len = ft->WeightDimension();
		std::copy(ft->Weights().begin(), ft->Weights().end(),
			x.begin() + base_idx);
		base_idx += ft_w_len;
	}
	assert(base_idx == x.size());
}

void StructuredSVM::StructuredSVMProblem::ParameterGradientToLinear(
	std::vector<double>& grad) {
	size_t base_idx = 0;
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		std::copy(parameter_gradient[*ft_name].begin(),
			parameter_gradient[*ft_name].end(), grad.begin() + base_idx);
		base_idx += parameter_gradient[*ft_name].size();
	}
}

unsigned int StructuredSVM::StructuredSVMProblem::Dimensions() const {
	return (dim);
}

StructuredSVM* StructuredSVM::StructuredSVMProblem::Base() {
	return (ssvm_base);
}


///
/// STOCHASTIC structured SVM problem
///
StructuredSVM::StochasticStructuredSVMProblem::StochasticStructuredSVMProblem(
	StructuredSVM::StructuredSVMProblem* ssvm_prob)
	: ssvm_prob(ssvm_prob),
		elements_count(ssvm_prob->Base()->loss_functions.size()) {
}

StructuredSVM::StochasticStructuredSVMProblem::~StochasticStructuredSVMProblem() {
}

// Evaluate objective at x and subgradient at x.
double StructuredSVM::StochasticStructuredSVMProblem::Eval(
	unsigned int sample_id, const std::vector<double>& x,
	std::vector<double>& grad) {
	assert(x.size() == ssvm_prob->Dimensions());
	assert(grad.size() == ssvm_prob->Dimensions());
	assert(sample_id < elements_count);

	// 1. Convert x into factor parameters
	ssvm_prob->LinearToFactorWeights(x);
	std::fill(grad.begin(), grad.end(), 0.0);

	// 2. Setup parameter gradient
	ssvm_prob->ClearParameterGradient();

	// 3. Compute loss-related parameter gradient
	Likelihood lh(ssvm_prob->Base()->fg_model);
	double obj = ssvm_prob->EvaluateLossGradient(lh, sample_id);

	// 4. Add (1/N)'th of the regularizer
	obj += ssvm_prob->AddRegularizer(1.0/static_cast<double>(elements_count));

	// 5. Convert gradient into linear form
	ssvm_prob->ParameterGradientToLinear(grad);

	return (obj);
}

unsigned int StructuredSVM::StochasticStructuredSVMProblem::Dimensions() const {
	return (ssvm_prob->Dimensions());
}

size_t
StructuredSVM::StochasticStructuredSVMProblem::NumberOfElements() const {
	return (elements_count);
}

void StructuredSVM::StochasticStructuredSVMProblem::ProvideStartingPoint(
	std::vector<double>& x0) const {
	assert(x0.size() == ssvm_prob->Dimensions());
	std::fill(x0.begin(), x0.end(), 0.0);
}


///
/// BMRM2
///

StructuredSVM::BMRM2StructuredSVMProblem::BMRM2StructuredSVMProblem(
	StructuredSVMProblem* ssvm_prob)
	: ssvm_prob(ssvm_prob) {
	std::vector<double> zero_vec(ssvm_prob->Dimensions(), 0.0);
	At.push_back(zero_vec);
	bt.push_back(0.0);
}

StructuredSVM::BMRM2StructuredSVMProblem::~BMRM2StructuredSVMProblem() {
}

void StructuredSVM::BMRM2StructuredSVMProblem::AddCurrentSubgradient(
	double R_emp) {
	// Obtain gradient
	std::vector<double> grad(ssvm_prob->Dimensions());
	ssvm_prob->ParameterGradientToLinear(grad);

	// Obtain current weights
	std::vector<double> w(ssvm_prob->Dimensions());
	ssvm_prob->FactorWeightsToLinear(w);
	double b_t = R_emp - std::inner_product(w.begin(), w.end(),
		grad.begin(), 0.0);

	// Add aggregated cutting plane: <a,x> + b_t <= xi
	At.push_back(grad);
	bt.push_back(b_t);
}

double StructuredSVM::BMRM2StructuredSVMProblem::OptimizeDual(
	std::vector<double>& w_opt, std::vector<double>& alpha_opt,
	double conv_tol, unsigned int max_iter, double& pd_gap, bool verbose) {
	// Basic problem properties
	size_t t = At.size();
	assert(t >= 1);
	size_t n = At[0].size();
	std::vector<double> alpha_grad(t, 0.0);

	// Repeatedly used quantities
	std::vector<double> neg_A_alpha;
	std::vector<double> A_w(t);

	// Initialize alpha, w
	if (alpha_opt.empty()) {
		alpha_opt.resize(t);
		std::fill(alpha_opt.begin(), alpha_opt.end(),
			1.0 / static_cast<double>(t));
	}
	w_opt.resize(n);
	pd_gap = std::numeric_limits<double>::infinity();

	// SPGA parameters
	double nu = 1.0e-4;
	double ss_min = 1.0e-10;
	double ss_max = 1.0e10;
	size_t spga_h = 10;
	size_t spga_hi = 0;
	std::vector<double> spga_hval(spga_h,
		-std::numeric_limits<double>::infinity());
	double ss_bb = -1.0;
	double interp_cubic_allowed = 1.0e-6;

	// Problem objective to be returned later
	double obj_dual = std::numeric_limits<double>::signaling_NaN();
	double d_fval = std::numeric_limits<double>::signaling_NaN();
	bool is_uptodate = false;

	// Spectral Projected Gradient Algorithm (SPG2 in Birgin et al., 1999)
	// (see prototype/eg/spga.m)
	double conv = std::numeric_limits<double>::infinity();
	for (unsigned int iter = 1; conv >= conv_tol
		&& (max_iter == 0 || iter <= max_iter); ++iter) {
		// Evaluate dual objective and gradient in alpha
		if (is_uptodate == false) {
			d_fval = Eval(alpha_opt, alpha_grad, w_opt);
		}
		obj_dual = -d_fval;

		// Initial step size heuristic: ss_bb = 1/max(abs(grad))
		if (ss_bb < 0.0) {
			double grad_max = -std::numeric_limits<double>::infinity();
			for (size_t ti = 0; ti < t; ++ti)
				grad_max = std::max(std::fabs(alpha_grad[ti]), grad_max);
			if (grad_max <= 1.0e-10) {
				if (verbose) {
					std::cout << "   [spga] initial gradient zero, "
						<< "assuming convergence." << std::endl;
				}
				break;	// converged
			}
			ss_bb = 1.0 / grad_max;
		}
		ss_bb = std::min(ss_max, std::max(ss_min, ss_bb));

		// Evaluate primal objective and compute primal-dual gap
		double obj_primal = EvalPrimal(w_opt);
		pd_gap = obj_primal - obj_dual;
		conv = pd_gap;
		if (verbose) {
			std::cout << "   [spga] iter " << iter << ", pobj " << obj_primal
				<< ", dobj " << obj_dual << ", gap " << pd_gap
				<< ", ss_bb " << ss_bb << std::endl;
		}
		if (pd_gap < -1.0e-5) {
			std::cout << "### WARNING: pd_gap " << pd_gap << " < 0" << std::endl;
		}
		if (conv <= conv_tol) {
			if (verbose) {
				std::cout << "   [spga] converged with tolerance "
					<< conv << std::endl;
			}
			break;
		}

		// Manage monotonicity array
		spga_hval[spga_hi] = d_fval;
		spga_hi = (spga_hi + 1) % spga_h;

		// Project the gradient onto the simplex: d_k = P(x - ss_bb*grad) - x
		std::vector<double> d_k(alpha_opt);
		std::transform(d_k.begin(), d_k.end(), alpha_grad.begin(), d_k.begin(),
			[ss_bb](double alpha_i, double grad_i) -> double {
				return (alpha_i - ss_bb*grad_i); });
		ProjectOntoSimplex(d_k);
		std::transform(d_k.begin(), d_k.end(), alpha_opt.begin(), d_k.begin(),
			[](double di, double ai) -> double { return (di - ai); });

		// Monotonicity guard
		double f_b = *std::max_element(spga_hval.begin(), spga_hval.end());

		// Line search
		double gxk_dk = std::inner_product(alpha_grad.begin(),
			alpha_grad.end(), d_k.begin(), 0.0);
		std::vector<double> alpha_opt_ss(alpha_opt.size(), 0.0);
		std::vector<double> alpha_grad_ss(alpha_opt_ss.size(), 0.0);
		std::vector<double> w_opt_ss(w_opt.size(), 0.0);
		double d_fval_xss = 0.0;
		double ss = 1.0;
		unsigned int tries = 0;
		do {
			// f(x + ss*d_k)
			std::transform(alpha_opt.begin(), alpha_opt.end(), d_k.begin(),
				alpha_opt_ss.begin(), [ss](double ai, double dki) -> double {
					return (ai + ss*dki); });
			d_fval_xss = Eval(alpha_opt_ss, alpha_grad_ss, w_opt_ss);
			tries += 1;
			if (d_fval_xss <= f_b + nu*ss*gxk_dk) {
				// Accept
				if (verbose) {
					std::cout << "      accepted step size " << ss
						<< " in " << tries << " steps" << std::endl;
				}
				break;
			}

			// Safe-guarded cubic interpolation
			double g_a = d_fval;
			double gt_a = gxk_dk;
			double g_b = d_fval_xss;
			double gt_b = std::inner_product(alpha_grad_ss.begin(),
				alpha_grad_ss.end(), d_k.begin(), 0.0);
			double z = (3.0*(g_a - g_b) / (ss - 0.0)) + gt_a + gt_b;
			double w = std::sqrt(z*z - gt_a*gt_b);
			double ss_cubic = ss - (ss-0.0)*((gt_b+w-z)/(gt_b-gt_a+2.0*w));
			if (ss_cubic >= interp_cubic_allowed*ss &&
				ss_cubic <= (1.0 - interp_cubic_allowed)*ss) {
				ss = ss_cubic;
			} else {
				ss *= 0.5;
			}
		} while (true);

		// Compute Barzilai-Borwein step size ss_bb
		std::vector<double> s_k(alpha_opt.size(), 0.0);
		std::transform(alpha_opt_ss.begin(), alpha_opt_ss.end(),
			alpha_opt.begin(), s_k.begin(),
			[](double ani, double ai) -> double { return (ani - ai); });
		std::vector<double> y_k(alpha_grad.size(), 0.0);
		std::transform(alpha_grad_ss.begin(), alpha_grad_ss.end(),
			alpha_grad.begin(), y_k.begin(),
			[](double agni, double agi) -> double { return (agni - agi); });
		double b_k = std::inner_product(s_k.begin(), s_k.end(),
			y_k.begin(), 0.0);
		if (b_k <= 0.0) {
			ss_bb = ss_max;
		} else {
			ss_bb = std::inner_product(y_k.begin(), y_k.end(),
				y_k.begin(), 0.0) / b_k;
		}

		// Update
		alpha_opt = alpha_opt_ss;
		alpha_grad = alpha_grad_ss;
		d_fval = d_fval_xss;
		w_opt = w_opt_ss;
		is_uptodate = true;
	}
	ssvm_prob->LinearToFactorWeights(w_opt);
	return (obj_dual);
}

// Return minimization objective (maximization obj_dual is negative of it)
double StructuredSVM::BMRM2StructuredSVMProblem::Eval(
	const std::vector<double>& alpha, std::vector<double>& alpha_grad,
	std::vector<double>& w_opt) {
	// Obtain dual problem gradient
	std::vector<double> neg_A_alpha;
	Eval_neg_A_alpha(alpha, neg_A_alpha);	// u = -A alpha
	ssvm_prob->LinearToFactorWeights(neg_A_alpha);

	// [d_fval,Og]=Omega_fd(-A*alpha)
	ssvm_prob->ClearParameterGradient();
	double d_fval = ssvm_prob->EvaluateFenchelDual();	// Omega_fd(u)
	d_fval -= std::inner_product(bt.begin(), bt.end(), alpha.begin(), 0.0);

	// w_opt is Og
	size_t n = At[0].size();
	w_opt.resize(n);
	std::fill(w_opt.begin(), w_opt.end(), 0.0);
	ssvm_prob->ParameterGradientToLinear(w_opt);

	// alpha_grad = -A'*Og - b
	Eval_neg_AT_u(w_opt, alpha_grad);
	std::transform(bt.begin(), bt.end(), alpha_grad.begin(),
		alpha_grad.begin(), [](double bt_i, double ag_i) -> double {
			return (ag_i - bt_i);   // - b
		});

	return (d_fval);
}

double StructuredSVM::BMRM2StructuredSVMProblem::EvalPrimal(
	const std::vector<double>& w_opt) {
	// Compute primal objective: Omega(w_opt) + max(A'*w_opt + b)
	ssvm_prob->LinearToFactorWeights(w_opt);
	double obj_primal = ssvm_prob->AddRegularizer();

	// alpha_grad = -A'*Og - b
	size_t t = At.size();
	assert(t >= 1);
	std::vector<double> nATw(t, 0.0);
	Eval_neg_AT_u(w_opt, nATw);	// -A'w_opt
	std::transform(bt.begin(), bt.end(), nATw.begin(),
		nATw.begin(), [](double bt_i, double ag_i) -> double {
			return (ag_i - bt_i);	// - b
		});

	// + max(A'*w_opt + b)
	double max_resp = -std::numeric_limits<double>::infinity();
	for (size_t ti = 0; ti < t; ++ti)
		max_resp = std::max(max_resp, -nATw[ti]);
	obj_primal += max_resp;

	return (obj_primal);
}

// neg_A_alpha = -A alpha, A is (n,t), alpha is (t,1)
void StructuredSVM::BMRM2StructuredSVMProblem::Eval_neg_A_alpha(
	const std::vector<double>& alpha,
	std::vector<double>& neg_A_alpha) const {
	assert(At.size() >= 1);
	assert(At.size() == alpha.size());

	neg_A_alpha.resize(At[0].size());	// (n,1)
	std::fill(neg_A_alpha.begin(), neg_A_alpha.end(), 0.0);
	for (size_t c = 0; c < At.size(); ++c) {
		std::transform(At[c].begin(), At[c].end(), neg_A_alpha.begin(),
			neg_A_alpha.begin(),
			[&alpha,c](double At_i, double nAa_i) -> double {
				return (nAa_i - alpha[c]*At_i);
			});
	}
}

// neg_AT_u = -A'u, A' is (t,n), u is (n,1)
void StructuredSVM::BMRM2StructuredSVMProblem::Eval_neg_AT_u(
	const std::vector<double>& u, std::vector<double>& neg_AT_u) const {
	assert(At.size() >= 1);
	assert(At[0].size() == u.size());

	neg_AT_u.resize(At.size());	// (t,1)
	for (size_t c = 0; c < At.size(); ++c) {
		neg_AT_u[c] = -std::inner_product(At[c].begin(), At[c].end(),
			u.begin(), 0.0);
	}
}

// Project alpha \in R^d onto the standard simplex
void StructuredSVM::BMRM2StructuredSVMProblem::ProjectOntoSimplex(
	std::vector<double>& alpha) const {
	if (alpha.empty())
		return;
	if (alpha.size() == 1) {
		alpha[0] = 1.0;
		return;
	}

	// d >= 2 case
	std::vector<double> ta(alpha);
	std::sort(ta.begin(), ta.end(), std::greater<double>());
	double ts = 0.0;
	double tx = 0.0;
	bool set_tx = false;
	for (size_t i = 0; i < alpha.size()-1; ++i) {
		ts += ta[i];
		tx = (ts - 1.0) / (static_cast<double>(i + 1));
		if (tx < ta[i+1])
			continue;

		set_tx = true;
		break;
	}
	if (set_tx == false) {
		tx = (ts + ta[ta.size()-1] - 1.0) / static_cast<double>(ta.size());
	}
	std::transform(alpha.begin(), alpha.end(), alpha.begin(),
		[tx](double a) -> double { return (std::max(0.0, a - tx)); });
}

}

