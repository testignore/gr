
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <functional>
#include <limits>
#include <ctime>
#include <cmath>
#include <cassert>

#include <boost/timer.hpp>
#include <boost/random.hpp>
#include <boost/math/tr1.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/lambda/lambda.hpp>

#include "FunctionMinimization.h"

using namespace boost::lambda;

namespace Grante {

double FunctionMinimization::BarzilaiBorweinMinimize(
	FunctionMinimizationProblem& prob,
	std::vector<double>& x_opt, double conv_tol,
	unsigned int max_iter, bool verbose) {
	unsigned int dim = prob.Dimensions();

	// Gradient, last gradient and alpha value required for BB iteration
	std::vector<double> grad(dim, 0.0);
	std::vector<double> grad_last(dim, 0.0);
	double alpha_last = std::numeric_limits<double>::infinity();

	// Initialize x
	std::vector<double> x(dim);
	prob.ProvideStartingPoint(x);

	boost::timer total_timer;
	double obj = std::numeric_limits<double>::signaling_NaN();
	for (unsigned int iter = 0; max_iter == 0 || iter < max_iter; ++iter) {
		obj = prob.Eval(x, grad);

		// Convergence check
		double grad_norm = EuclideanNorm(grad);
		if (verbose && (iter % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      objective      |grad|" << std::endl;
		}
		if (verbose) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			// Iteration
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << iter << "  ";
			// Total runtime
			std::cout << std::setiosflags(std::ios::left)
				<< std::resetiosflags(std::ios::scientific)
				<< std::setiosflags(std::ios::fixed)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setprecision(1)
				<< std::setw(6) << total_timer.elapsed() << "s  ";
			std::cout << std::resetiosflags(std::ios::fixed);

			// Objective function
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(5)
				<< std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::showpos)
				<< std::setw(7) << obj << "   ";
			// Gradient norm
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << grad_norm;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		if (grad_norm < conv_tol) {
			x_opt = x;
			return (obj);
		}

		// Choose alpha
		double alpha = grad_norm;	// Initialization heuristic
		if (iter == 0) {
			// First iteration: assert feasibility by line search
			WolfeLineSearch linesearch(&prob, x, grad, grad, obj, 1e-4, 0.9);
			linesearch.ComputeStepLength(alpha);
			alpha = 1.0 / alpha;
		}
		if (iter >= 1) {
			double anom = 0.0;
			double adenom = 0.0;
			for (unsigned int d = 0; d < dim; ++d) {
				anom += -grad_last[d]*(grad[d] - grad_last[d]);
				adenom += grad_last[d]*grad_last[d];
			}
			alpha = alpha_last * (anom / adenom);
#if 0
			std::cout << "   alpha = alpha_last(" << alpha_last
				<< ") * (anom(" << anom << ") / adenom(" << adenom << ")"
				<< " = " << alpha << std::endl;
#endif
		}
		assert(alpha > 0.0);

		// Update x
		for (unsigned int d = 0; d < dim; ++d)
			x[d] -= grad[d] / alpha;

		// Keep iterates of the gradient and stepsize
		alpha_last = alpha;
		grad_last = grad;
	}

	// Iteration limit reached
	x_opt = x;
	return (prob.Eval(x, grad));
}

double FunctionMinimization::LimitedMemoryBFGSMinimize(
	FunctionMinimizationProblem& prob,
	std::vector<double>& x_opt, double conv_tol, unsigned int max_iter,
	bool verbose, unsigned int lbfgs_m) {
	unsigned int dim = prob.Dimensions();

	// Gradient, last gradient and alpha value required for BB iteration
	std::vector<double> grad(dim, 0.0);
	std::vector<double> grad_last(dim, 0.0);

	// Initialize x
	std::vector<double> x(dim);
	prob.ProvideStartingPoint(x);
	std::vector<double> xprev(dim);
	std::vector<double> gradprev(dim);

	// List of previous s,y,rho_i
	lbfgs_mem_type lbfgs_mem;

	boost::timer total_timer;
	double obj = std::numeric_limits<double>::signaling_NaN();
	bool is_valid = false;
	bool is_restart = false;
	unsigned int ls_evals = 0;
	unsigned int iter = 0;
	for ( ; max_iter == 0 || iter < max_iter; ++iter) {
		// If no information about current query point has been computed, do
		// so now
		if (is_valid == false) {
			obj = prob.Eval(x, grad);
			is_valid = true;
		}
		double grad_norm = EuclideanNorm(grad);

		if (verbose && (iter % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      objective      |grad|   "
				<< "mem     ls#" << std::endl;
		}
		if (verbose) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			// Iteration
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << iter << "  ";
			// Total runtime
			std::cout << std::setiosflags(std::ios::left)
				<< std::resetiosflags(std::ios::scientific)
				<< std::setiosflags(std::ios::fixed)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setprecision(1)
				<< std::setw(6) << total_timer.elapsed() << "s  ";
			std::cout << std::resetiosflags(std::ios::fixed);

			// Objective function
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(5)
				<< std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::showpos)
				<< std::setw(7) << obj << "   ";
			// Gradient norm
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << grad_norm;
			// LBFGS memory size
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << lbfgs_mem.size() << "  ";
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << ls_evals << "  ";
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		// Convergence check based on gradient norm
		if (prob.HasConverged(x, grad, conv_tol))
			break;	// converged

		// Insert differential information into Hessian approximation
		if (iter > 0 && is_restart == false) {
			// xprev: s_k
			std::transform(x.begin(), x.end(), xprev.begin(),
				xprev.begin(), _1 - _2);
			// gradprev: y_k
			std::transform(grad.begin(), grad.end(), gradprev.begin(),
				gradprev.begin(), _1 - _2);

			// Heuristically ensure stability by ignoring unstable updates.
			// As to what entails 'unstable' there exist different opinions.
			// TODO: replace with true damped-Newton update (in inverse H
			// form used by L-BFGS)
			double ys_p = std::inner_product(xprev.begin(), xprev.end(),
				gradprev.begin(), 0.0);
			double yy_p = std::inner_product(gradprev.begin(), gradprev.end(),
				gradprev.begin(), 0.0);
//			if (ys_p >= 1.0e-12) {
			if (ys_p >= 1.0e-12*yy_p) {
#if 0
				std::cout << "    lbfgs update with ys_p " << ys_p
					<< std::endl;
#endif
				double rho = 1.0 / ys_p;
				lbfgs_mem.push_front(lbfgs_mem_type::value_type(
					xprev, gradprev, rho));

				// Remove old element from the lbfgs memory, if necessary
				if (lbfgs_mem.size() > lbfgs_m)
					lbfgs_mem.pop_back();
			} else {
				std::cout << "    LBFGS update too large (ys "
					<< ys_p << ", yy " << yy_p << ")" << std::endl;
			}
		}
		// Save current iterate for next update
		std::copy(x.begin(), x.end(), xprev.begin());
		std::copy(grad.begin(), grad.end(), gradprev.begin());

		// Compute new ascent direction H_k \nabla_x f(x_k)
		// Recent-to-oldest
		std::list<double> alpha_list;
		for (lbfgs_mem_type::const_iterator li = lbfgs_mem.begin();
			li != lbfgs_mem.end(); ++li) {
			// alpha_i = rho_i s_i' q
			double alpha_i = li->get<2>() * std::inner_product(
				grad.begin(), grad.end(), li->get<0>().begin(), 0.0);
			// q <= q - alpha_i y_i
			std::transform(grad.begin(), grad.end(), li->get<1>().begin(),
				grad.begin(), _1 - alpha_i * _2);
			alpha_list.push_back(alpha_i);
		}
		// Diagonal scaling: q = H^0 q
		double gamma = 1.0;
		if (iter > 0 && is_restart == false && lbfgs_mem.empty() == false) {
			lbfgs_mem_type::const_iterator li_last = lbfgs_mem.begin();

			// gamma = (s_{k-1}' y_{k-1}) / (y_{k-1}' y_{k-1})
			gamma = std::inner_product(li_last->get<0>().begin(),
				li_last->get<0>().end(), li_last->get<1>().begin(), 0.0);
			gamma /= std::inner_product(li_last->get<1>().begin(),
				li_last->get<1>().end(), li_last->get<1>().begin(), 0.0);
		}
		std::transform(grad.begin(), grad.end(), grad.begin(), gamma * _1);
		// Reverse: oldest-to-recent
		std::list<double>::const_reverse_iterator ai = alpha_list.rbegin();
		for (lbfgs_mem_type::const_reverse_iterator li = lbfgs_mem.rbegin();
			li != lbfgs_mem.rend(); ++li, ++ai) {
			// beta = rho_i y_i' q
			double beta = li->get<2>() * std::inner_product(
				grad.begin(), grad.end(), li->get<1>().begin(), 0.0);

			// q <- q + (alpha_i-beta)*s_i
			std::transform(grad.begin(), grad.end(), li->get<0>().begin(),
				grad.begin(), _1 + (*ai-beta)*_2);
		}
		// Now 'grad' contains an adjusted gradient direction
		is_restart = false;

		// Check cosine angle between gradient and transformed gradient
		double x0_phi_grad = std::inner_product(grad.begin(), grad.end(),
			gradprev.begin(), 0.0);
		double cos_a = x0_phi_grad /
			(EuclideanNorm(gradprev) * EuclideanNorm(grad));
		if (cos_a <= -1.0e-8) {
			std::cout << "### FATAL: LBFGS approximation lost psd, angle "
				<< cos_a << std::endl;
			assert(0);
		} else if (cos_a <= 1.0e-7) {
			// Numerical issues, degenerate true Hessian or converged.
			std::cout << "### WARNING: LBFGS gradient orthogonality issue, "
				<< "aborting." << std::endl;
			break;
		}

		if ((boost::math::isnan)(x0_phi_grad)) {
			std::cout << "### WARNING: LBFGS gradient or perturbed gradient NaN, "
				<< "aborting." << std::endl;
			break;
		}
		// Check it is a descent direction
		if (x0_phi_grad <= -1.0e-8) {
			std::cout << "### FATAL: LBFGS approximation lost psd, x0_phi_grad "
				<< x0_phi_grad << std::endl;
			assert(0);
		} else if (x0_phi_grad <= 1.0e-10) {
#if 0
			std::cout << "### WARNING: LBFGS gradient, numerical issue, "
				<< "phi'(0) = " << x0_phi_grad << std::endl;
#endif
			// Numerical issues, degenerate true Hessian or converged.
			break;
		}

		// Perform linesearch in descent direction
		WolfeLineSearch linesearch(&prob, x, gradprev, grad, obj, 1e-4, 0.9);
		//SimpleLineSearch linesearch(&prob, x, gradprev, grad, obj);
		double alpha = 1.0;
		// Be very careful on the first step
		if (iter == 0) {
//			alpha = 1.0e-5;
			if (EuclideanNorm(grad) >= 1.0)
				alpha = 1.0 / EuclideanNorm(grad);
			alpha = std::max(1.0e-12, alpha);
		}

		ls_evals = linesearch.ComputeStepLengthUpdate(x, grad, obj, alpha);
#if 0
		std::cout << "   step size " << alpha
			<< " in " << eval << " evaluations" << std::endl;
#endif

		if ((boost::math::isnan)(alpha)) {
			std::cout << "   * Line search failed (alpha="
				<< alpha << ")" << std::endl;
			std::copy(xprev.begin(), xprev.end(), x.begin());
			break;
		}
		if (alpha <= 1.0e-12) {
			std::cout << "   * Line search yielded step size alpha="
				<< alpha << std::endl;
			if (iter > 0) {
				std::cout << "   * Assuming convergence." << std::endl;
				break;
			}
		}
		// Successful line search with up-to-date step
		is_valid = true;
	}

	// Iteration limit reached
	x_opt = x;
	return (prob.Eval(x, grad));
}

double FunctionMinimization::SubgradientMethodMinimize(
	FunctionMinimizationProblem& prob,
	std::vector<double>& x_opt, double conv_tol, unsigned int max_iter,
	bool verbose) {
	unsigned int dim = prob.Dimensions();
	std::vector<double> grad(dim, 0.0);

	// Initialize x
	std::vector<double> x(dim);
	prob.ProvideStartingPoint(x);

	boost::timer total_timer;
	for (unsigned int iter = 0; (max_iter == 0) || iter < max_iter; ++iter) {
		double obj = prob.Eval(x, grad);

		// Convergence check
		double grad_norm = EuclideanNorm(grad);
		if (verbose && (iter % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      objective      |grad|" << std::endl;
		}
		if (verbose) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			// Iteration
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << iter << "  ";
			// Total runtime
			std::cout << std::setiosflags(std::ios::left)
				<< std::resetiosflags(std::ios::scientific)
				<< std::setiosflags(std::ios::fixed)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setprecision(1)
				<< std::setw(6) << total_timer.elapsed() << "s  ";
			std::cout << std::resetiosflags(std::ios::fixed);

			// Objective function
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(5)
				<< std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::showpos)
				<< std::setw(7) << obj << "   ";
			// Gradient norm
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << grad_norm;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		if (grad_norm < conv_tol) {
			x_opt = x;
			return (obj);
		}

		// Choose step size
		double alpha_m = 200.0;
		double alpha = (1.0 + alpha_m) /
			(static_cast<double>(iter + 1) + alpha_m);
		alpha /= grad_norm * grad_norm;

		// Update
		for (unsigned int d = 0; d < dim; ++d)
			x[d] -= alpha * grad[d];
	}

	// Iteration limit reached
	x_opt = x;
	return (prob.Eval(x, grad));
}

double FunctionMinimization::GradientMethodMinimize(
	FunctionMinimizationProblem& prob,
	std::vector<double>& x_opt, double conv_tol, unsigned int max_iter,
	bool verbose) {
	unsigned int dim = prob.Dimensions();
	std::vector<double> grad(dim, 0.0);
	std::vector<double> grad_prev(dim, 0.0);

	// Initialize x
	std::vector<double> x(dim);
	prob.ProvideStartingPoint(x);

	boost::timer total_timer;
	double alpha = -1.0;
	double beta = 0.5;
	for (unsigned int iter = 0; (max_iter == 0) || iter < max_iter; ++iter) {
		double obj = prob.Eval(x, grad);

		// Convergence check
		double grad_norm = EuclideanNorm(grad);
		if (verbose && (iter % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      objective      |grad|       alpha"
				<< std::endl;
		}
		if (verbose) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			// Iteration
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << iter << "  ";
			// Total runtime
			std::cout << std::setiosflags(std::ios::left)
				<< std::resetiosflags(std::ios::scientific)
				<< std::setiosflags(std::ios::fixed)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setprecision(1)
				<< std::setw(6) << total_timer.elapsed() << "s  ";
			std::cout << std::resetiosflags(std::ios::fixed);

			// Objective function
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(5)
				<< std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::showpos)
				<< std::setw(7) << obj << "   ";
			// Gradient norm
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << grad_norm
				<< "   ";
			// alpha
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setw(5)
				<< std::setiosflags(std::ios::left) << alpha;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		if (grad_norm < conv_tol) {
			x_opt = x;
			return (obj);
		}

		// Choose step size
		if (alpha < 0.0) {
			alpha = 1.0 / (grad_norm * grad_norm);
		} else {
			if (std::inner_product(grad.begin(), grad.end(),
				grad_prev.begin(), 0.0) < 0.0) {
				alpha *= beta;
			}
		}
		std::copy(grad.begin(), grad.end(), grad_prev.begin());

		// Update
		for (unsigned int d = 0; d < dim; ++d)
			x[d] -= alpha * grad[d];
	}

	// Iteration limit reached
	x_opt = x;
	return (prob.Eval(x, grad));
}

bool FunctionMinimization::CheckDerivative(FunctionMinimizationProblem& prob,
	double x_range, unsigned int test_count, double dim_eps, double grad_tol) {
	assert(dim_eps > 0.0);
	assert(grad_tol > 0.0);

	// Random number generation, for random perturbations
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > rand_perturb(rgen, rdestu);

	// Random number generation, for random dimensions
	unsigned int dim = prob.Dimensions();
	boost::mt19937 rgen2(static_cast<const boost::uint32_t>(std::time(0))+2);
	boost::uniform_int<unsigned int> rdestd(0, dim-1);
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_dim(rgen2, rdestd);

	// Get base
	std::vector<double> x0(dim);
	prob.ProvideStartingPoint(x0);
	std::vector<double> xtest(dim);
	std::vector<double> grad(dim);
	std::vector<double> grad_d(dim);	// dummy

	for (unsigned int test_id = 0; test_id < test_count; ++test_id) {
		xtest = x0;
		for (unsigned int d = 0; d < dim; ++d)
			xtest[d] += 2.0*x_range*rand_perturb() - x_range;

		// Get exact derivative
		double xtest_fval = prob.Eval(xtest, grad);

		// Compute first-order finite difference approximation
		unsigned int test_dim = rand_dim();
		xtest[test_dim] += dim_eps;
		double xtest_d_fval = prob.Eval(xtest, grad_d);
		double deriv_fd = (xtest_d_fval - xtest_fval) / dim_eps;

		// Check accuracy
		if (fabs(deriv_fd - grad[test_dim]) > grad_tol) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			std::cout << std::endl;
			std::cout << "### DERIVATIVE CHECKER WARNING" << std::endl;
			std::cout << "### during test " << (test_id+1) << " a violation "
				<< "in gradient computation was found:" << std::endl;
			std::cout << std::setprecision(6)
				<< std::setiosflags(std::ios::scientific);
			std::cout << "### dim " << test_dim << ", exact " << grad[test_dim]
				<< ", finite-diff " << deriv_fd
				<< ", absdiff " << fabs(deriv_fd - grad[test_dim])
				<< std::endl;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);

			return (false);
		}
	}
	return (true);
}

double FunctionMinimization::EuclideanNorm(const std::vector<double>& vec) {
	double rs = 0.0;
	for (std::vector<double>::const_iterator vi = vec.begin();
		vi != vec.end(); ++vi) {
		rs += (*vi) * (*vi);
	}
	return (sqrt(rs));
}

// Wolfe line-search method, see [Nocedal&Wright], page 60.
FunctionMinimization::WolfeLineSearch::WolfeLineSearch(
	FunctionMinimizationProblem* prob, const std::vector<double>& x0,
	const std::vector<double>& x0_grad,
	const std::vector<double>& H_grad, double x0_fval,
	double c1, double c2)
	: prob(prob), x0(x0), x0_grad(x0_grad), H_grad(H_grad),
		x0_fval(x0_fval), x0_phi_grad(0),
		evaluation_count(0),
		xalpha_val(std::numeric_limits<double>::signaling_NaN()),
		c1(c1), c2(c2)
{
	assert(x0.size() > 0);
	xalpha.resize(x0.size());
	xalphagrad.resize(x0.size());

	// phi'(0) = - p' \nabla_x f(x_k)
	x0_phi_grad = -std::inner_product(x0_grad.begin(), x0_grad.end(),
		H_grad.begin(), 0.0);
	assert(x0_phi_grad < 0.0);
}

unsigned int FunctionMinimization::WolfeLineSearch::ComputeStepLength(
	double& alpha) {
	// Previous alpha
	double alpha_prev = 0.0;
	double alpha_prev_fval = x0_fval;
	double alpha_prev_grad = x0_phi_grad;
	double alpha_max = 1e6;
//	alpha = 1.0;

#if 0
	for (double bf = 0.0; bf < 1.0; bf += 0.01) {
		double tfval, tfgrad;
		Evaluate(bf, tfval, tfgrad);
		std::cout << bf << " " << tfval << " " << tfgrad << " # TEST" << std::endl;
	}
#endif

	// Current alpha
	double phi_alpha_fval;
	double phi_alpha_grad;

	// Find a stepsize interval satisfying the strong Wolfe conditions for a
	// given iterate x and gradient gx and descent direction d:
	//   1. Armijo: f(x + alpha d) <= f(x) + c1 alpha gx' d,
	//   2. Curvature: |nabla_alpha f(x + alpha d)| <= c2 |gx' d|.
	for (unsigned int n = 0; true; ++n) {
		Evaluate(alpha, phi_alpha_fval, phi_alpha_grad);

		// If Armijo condition is violated: zoom, as a point satisfying the
		// Wolfe condition must exist in [alpha_{i-1}, alpha_i].
		if (phi_alpha_fval > (x0_fval + c1*alpha*x0_phi_grad)) {
			if (n == 0) {
				alpha = Zoom(0.0, alpha, x0_fval, phi_alpha_fval, x0_phi_grad);
			} else {
				alpha = Zoom(alpha_prev, alpha, alpha_prev_fval,
					phi_alpha_fval, alpha_prev_grad);
			}
			break;
		}
		if (n > 0 && phi_alpha_fval >= alpha_prev_fval) {
			alpha = Zoom(alpha_prev, alpha, alpha_prev_fval,
				phi_alpha_fval, alpha_prev_grad);
			break;
		}

		// The Armijo condition is satisfied.  If in addition the curvature
		// condition ("function must be flat around alpha") is satisfied, then
		// we found a point satisfying the Wolfe conditions.
#if 0
		// STRONG
		if (std::fabs(phi_alpha_grad) <= -c2*x0_phi_grad)
			break;
#endif
		if (phi_alpha_grad >= c2*x0_phi_grad)
			break;

		// If the gradient increases again, then a point satisfying the Wolfe
		// condition must exist in [alpha_{i-1}, alpha_i].
		if (phi_alpha_grad >= 0) {
			alpha = Zoom(alpha_prev, alpha, alpha_prev_fval,
				phi_alpha_fval, alpha_prev_grad);
			break;
		}

		// Further progress can be made, scale step size up by a factor
		alpha_prev = alpha;
		alpha_prev_fval = phi_alpha_fval;
		alpha_prev_grad = phi_alpha_grad;
		alpha *= std::sqrt(2.0);
		assert(alpha <= alpha_max);
	}

#if 0
	// If line search failed: try fall back to backtracking line search
	if ((boost::math::isnan)(alpha)) {
		alpha = 1.0;
		for (unsigned int bt_try = 0; bt_try < 20; ++bt_try) {
			alpha *= 0.25;
			Evaluate(alpha, phi_alpha_fval, phi_alpha_grad);
			std::cout << "    backtrack, alpha " << alpha
				<< ", phi(a) " << phi_alpha_fval
				<< ", phi'(a) " << phi_alpha_grad << std::endl;
			std::cout << "    phi(a) = " << phi_alpha_fval
				<< " <= " << x0_fval << " + " << c1*alpha*x0_phi_grad
				<< "?" << std::endl;
			if (phi_alpha_fval <= (x0_fval + c1*alpha*x0_phi_grad))
				return (evaluation_count);
		}
		std::cout << "  Backtracking line search failed." << std::endl;
		alpha = std::numeric_limits<double>::signaling_NaN();
	}
#endif

	return (evaluation_count);
}

unsigned int FunctionMinimization::WolfeLineSearch::ComputeStepLengthUpdate(
	std::vector<double>& x_out, std::vector<double>& grad_out,
	double& fval_out, double& alpha) {
//	alpha = 0.0;
	unsigned int ecount = ComputeStepLength(alpha);

	if ((boost::math::isnan)(alpha))
		return (ecount);

	// A valid step size has been computed
	if (xalpha_val != alpha) {
		// Is not up-to-date, update
		double d1;	// dummy
		double d2;
		Evaluate(alpha, d1, d2);
	}
	assert(xalpha_val == alpha);
	x_out = xalpha;
	grad_out = xalphagrad;
	fval_out = xalphaobj;

	return (ecount);
}

// phi(alpha), alpha >= 0
void FunctionMinimization::WolfeLineSearch::Evaluate(
	double alpha, double& phi_fval, double& phi_grad) {
	// x(alpha) = x - alpha*p
	std::transform(x0.begin(), x0.end(), H_grad.begin(),
		xalpha.begin(), _1 - alpha * _2);
	std::fill(xalphagrad.begin(), xalphagrad.end(), 0.0);

	// Evaluate phi(alpha) and derivative phi'(alpha)
	// phi(alpha) = f(x_k - alpha H_grad)
	// phi'(alpha) = -H_grad' \nabla_x f(x_k - alpha H_grad)
	phi_fval = prob->Eval(xalpha, xalphagrad);
	xalphaobj = phi_fval;	// save phi(alpha)
	xalpha_val = alpha;	// save alpha

	// Univariate derivative is projection onto ascent direction
	phi_grad = -std::inner_product(xalphagrad.begin(), xalphagrad.end(),
		H_grad.begin(), 0.0);
	evaluation_count += 1;
#if 0
	std::cout << "   phi(" << alpha << ") = " << phi_fval << ", grad "
		<< phi_grad << std::endl;
#endif
}

double FunctionMinimization::WolfeLineSearch::Zoom(double alpha_lo,
	double alpha_hi, double alpha_lo_fval, double alpha_hi_fval,
	double alpha_lo_grad) {
	double phi_trial_fval;
	double phi_trial_grad;

#if 0
	double tfval, tfgrad;
	Evaluate(0.0, tfval, tfgrad);
	std::cout << "  phi(0) = " << tfval << ", grad " << tfgrad << std::endl;
#endif
#if 0
	for (double bf = alpha_lo; bf < alpha_hi; bf += 0.01) {
		double tfval, tfgrad;
		Evaluate(bf, tfval, tfgrad);
		std::cout << bf << " " << tfval << " " << tfgrad << " # TEST" << std::endl;
	}
#endif

	unsigned int tries_max = 150;
	unsigned int tries = 0;
	for (; tries < tries_max; ++tries) {
		if ((alpha_hi - alpha_lo) < 1.0e-14) {
			std::cout << "   * Wolfe line search, too small bracket: ["
				<< alpha_lo << "; " << alpha_hi << "]" << std::endl;
			//return (std::numeric_limits<double>::signaling_NaN());
			return (alpha_hi);
		}
#if 0
		{	// check preconditions: alpha_lo satisfies Armijo cond
			double tfval_lo, tfgrad_lo;
			Evaluate(alpha_lo, tfval_lo, tfgrad_lo);
			std::cout << "      ### lo: alpha = " << alpha_lo
				<< ", phi(a) = " << tfval_lo
				<< ", phi'(a) = " << tfgrad_lo << std::endl;
			assert(tfval_lo <= (x0_fval + c1*alpha_lo*x0_phi_grad));

			double tfval_hi, tfgrad_hi;
			Evaluate(alpha_hi, tfval_hi, tfgrad_hi);
			std::cout << "      ### hi: alpha = " << alpha_hi
				<< ", phi(a) = " << tfval_hi
				<< ", phi'(a) = " << tfgrad_hi << std::endl;

			// Check derivative
			double cd_fval, cd_grad;
			Evaluate(alpha_lo + 1e-8, cd_fval, cd_grad);
			double cd_apx = (cd_fval - tfval_lo) / 1e-8;
			std::cout << "        # lo deriv: " << tfgrad_lo
				<< " (exa) vs " << cd_apx << " (apx)"
				<< ", phi(0) " << x0_phi_grad
				<< std::endl;
			Evaluate(alpha_hi + 1e-8, cd_fval, cd_grad);
			cd_apx = (cd_fval - tfval_hi) / 1e-8;
			std::cout << "        # hi deriv: " << tfgrad_hi
				<< " (exa) vs " << cd_apx << " (apx)"
				<< ", phi(hi) " << tfval_hi
				<< std::endl;

#if 1
			if (std::fabs(cd_apx - tfgrad_hi) >= 1.0) {
				for (double bf = alpha_lo; bf < alpha_hi; bf +=
0.01*(alpha_hi-alpha_lo)) {
					double tfval, tfgrad;
					Evaluate(bf, tfval, tfgrad);
					std::cout << bf << " " << tfval << " " << tfgrad << " # TEST" << std::endl;
				}
			}
#endif
		}
#endif
		double alpha_trial = std::numeric_limits<double>::signaling_NaN();
		if ((boost::math::isinf)(alpha_hi_fval)) {
			alpha_trial = 0.9*alpha_lo + 0.1*alpha_hi;
			std::cout << "   * Warning: phi(" << alpha_hi << ") is NaN, "
				<< "trying bisection." << std::endl;
		} else {
			// Interpolation by quadratic, fixing
			// phi(alpha_lo), phi'(alpha_lo), phi(alpha_hi)
			double q_a = (alpha_lo_fval - alpha_hi_fval +
				alpha_lo_grad*(alpha_hi-alpha_lo)) /
				(-alpha_lo*alpha_lo - alpha_hi*alpha_hi +
					2.0*alpha_lo*alpha_hi);
			double q_b = alpha_lo_grad - 2.0*q_a*alpha_lo;
			// double q_c = alpha_hi_fval - q_a*alpha_hi*alpha_hi
			//    - alpha_lo_grad*alpha_hi + 2.0*q_a*alpha_lo*alpha_hi;
			if (q_a <= 1.0e-15) {
				std::cout << "   * Wolfe line search failed due to small "
					<< "quadratic coefficient (q_a=" << q_a << ")." << std::endl;
				std::cout << "     alpha_lo/hi " << alpha_lo << ", "
					<< alpha_hi << std::endl;
				std::cout << "     alpha_lo/hi fval " << alpha_lo_fval << ", "
					<< alpha_hi_fval << std::endl;
				std::cout << "     alpha_lo grad " << alpha_lo_grad << std::endl;
				break;
			}
			alpha_trial = - q_b / (2.0*q_a);
		}
#if 0
		std::cout << "        # alpha_trial: " << alpha_trial
			<< std::endl;
#endif
		Evaluate(alpha_trial, phi_trial_fval, phi_trial_grad);

#if 0
		std::cout << "        # Armijo (c1): "
			<< ((phi_trial_fval <= x0_fval + c1*alpha_trial*x0_phi_grad) ?
				"SATISFIED" : "NOT SATISFIED") << std::endl;
		std::cout << "        # Curvature (c2): "
			<< ((phi_trial_grad >= c2*x0_phi_grad) ? "SATISFIED"
				: "NOT SATISFIED") << std::endl;
		std::cout << "        #   phi'(a) = " << phi_trial_grad
			<< " >= c2*" << x0_phi_grad
			<< " = " << (c2*x0_phi_grad) << std::endl;
#endif

		// Check Armijo condition:
		// phi(alpha) <= phi(0) + c1 alpha phi'(0)
		if (phi_trial_fval > (x0_fval + c1*alpha_trial*x0_phi_grad) ||
			phi_trial_fval >= alpha_lo_fval) {
			// Decrease upper bracket
			alpha_hi = alpha_trial;
			continue;
		}

		// Found a point satisfying the Wolfe conditions
#if 0
		// Strong Wolfe
		if (std::fabs(phi_trial_grad) <= -c2*x0_phi_grad)
			return (alpha_trial);
#endif
		if (phi_trial_grad >= c2*x0_phi_grad)
			return (alpha_trial);

		// Flip bracket
		if (phi_trial_grad*(alpha_hi - alpha_lo) >= 0.0)
			alpha_hi = alpha_lo;

		// Increase lower bracket
		alpha_lo = alpha_trial;
		alpha_lo_fval = phi_trial_fval;
	}

	if (tries >= tries_max) {
		std::cout << "   * Wolfe line search exhausted function evaluation "
			<< "budget (" << tries_max << ")." << std::endl;
	}
	return (std::numeric_limits<double>::signaling_NaN());
}

FunctionMinimization::SimpleLineSearch::SimpleLineSearch(
	FunctionMinimizationProblem* prob, const std::vector<double>& x0,
	const std::vector<double>& x0_grad,
	const std::vector<double>& H_grad, double x0_fval)
	: prob(prob), x0(x0), x0_grad(x0_grad), H_grad(H_grad),
		x0_fval(x0_fval), x0_phi_grad(0),
		evaluation_count(0),
		xalpha_val(std::numeric_limits<double>::signaling_NaN())
{
	assert(x0.size() > 0);
	xalpha.resize(x0.size());
	xalphagrad.resize(x0.size());

	// phi'(0) = - p' \nabla_x f(x_k)
	x0_phi_grad = -std::inner_product(x0_grad.begin(), x0_grad.end(),
		H_grad.begin(), 0.0);
	assert(x0_phi_grad < 0.0);
}

unsigned int FunctionMinimization::SimpleLineSearch::ComputeStepLength(
	double& alpha) {
	// Previous alpha
	double alpha_min = 1.0e-12;
	double alpha_max = 1e6;

	double fa_enlarge = 1.7;
	double fa_shrink = 0.5;

	// Current alpha
	double phi_alpha_fval;
	double phi_alpha_grad;

	// Find a stepsize interval satisfying the strong Wolfe conditions for a
	// given iterate x and gradient gx and descent direction d:
	//   1. Armijo: f(x + alpha d) <= f(x) + c1 alpha gx' d,
	//   2. Curvature: |nabla_alpha f(x + alpha d)| <= c2 |gx' d|.
	unsigned int max_test = 200;
	for (unsigned int n = 0; true; ++n) {
#if 0
		std::cout << "SL n " << n << ", alpha " << alpha << std::endl;
#endif
		if (alpha <= alpha_min) {
			break;
		} else if (alpha >= alpha_max || n >= max_test) {
			if (n >= max_test)
				std::cout << "### WARNING: line search count exceeded, alpha "
<< alpha << std::endl;
			alpha = std::numeric_limits<double>::signaling_NaN();
			break;
		}
		Evaluate(alpha, phi_alpha_fval, phi_alpha_grad);

		// If Armijo condition is violated, shrink.
		if (phi_alpha_fval > (x0_fval + 1.0e-4*alpha*x0_phi_grad)) {
			alpha *= fa_shrink;
#if 0
			std::cout << "    armijo fail, shrink to " << alpha << std::endl;
#endif
			continue;
		}

		// The Armijo condition is satisfied.  If in addition the curvature
		// condition ("function must be flat around alpha") is satisfied, then
		// we found a point satisfying the Wolfe conditions.
		// Otherwise, enlarge alpha.
		if (phi_alpha_grad < 0.9*x0_phi_grad) {
			alpha *= fa_enlarge;
#if 0
			std::cout << "    wolfe fail, increase to " << alpha << std::endl;
#endif
			continue;
		}

		break;
	}

	return (evaluation_count);
}

unsigned int FunctionMinimization::SimpleLineSearch::ComputeStepLengthUpdate(
	std::vector<double>& x_out, std::vector<double>& grad_out,
	double& fval_out, double& alpha) {
	unsigned int ecount = ComputeStepLength(alpha);

	if ((boost::math::isnan)(alpha))
		return (ecount);

	// A valid step size has been computed
	if (xalpha_val != alpha) {
		// Is not up-to-date, update
		double d1;	// dummy
		double d2;
		Evaluate(alpha, d1, d2);
	}
	assert(xalpha_val == alpha);
	x_out = xalpha;
	grad_out = xalphagrad;
	fval_out = xalphaobj;

	return (ecount);
}

void FunctionMinimization::SimpleLineSearch::Evaluate(
	double alpha, double& phi_fval, double& phi_grad) {
	// x(alpha) = x - alpha*p
	std::transform(x0.begin(), x0.end(), H_grad.begin(),
		xalpha.begin(), _1 - alpha * _2);
	std::fill(xalphagrad.begin(), xalphagrad.end(), 0.0);

	// Evaluate phi(alpha) and derivative phi'(alpha)
	// phi(alpha) = f(x_k - alpha H_grad)
	// phi'(alpha) = -H_grad' \nabla_x f(x_k - alpha H_grad)
	phi_fval = prob->Eval(xalpha, xalphagrad);
	xalphaobj = phi_fval;	// save phi(alpha)
	xalpha_val = alpha;	// save alpha

	// Univariate derivative is projection onto ascent direction
	phi_grad = -std::inner_product(xalphagrad.begin(), xalphagrad.end(),
		H_grad.begin(), 0.0);
	evaluation_count += 1;
#if 0
	std::cout << "   phi(" << alpha << ") = " << phi_fval << ", grad "
		<< phi_grad << std::endl;
#endif
}

}

