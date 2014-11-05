
#ifndef GRANTE_FUNCTIONMINIMIZATION_H
#define GRANTE_FUNCTIONMINIMIZATION_H

#include <vector>
#include <list>

#include <boost/tuple/tuple.hpp>

#include "FunctionMinimizationProblem.h"

namespace Grante {

class FunctionMinimization {
public:
	// Minimize a smooth unconstrained function by means of the
	// Barzilai-Borwein method.
	//
	// Note: the Barzilai-Borwein method is only known to always converge on
	// strictly convex quadratic problems.  When used on convex problems
	// 'similar to quadratic problem' it works very well empirically.  There
	// are known examples of non-convex and ill-conditioned convex problems on
	// which it is known to diverge.
	//
	// prob: The minimization problem, see FunctionMinimizationProblem.h.
	// x_opt: The resulting approximately optimal solution vector.  Does not
	//    have to be initialized.
	// conv_tol: The convergence tolerance as measured by the Euclidean norm
	//    of the gradient.
	// max_iter: Maximum number of iterations or zero for no limit.
	// verbose: If true some statistics are printed during optimization.
	//
	// The return value is the achieved objective of x_opt.
	static double BarzilaiBorweinMinimize(FunctionMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol, unsigned int max_iter = 0,
		bool verbose = true);

	static double LimitedMemoryBFGSMinimize(FunctionMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol, unsigned int max_iter = 0,
		bool verbose = true, unsigned int lbfgs_m = 50);

	// Minimize a continous (not necessarily differentiable) unconstrained
	// function by means of the subgradient method.
	//
	// Note: this is useful when non-differentiable priors (Laplace) are used
	// or when a non-differentiable loss function is used.
	//
	// prob: The minimization problem, see FunctionMinimizationProblem.h.
	//    The problem is only required to return a subgradient.
	// x_opt: The resulting approximately optimal solution vector.  Does not
	//    have to be initialized.
	// conv_tol: The convergence tolerance as measured by the Euclidean norm
	//    of the subgradient.
	// max_iter: The maximum number of iterations.  If zero (default), there
	//    is no limit.
	// verbose: If true some statistics are printed during optimization.
	static double SubgradientMethodMinimize(FunctionMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol, unsigned int max_iter = 0,
		bool verbose = true);

	// Stepsize reducing gradient method for convex function minimization.
	// It is based on the convergent simple method of exercise 1.2.20 in
	// Bertsekas, "Nonlinear Programming", 2nd Edition.
	static double GradientMethodMinimize(
		FunctionMinimizationProblem& prob,
		std::vector<double>& x_opt, double conv_tol, unsigned int max_iter,
		bool verbose = true);

	// Numerically test the gradient of a function by finite difference
	// approximations.
	//
	// prob: The minimization problem.
	// x_range: The initial point provided by the problem is perturbed in each
	//    dimension with an individual real sampled uniformly from
	//    [-x_range, x_range].
	// test_count: Number of tests to perform.
	// dim_eps: The numeric perturbation used to compute the finite-difference
	//    approximation.
	// grad_tol: The absolute tolerated difference per dimension.
	//
	// Return true in case the derivative seems to be correct.
	// Return false if the numeric approximation disagrees.
	static bool CheckDerivative(FunctionMinimizationProblem& prob,
		double x_range, unsigned int test_count = 100,
		double dim_eps = 1e-8, double grad_tol = 1e-5);

private:
	// list of (s_k, y_k, rho_k)
	typedef std::list<boost::tuple<std::vector<double>, std::vector<double>, double> >
		lbfgs_mem_type;

	static double EuclideanNorm(const std::vector<double>& vec);

	class WolfeLineSearch {
	public:
		// TODO doc
		// x0: Current iterate x_k,
		// x0_grad: \nabla_x f(x_k),
		// H_grad: H \nabla_x f(x_k), ascent direction
		// x0_fval: f(x_k)
		//
		// c1: constant related to Armijo condition (larger: stricter)
		// c2: constant related to curvature condition (lower: stricter),
		//   need to satisfy 0 < c1 < c2 < 1.0.
		WolfeLineSearch(FunctionMinimizationProblem* prob,
			const std::vector<double>& x0,
			const std::vector<double>& x0_grad,
			const std::vector<double>& H_grad, double x0_fval,
			double c1 = 1.0e-4, double c2 = 0.9);

		// The initial stepsize tried must be given in alpha.
		// Returns the number of evaluations and alpha in its argument.
		unsigned int ComputeStepLength(double& alpha);
		unsigned int ComputeStepLengthUpdate(
			std::vector<double>& x_out, std::vector<double>& grad_out,
			double& fval_out, double& alpha_out);

	private:
		FunctionMinimizationProblem* prob;
		const std::vector<double> x0;
		const std::vector<double> x0_grad;
		const std::vector<double> H_grad;
		double x0_fval;
		double x0_phi_grad;

		unsigned int evaluation_count;

		// All information from the function evaluation if xalpha_val != nan
		std::vector<double> xalpha;
		std::vector<double> xalphagrad;
		double xalphaobj;
		double xalpha_val;

		double c1;
		double c2;

		void Evaluate(double alpha, double& phi_fval, double& phi_grad);
		double Zoom(double alpha_lo, double alpha_hi, double alpha_lo_fval,
			double alpha_hi_fval, double alpha_lo_grad);
	};

	class SimpleLineSearch {
	public:
		// A simple backtracking line-search method.  This is more robust
		// than the Wolfe line search.
		//
		// x0: Current iterate x_k,
		// x0_grad: \nabla_x f(x_k),
		// H_grad: H \nabla_x f(x_k), ascent direction
		// x0_fval: f(x_k)
		SimpleLineSearch(FunctionMinimizationProblem* prob,
			const std::vector<double>& x0,
			const std::vector<double>& x0_grad,
			const std::vector<double>& H_grad, double x0_fval);

		// The initial stepsize tried must be given in alpha.
		// Returns the number of evaluations and alpha in its argument.
		unsigned int ComputeStepLength(double& alpha);
		unsigned int ComputeStepLengthUpdate(
			std::vector<double>& x_out, std::vector<double>& grad_out,
			double& fval_out, double& alpha_out);

	private:
		FunctionMinimizationProblem* prob;
		const std::vector<double> x0;
		const std::vector<double> x0_grad;
		const std::vector<double> H_grad;
		double x0_fval;
		double x0_phi_grad;

		unsigned int evaluation_count;

		// All information from the function evaluation if xalpha_val != nan
		std::vector<double> xalpha;
		std::vector<double> xalphagrad;
		double xalphaobj;
		double xalpha_val;

		void Evaluate(double alpha, double& phi_fval, double& phi_grad);
	};
};

}

#endif

