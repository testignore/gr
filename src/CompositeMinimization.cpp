
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <limits>

#include <boost/timer.hpp>

#include "CompositeMinimization.h"

namespace Grante {

double CompositeMinimization::FISTAMinimize(
	CompositeMinimizationProblem& prob, std::vector<double>& x_opt,
	double conv_tol, unsigned int max_iter, bool verbose) {
	double Lub = 0.25;	// Lipschitz upper bound estimate for the gradient
	double eta = 2.0;	// Lipschitz bound scale factor

	// Initialize iterates
	unsigned int dim = prob.Dimensions();
	x_opt.resize(dim);
	prob.ProvideStartingPoint(x_opt);
	std::vector<double> xprev(x_opt);
	std::vector<double> x_fgrad_dummy;
	std::vector<double> y(x_opt);
	std::vector<double> y_fgrad(dim, 0.0);
	std::vector<double> u(dim, 0.0);

	// Convergence criterion
	double conv = std::numeric_limits<double>::infinity();
	double t = 1.0;

	// Objective F(x)=f(x)+g(x)
	double Fval = std::numeric_limits<double>::signaling_NaN();
	boost::timer total_timer;
	unsigned int lip_iter = 0;
	for (unsigned int iter = 0; max_iter == 0 || iter < max_iter; ++iter) {
		double y_fval = prob.EvalF(y, y_fgrad);
		double obj = y_fval + prob.EvalG(y, x_fgrad_dummy);

		// Verbose output
		if (verbose && (iter % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      objective        conv  lipiter"
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
			// Convergence criterion
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << conv << "   ";
			// Lipschitz iterations
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << lip_iter;

			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		// Backtracking
		lip_iter = 0;
		do {
			lip_iter += 1;

			// Solve (2.6) in Beck and Teboulle
			std::transform(y.begin(), y.end(), y_fgrad.begin(), u.begin(),
				[Lub](double y_e, double y_fgrad_e) -> double {
					return (y_e - y_fgrad_e/Lub); });
			// update x_opt
			prob.EvalGProximalOperator(u, Lub, x_opt);

			// Evaluate F(x)
			Fval = prob.EvalF(x_opt, x_fgrad_dummy);
			Fval += prob.EvalG(x_opt, x_fgrad_dummy);

			// Evaluate Q_L(x,y)
			double Qval = y_fval + prob.EvalG(x_opt, x_fgrad_dummy);
			for (unsigned int di = 0; di < dim; ++di) {
				double xsuby = x_opt[di] - y[di];
				Qval += xsuby*y_fgrad[di];
				Qval += 0.5*Lub*xsuby*xsuby;
			}

			// Sufficient upper bound on the Lipschitz constant?
			if (Fval <= Qval)
				break;

			Lub *= eta;	// Increase upper bound estimate
		} while (true);

		// Update t sequence
		double tprev = t;
		t = 0.5*(1.0 + std::sqrt(1.0 + 4.0*tprev*tprev));

		// Perform averaged step: update y
		std::transform(x_opt.begin(), x_opt.end(), xprev.begin(),
			y.begin(), [t,tprev](double xe, double xle) -> double {
				return (xe + ((tprev-1.0)/t)*(xe-xle)); });
		// conv=norm(x_opt-xprev)
		conv = 0.0;
		for (unsigned int di = 0; di < dim; ++di)
			conv += (x_opt[di]-xprev[di]) * (x_opt[di]-xprev[di]);
		conv = std::sqrt(conv);
		if (conv <= conv_tol) {
			if (verbose) {
				std::cout << "Converged with tolerance " << conv
					<< " (<= " << conv_tol << ")." << std::endl;
			}
			return (Fval);
		}

		std::copy(x_opt.begin(), x_opt.end(), xprev.begin());
	}
	return (Fval);	// objective of x_opt
}

}

