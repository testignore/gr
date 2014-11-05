
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <limits>
#include <ctime>
#include <cmath>
#include <cassert>

#include <boost/timer.hpp>
#include <boost/random.hpp>

#include "StochasticFunctionMinimization.h"

namespace Grante {

double StochasticFunctionMinimization::StochasticSubgradientMethodMinimize(
	StochasticFunctionMinimizationProblem& prob, std::vector<double>& x_opt,
	double conv_tol, unsigned int max_epochs, bool verbose) {
	unsigned int dim = prob.Dimensions();
	std::vector<double> grad(dim, 0.0);

	// Initialize x
	std::vector<double> x(dim);
	prob.ProvideStartingPoint(x);

	// Random instance generator
	size_t N = prob.NumberOfElements();
	assert(N > 0);
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_int<unsigned int> rdestd(0, static_cast<unsigned int>(N-1));
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_n(rgen, rdestd);

	// Optimize a given number of epochs
	boost::timer total_timer;
	std::vector<double> avg_grad(dim, 0.0);
	double lambda = 1.0;	// (should be lambda=1/C)
	double avg_obj = 0.0;
	for (unsigned int epoch = 0; max_epochs == 0 || epoch < max_epochs;
		++epoch) {
		avg_obj = 0.0;
		std::fill(avg_grad.begin(), avg_grad.end(), 0.0);

		// Choose epoch-wide step size
		double alpha = 1.0 / (static_cast<double>(epoch + 1) * lambda);

		// Optimize by sampling instances
		for (size_t n = 0; n < N; ++n) {
			unsigned int id = rand_n();
			// Update average objective and averaged gradient of this epoch
			avg_obj += prob.Eval(id, x, grad);
			std::transform(grad.begin(), grad.end(), avg_grad.begin(),
				avg_grad.begin(), std::plus<double>());

			// Perform incremental subgradient update
			for (unsigned int d = 0; d < dim; ++d)
				x[d] -= alpha * grad[d];
		}
		// Compute mean gradient and estimated objective
		double avg_grad_norm = 0.0;
		for (unsigned int d = 0; d < dim; ++d)
			avg_grad_norm += avg_grad[d]*avg_grad[d];
		avg_grad_norm = std::sqrt(avg_grad_norm);

		// Output statistics
		if (verbose && (epoch % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time        avg_obj  |avg_grad|" << std::endl;
		}
		if (verbose) {
			std::ios_base::fmtflags original_format = std::cout.flags();
			std::streamsize original_prec = std::cout.precision();

			// Iteration
			std::cout << std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::adjustfield)
				<< std::setw(6) << epoch << "  ";
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
				<< std::setw(7) << avg_obj << "   ";
			// Gradient norm
			std::cout << std::setiosflags(std::ios::scientific)
				<< std::setprecision(2)
				<< std::resetiosflags(std::ios::showpos)
				<< std::setiosflags(std::ios::left) << avg_grad_norm;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}

		// Convergence check
		if (avg_grad_norm < conv_tol)
			break;
	}

	x_opt = x;
	return (avg_obj);	// This is not exact, but stochastic anyway
}

}

