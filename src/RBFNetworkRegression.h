
#ifndef GRANTE_RBFNETWORKREGRESSION_H
#define GRANTE_RBFNETWORKREGRESSION_H

#include <vector>

#include "FunctionMinimizationProblem.h"
#include "RBFNetwork.h"

namespace Grante {

class RBFNetworkRegression {
public:
	RBFNetworkRegression(unsigned int N, unsigned int d);
	RBFNetworkRegression(const std::vector<std::vector<double> >& prototypes);

	void FixBeta(double beta);

	// Minimize L2 loss over samples X and targets Y:
	//   \sum_i |f(x_i)-y_i|^2
	double Fit(const std::vector<std::vector<double> >& X,
		const std::vector<double>& Y, double conv_tol = 1.0e-5,
		unsigned int max_iter = 5000);

	// Evaluate learned regression function
	double Evaluate(const std::vector<double>& x) const;

private:
	size_t N;
	size_t d;

	RBFNetwork rbfnet;
	std::vector<double> param;

	const std::vector<std::vector<double> >* sample_X;
	const std::vector<double>* sample_Y;

	class RBFL2Problem : public FunctionMinimizationProblem {
	public:
		RBFL2Problem(RBFNetworkRegression* reg_base);
		virtual ~RBFL2Problem();

		virtual double Eval(const std::vector<double>& x,
			std::vector<double>& grad);
		virtual unsigned int Dimensions() const;
		virtual void ProvideStartingPoint(std::vector<double>& x0) const;

	protected:
		RBFNetworkRegression* reg_base;
	};
};

}

#endif

