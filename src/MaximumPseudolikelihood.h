
#ifndef GRANTE_MAXIMUMPSEUDOLIKELIHOOD_H
#define GRANTE_MAXIMUMPSEUDOLIKELIHOOD_H

#include <tr1/unordered_map>
#include <vector>

#include "MaximumLikelihood.h"
#include "FactorGraphUtility.h"

namespace Grante {

class MaximumPseudolikelihood : public MaximumLikelihood {
public:
	explicit MaximumPseudolikelihood(FactorGraphModel* fg_model);
	virtual ~MaximumPseudolikelihood();

	class MPLEProblem;
	MaximumPseudolikelihood::MPLEProblem* GetLearnProblem();

	void SetOptimizationMethod(
		MaximumLikelihood::MLEOptimizationMethod opt_method);

	virtual double Train(double conv_tol, unsigned int max_iter = 0);

	// FIXME
//protected:
	MaximumLikelihood::MLEOptimizationMethod opt_method;

	class MPLEProblem : public MLEProblem {
	public:
		MPLEProblem(MaximumPseudolikelihood* mple_base);
		virtual ~MPLEProblem();

	protected:
		MaximumPseudolikelihood* mple_base;

		virtual double EvaluateLikelihoodGradient(
			std::tr1::unordered_map<std::string, std::vector<double> >&
				parameter_gradient);

	private:
		// Keep an initialized Gibbs sampler for each training factor graph.
		// This is used to compute single-site conditional distributions.
		std::vector<FactorGraphUtility*> fgu;
	};
};

}

#endif

