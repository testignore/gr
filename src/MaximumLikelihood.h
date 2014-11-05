
#ifndef GRANTE_MAXIMUMLIKELIHOOD_H
#define GRANTE_MAXIMUMLIKELIHOOD_H

#include <vector>
#include <string>
#include <tr1/unordered_map>

#include "ParameterEstimationMethod.h"
#include "CompositeMinimizationProblem.h"
#include "InferenceMethod.h"

namespace Grante {

class MaximumLikelihood : public ParameterEstimationMethod {
public:
	explicit MaximumLikelihood(FactorGraphModel* fg_model);

	virtual ~MaximumLikelihood();

	enum MLEOptimizationMethod {
		LBFGSMethod = 0,
		SimpleGradientMethod,
		BarzilaiBorweinMethod,
		FISTAMethod,
	};
	void SetOptimizationMethod(MLEOptimizationMethod opt_method);

	// NOTE: training using multiple cores (OpenMP) is only safe if
	// each factor graph is unique in the training set
	virtual double Train(double conv_tol, unsigned int max_iter = 0);


//protected:
	MLEOptimizationMethod opt_method;

	// Function minimization definition of Maximum Likelihood Estimation (MLE)
	// The function minimized is:
	// f(w) = - 1/N \sum_{n=1}^{N} log p(x_n;w) - 1/N log p(w)
	class MLEProblem : public CompositeMinimizationProblem {
	public:
		MLEProblem(MaximumLikelihood* mle_base);
		virtual ~MLEProblem();

		virtual double EvalF(const std::vector<double>& x,
			std::vector<double>& grad);
		virtual double EvalG(const std::vector<double>& x,
			std::vector<double>& subgrad);
		virtual void EvalGProximalOperator(const std::vector<double>& u,
			double L, std::vector<double>& wprox) const;

		virtual unsigned int Dimensions() const;
		virtual void ProvideStartingPoint(std::vector<double>& x0) const;
		void LinearToFactorWeights(const std::vector<double>& x);

	protected:
		MaximumLikelihood* mle_base;
		unsigned int dim;
		std::vector<std::string> parameter_order;

		virtual double EvaluateLikelihoodGradient(
			std::tr1::unordered_map<std::string, std::vector<double> >&
				parameter_gradient);

	private:
		void SetupParameterGradient(std::tr1::unordered_map<std::string,
			std::vector<double> >& parameter_gradient) const;
		void AddParameterGradient(const std::tr1::unordered_map<std::string,
			std::vector<double> >& parameter_gradient,
			std::vector<double>& grad) const;
	};

	// FIXME: temporary
	MLEProblem* GetLearnProblem();

};

}

#endif

