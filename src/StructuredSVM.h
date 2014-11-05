
#ifndef GRANTE_STRUCTUREDSVM_H
#define GRANTE_STRUCTUREDSVM_H

#include <vector>
#include <string>
#include <tr1/unordered_map>

#include "FactorGraphModel.h"
#include "ParameterEstimationMethod.h"
#include "FunctionMinimizationProblem.h"
#include "StochasticFunctionMinimizationProblem.h"
#include "InferenceMethod.h"
#include "Likelihood.h"
#include "StructuredLossFunction.h"

namespace Grante {

/* Structured Support Vector Machine
 *
 * Only the linear margin-rescaling variant is implemented.  The problem
 * solved approximately is the following:
 *
 * min_w  Omega(w) + (C/N) \sum_{n=1}^{N} \xi_n
 * sb.t.  E(y_n;x,w) + Delta(y,y_n) <= E(y;x,w) + \xi_n,
 *                 for all n=1,...,N, for all y in Y(n).
 *
 * The structured loss function Delta is provided by the user, as are methods
 * to perform (approximate) MAP inference on the training instances.  The
 * regularization parameter C sets the trade-off between the empirical loss
 * and the regularization strength.  Large values of C (eg. 1000) are harder
 * to optimize and might overfit the training set, but lead to low training
 * errors.  Small values of C (eg. 1e-4) lead to strongly regularized learning
 * problems, possibly underfitting the training data.
 *
 * Omega(w) is -log p(w) of the prior.
 */
class StructuredSVM : public ParameterEstimationMethod {
public:
	// fg_model: Factor graph model the instances realize.
	// ssvm_C: Structured SVM regularization parameter.
	// opt_method: "stochastic", "batch", "bmrm", or "bmrm2".  Only
	//    "stochastic" and "bmrm2" are robust, and "bmrm2" is recommended.
	//
	// NOTE: you must add a prior for all factor types before calling Train.
	StructuredSVM(FactorGraphModel* fg_model, double ssvm_C,
		const std::string& opt_method = "bmrm");
	virtual ~StructuredSVM();

	// (ParameterEstimationMethod interface)
	// Using the default StructuredHammingLoss
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*> inference_methods);

	// Initialize training data, structured loss functions and MAP inference
	// methods.
	void SetupTrainingData(
		const std::vector<FactorGraph*>& instances,
		const std::vector<StructuredLossFunction*>& loss,
		const std::vector<InferenceMethod*>& inference_methods);

	virtual double Train(double conv_tol, unsigned int max_iter = 0);

private:
	std::vector<FactorGraph*> training_instances;
	std::vector<StructuredLossFunction*> loss_functions;
	std::vector<InferenceMethod*> inference_methods;

	// Utility storage to support ParameterEstimationMethod interface
	std::vector<FactorGraph*> t_instances;
	std::vector<StructuredLossFunction*> t_loss;

	// Structured SVM regularization parameter C
	double ssvm_C;
	std::string opt_method;

	class StructuredSVMProblem {
	public:
		StructuredSVMProblem(StructuredSVM* ssvm_base);
		~StructuredSVMProblem();

		// Evaluate loss gradient for all instances
		double EvaluateLossGradient();
		// Evaluate loss gradient for a single instance
		double EvaluateLossGradient(Likelihood& lh, unsigned int n);
		double AddRegularizer(double scale = 1.0);

		// Input: u (stored in model weights),
		// Output: [Of,Og] (Of returned and Og in parameter_gradient)
		double EvaluateFenchelDual(void);
		void ClearParameterGradient();
		void LinearToFactorWeights(const std::vector<double>& x);
		void FactorWeightsToLinear(std::vector<double>& x);
		void ParameterGradientToLinear(std::vector<double>& grad);
		unsigned int Dimensions() const;

		StructuredSVM* Base();

	private:
		StructuredSVM* ssvm_base;
		unsigned int dim;

		std::vector<std::string> parameter_order;
		std::tr1::unordered_map<std::string, std::vector<double> >
			parameter_gradient;
	};

	// Stochastic minimization problem
	class StochasticStructuredSVMProblem :
		public StochasticFunctionMinimizationProblem {
	public:
		StochasticStructuredSVMProblem(StructuredSVMProblem* ssvm_prob);
		virtual ~StochasticStructuredSVMProblem();

		virtual double Eval(unsigned int sample_id,
			const std::vector<double>& x, std::vector<double>& grad);
		virtual unsigned int Dimensions() const;
		virtual size_t NumberOfElements() const;
		virtual void ProvideStartingPoint(std::vector<double>& x0) const;

	private:
		StructuredSVMProblem* ssvm_prob;
		size_t elements_count;
	};

	// BMRM dual structured SVM formulation (equivalent to Joachims' the
	// 1-slack dual), described in
	//
	// [Tao2009]  Choon Hui Teo, SVN Vishwanathan, Alex Smola, Quoc V. Le,
	// "Bundle Methods for Regularized Risk Minimization", JMLR 2009.
	//
	// Primal:   min_w       Omega(w) + max_i [<a_i,w> + b_i]
	// Dual:     max_\alpha  -Omega^*(-A'\alpha) + b'*\alpha
	//           sb.t.       \alpha \in \Delta^d,
	//
	// where \Delta^d is the d-dimensional canonical simplex, Omega(w) is the
	// primal regularizer (- log p(w)), and Omega^*(u) is the Fenchel dual of
	// Omega(w).
	class BMRM2StructuredSVMProblem {
	public:
		BMRM2StructuredSVMProblem(StructuredSVMProblem* ssvm_prob);
		virtual ~BMRM2StructuredSVMProblem();

		void AddCurrentSubgradient(double R_emp);

		// Solve the dual BMRM subproblem using the spectral projected
		// gradient algorithm, SPG2, described in
		//
		// [Birgin1999] Ernesto G. Birgin, Jose Mario Martinez, Marcos Raydan,
		//    "Nonmonotone Spectral Projected Gradient Methods on Convex
		//    Sets", 1999.
		//
		// The approximately optimal primal/dual solution is returned in
		// w_opt and alpha_opt, respectively.  The convergence tolerance is
		// measured on the primal-dual gap, and the actual achieved
		// primal-dual gap is returned in pd_gap.  Note that if the maximum
		// number of iterations max_iter is exceeded we have
		// pd_gap > conv_tol.
		double OptimizeDual(std::vector<double>& w_opt,
			std::vector<double>& alpha_opt, double conv_tol,
			unsigned int max_iter, double& pd_gap, bool verbose);

	private:
		StructuredSVMProblem* ssvm_prob;

		// Cutting plane model
		std::vector<std::vector<double> > At;
		std::vector<double> bt;

		// Evaluate neg_A_alpha = -A alpha
		void Eval_neg_A_alpha(const std::vector<double>& alpha,
			std::vector<double>& neg_A_alpha) const;
		// Evaluate neg_AT_u = -A'u
		void Eval_neg_AT_u(const std::vector<double>& u,
			std::vector<double>& neg_AT_u) const;

		// [Chen2011], Yunmei Chen, Xiaojing Ye, "Projection Onto A Simplex",
		// arxiv.org, 2011.
		// This method has also been described by Duchi and others, but I did
		// not locate the origins.
		void ProjectOntoSimplex(std::vector<double>& alpha) const;

		double Eval(const std::vector<double>& alpha,
			std::vector<double>& alpha_grad,
			std::vector<double>& w_opt);
		double EvalPrimal(const std::vector<double>& w_opt);
	};

	// Train using the BMRM formulation,
	// Choon Hui Teo, SVN Vishwanathan, Alex Smola, Quoc V. Le, "Bundle
	// Methods for Regularized Risk Minimization", JMLR 2009.
	double TrainBMRM(StructuredSVMProblem* ssvm_prob,
		double conv_tol, unsigned int max_iter);
};

}

#endif

