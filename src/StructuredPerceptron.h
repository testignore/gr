
#ifndef GRANTE_STRUCTUREDPERCEPTRON_H
#define GRANTE_STRUCTUREDPERCEPTRON_H

#include <vector>

#include "ParameterEstimationMethod.h"
#include "FactorGraphModel.h"
#include "InferenceMethod.h"
#include "Likelihood.h"

namespace Grante {

/* (Averaged) Structured Perceptron
 */
class StructuredPerceptron : public ParameterEstimationMethod {
public:
	StructuredPerceptron(FactorGraphModel* fg_model, bool do_averaging = true,
		bool verbose = true);
	virtual ~StructuredPerceptron();

	// Priors are not supported, this method will abort
	virtual void AddPrior(const std::string& factor_type, Prior* prior);

	// conv_tol is ignored by this parameter learning method.
	virtual double Train(double conv_tol, unsigned int max_epochs = 100);

private:
	// Whether to perform averaging over weight vectors of all iterates
	bool do_averaging;
	bool verbose;
	Likelihood lh;

	// Gradient updates
	std::vector<std::string> parameter_order;
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_averaged;

	// Perceptron update for a single sample
	bool ProcessSample(unsigned int sample_id);
	// Apply parameter_gradient to factor graph weights
	void UpdateFactorWeights();
	// Update averaged weight vector
	void UpdateAveragedParameters(double old_factor, double new_factor);
	void SetFactorWeights();
	// Initialize and set the parameter gradient to zero
	void ClearParameterGradient();
};

}

#endif

