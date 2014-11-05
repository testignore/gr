
#ifndef GRANTE_CONTRASTIVEDIVERGENCETRAINING_H
#define GRANTE_CONTRASTIVEDIVERGENCETRAINING_H

#include <vector>
#include <tr1/unordered_map>

#include "ParameterEstimationMethod.h"
#include "FactorGraphObservation.h"
#include "FactorGraphPartialObservation.h"
#include "ContrastiveDivergence.h"

namespace Grante {

/* Contrastive divergence training for fully and partially observed data.
 */
class ContrastiveDivergenceTraining : public ParameterEstimationMethod {
public:
	// cd_k: number of Gibbs sweeps to estimate model distributions sample,
	//    must be >0.
	// mini_batch_size: number of instances to use for expectation.  If zero
	//    is given, all instances are used.  Typical values are 10, 100, 0.
	// stepsize: >0 constant stepsize.
	ContrastiveDivergenceTraining(FactorGraphModel* fg_model,
		unsigned int cd_k, unsigned int mini_batch_size,
		double stepsize = 1.0e-2);
	virtual ~ContrastiveDivergenceTraining();

	// No inference method needed here, and we assume inference_methods is
	// empty.
	virtual void SetupTrainingData(
		const std::vector<labeled_instance_type>& training_data,
		const std::vector<InferenceMethod*> inference_methods);

	// No inference method needed.
	void SetupPartiallyObservedTrainingData(
		const std::vector<partially_labeled_instance_type>&
			pobs_training_data);

	// TODO: create a convergence test, right now conv_tol is ignored
	// max_iter: Number of epochs over the training set.
	virtual double Train(double conv_tol, unsigned int max_iter = 0);

private:
	unsigned int mini_batch_size;
	ContrastiveDivergence cd;
	double stepsize;

	std::vector<partially_labeled_instance_type> pobs_training_data;

	void GradientSetup(std::tr1::unordered_map<std::string,
		std::vector<double> >& parameter_gradient) const;
	void GradientScale(std::tr1::unordered_map<std::string,
		std::vector<double> >& parameter_gradient, double scale) const;
};

}

#endif

