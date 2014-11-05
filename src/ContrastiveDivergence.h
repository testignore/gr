
#ifndef GRANTE_CONTRASTIVEDIVERGENCE_H
#define GRANTE_CONTRASTIVEDIVERGENCE_H

#include <vector>
#include <string>
#include <tr1/unordered_map>

#include "FactorGraphModel.h"
#include "FactorGraph.h"
#include "FactorGraphObservation.h"
#include "FactorGraphPartialObservation.h"

namespace Grante {

/* Helper class to evaluate contrastive divergence expressions on fully and
 * partially observed instances.
 *
 * References
 * [Hinton2002] Geoffrey Hinton,
 *    "Training products of experts by minimizing contrastive divergence",
 *    Neural Computation, 14(8): 1771-1800, 2002.
 *
 * [CarreiraPerpinan2005] Miguel Carreira-Perpinan, Geoffrey Hinton,
 *    "On contrastive divergence learning", AISTATS 2005.
 *
 * [He2004], He, Zemel, Carreira-Perpinan,
 *    "Multiscale conditional random fields for image labeling", CVPR 2004.
 */
class ContrastiveDivergence {
public:
	// cd_k: Number of Gibbs sweeps to obtain a sample from the model
	//    distribution, typical values are 1, 10, 100.
	ContrastiveDivergence(FactorGraphModel* fg_model, unsigned int cd_k);

	// Add the following contrastive divergence gradient to the
	// parameter_gradient,
	//     \nabla_w E(y_obs,x,w) - \nabla_w E(y_model,x,w),
	// where y_obs is the ground truth observation and y_model is a sample
	// from the model distribution obtained by a small number of Gibbs sweeps.
	//
	// parameter_gradient: set of model parameter vectors to add the gradient
	//    to,
	// fg: factor graph model with conditional data observations,
	// obs: fully observed ground truth, must be a discrete observation for
	//    now.
	void ComputeGradientFullyObserved(
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient, const FactorGraph* fg,
		const FactorGraphObservation* obs) const;

	// As for the fully observed case, but using partial observations.  Add
	// the following contrastive divergence gradient to the
	// parameter_gradient,
	//     \nabla_w E(y_pobs,x,w) - \nabla_w E(y_model,x,w),
	// where y_pobs is a label obtained by fixing the ground truth
	// observations and sampling the hidden variables, and y_model is a full
	// sample from the model distribution.  Both samples are obtained using
	// a small number of Gibbs sweeps.
	//
	// parameter_gradient: set of model parameter vectors to add the gradient
	//    to,
	// fg: factor graph model with conditional data observations,
	// pobs: partially observed ground truth, must be a discrete observation
	//    for now.
	void ComputeGradientPartiallyObserved(
		std::tr1::unordered_map<std::string, std::vector<double> >&
			parameter_gradient, const FactorGraph* fg,
		const FactorGraphPartialObservation* pobs) const;

private:
	FactorGraphModel* model;
	unsigned int cd_k;

	// Temporary marginal distributions, key: size
	mutable std::tr1::unordered_map<size_t, std::vector<double> >
		temp_marginals;

	// Add scale*(\nabla_w E(y,x,w)) to parameter_gradient.
	void AddBackwardMap(std::tr1::unordered_map<std::string,
		std::vector<double> >& parameter_gradient, const FactorGraph* fg,
		const std::vector<unsigned int>& y, double scale) const;
};

}

#endif

