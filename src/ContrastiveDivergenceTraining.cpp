
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <boost/lambda/lambda.hpp>

#include "RandomSource.h"
#include "ContrastiveDivergenceTraining.h"

using namespace boost::lambda;

namespace Grante {

ContrastiveDivergenceTraining::ContrastiveDivergenceTraining(
	FactorGraphModel* fg_model, unsigned int cd_k,
	unsigned int mini_batch_size, double stepsize)
	: ParameterEstimationMethod(fg_model), mini_batch_size(mini_batch_size),
		cd(fg_model, cd_k), stepsize(stepsize) {
	assert(cd_k > 0);
}

ContrastiveDivergenceTraining::~ContrastiveDivergenceTraining() {
	// nothing to do
}

void ContrastiveDivergenceTraining::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	assert(training_data.size() > 0);
	assert(inference_methods.empty());
	// Call parent
	ParameterEstimationMethod::SetupTrainingData(training_data,
		inference_methods);
}

void ContrastiveDivergenceTraining::SetupPartiallyObservedTrainingData(
	const std::vector<partially_labeled_instance_type>& pobs_training_data) {
	assert(training_data.empty());
	assert(pobs_training_data.size() > 0);

	this->pobs_training_data = pobs_training_data;
}

double ContrastiveDivergenceTraining::Train(double conv_tol,
	unsigned int max_iter) {
	assert(pobs_training_data.empty() ^ training_data.empty());

	size_t instance_count = 0;
	if (training_data.empty() == false) {
		// fully observed training data
		instance_count = training_data.size();
	} else {
		// partially observed training data
		instance_count = pobs_training_data.size();
	}

	// Global scaling constants
	double scale = 1.0 / static_cast<double>(instance_count);

	// Mini batch indices
	std::vector<unsigned int> instance_idx(instance_count);
	for (unsigned int ni = 0; ni < instance_count; ++ni)
		instance_idx[ni] = ni;

	// Parameter gradient
	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	double mean_nabla_w_norm = 0.0;
	for (unsigned int iter = 1; max_iter == 0 || (iter <= max_iter); ++iter) {
		// 1. Setup mini-batches
		RandomSource::ShuffleRandom(instance_idx);

		size_t mb_count = 1;	// number of mini batches
		if (mini_batch_size > 0) {
			mb_count = instance_idx.size() / mini_batch_size;
			if (instance_idx.size() % mini_batch_size > 0)
				mb_count += 1;
		}

		// Process all mini batches
		mean_nabla_w_norm = 0.0;
		for (unsigned int mbi = 0; mbi < mb_count; ++mbi) {
			GradientSetup(parameter_gradient);	// reset gradient

			size_t mb_start = 0;
			size_t mb_end = instance_idx.size();
			if (mb_count > 1) {
				mb_start = mbi*mini_batch_size;
				mb_end = (mbi+1)*mini_batch_size;
				if (mb_end > instance_idx.size())
					mb_end = instance_idx.size();
			}
			assert(mb_end > mb_start);

			// composite scaling term for energy gradients
			double scale_mb = scale;
			if (mb_count > 1)
				scale_mb = 1.0 / static_cast<double>(mb_end-mb_start);

			// 2a. Compute CD parameter gradient, scaled, in order to obtain
			//     an unbiased estimator of the full 1/N \sum_{n=1}^N
			//     CD-gradient.
			for (size_t mb_si = mb_start; mb_si < mb_end; ++mb_si) {
				unsigned int ni = instance_idx[mb_si];
				if (training_data.empty() == false) {
					// fully observed
					cd.ComputeGradientFullyObserved(parameter_gradient,
						training_data[ni].first, training_data[ni].second);
				} else {
					// partially observed
					cd.ComputeGradientPartiallyObserved(parameter_gradient,
						pobs_training_data[ni].first,	// fg
						pobs_training_data[ni].second);	// pobs
				}
			}
			GradientScale(parameter_gradient, scale_mb);

			// 2b. Compute parameter prior gradient: (1/N) \nabla_w -log p(w)
			for (std::multimap<std::string, Prior*>::const_iterator
				prior = priors.begin(); prior != priors.end(); ++prior) {
				FactorType* ft = fg_model->FindFactorType(prior->first);
				prior->second->EvaluateNegLogP(ft->Weights(),
					parameter_gradient[prior->first], scale);
			}

			// 3. Select step size
			double alpha = stepsize;
			//double alpha = 1.0e-1 / std::sqrt(static_cast<double>(iter));

			// 4. Update parameters: w <-- w - alpha * \nabla_w E
			double nabla_w_norm = 0.0;	// norm of approximate gradient
			for (std::tr1::unordered_map<std::string,
				std::vector<double> >::iterator pgi =
				parameter_gradient.begin(); pgi != parameter_gradient.end();
				++pgi) {
				nabla_w_norm += std::inner_product(pgi->second.begin(),
					pgi->second.end(), pgi->second.begin(), 0.0);

				// Update model weights
				FactorType* ft = fg_model->FindFactorType(pgi->first);
				std::transform(ft->Weights().begin(), ft->Weights().end(),
					pgi->second.begin(), ft->Weights().begin(),
					_1 - alpha * _2);
			}
			nabla_w_norm = std::sqrt(nabla_w_norm);
			mean_nabla_w_norm += nabla_w_norm;

#if 0
			// TODO
			std::cout << "cd iter " << iter << ", mb " << mbi
				<< " of " << mb_count << ", |nabla w| " << nabla_w_norm
				<< std::endl;
#endif
		}
		mean_nabla_w_norm /= static_cast<double>(mb_count);
		std::cout << "cd iter " << iter << " completed, mean |nabla w| "
			<< mean_nabla_w_norm << std::endl;
	}
	return (mean_nabla_w_norm);
}

void ContrastiveDivergenceTraining::GradientSetup(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient) const {
	const std::vector<FactorType*>& factor_types = fg_model->FactorTypes();
	for (std::vector<FactorType*>::const_iterator fti = factor_types.begin();
		fti != factor_types.end(); ++fti) {
		const std::string& ftname = (*fti)->Name();
		std::tr1::unordered_map<std::string, std::vector<double> >::iterator
			pgi = parameter_gradient.find(ftname);
		if (pgi != parameter_gradient.end()) {
			// Reset to zero
			std::fill(pgi->second.begin(), pgi->second.end(), 0.0);
		} else {
			// Create
			parameter_gradient[ftname] =
				std::vector<double>((*fti)->WeightDimension(), 0.0);
		}
	}
}

void ContrastiveDivergenceTraining::GradientScale(
	std::tr1::unordered_map<std::string, std::vector<double> >&
		parameter_gradient, double scale) const {
	for (std::tr1::unordered_map<std::string, std::vector<double> >::iterator
		pgi = parameter_gradient.begin(); pgi != parameter_gradient.end();
		++pgi) {
		std::transform(pgi->second.begin(), pgi->second.end(),
			pgi->second.begin(), scale * _1);
	}
}

}

