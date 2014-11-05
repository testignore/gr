
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <functional>
#include <ctime>
#include <cassert>

#include <boost/timer.hpp>
#include <boost/random.hpp>
#include <boost/lambda/lambda.hpp>

#include "StructuredPerceptron.h"

using namespace boost::lambda;

namespace Grante {

StructuredPerceptron::StructuredPerceptron(FactorGraphModel* fg_model,
	bool do_averaging, bool verbose)
	: ParameterEstimationMethod(fg_model), do_averaging(do_averaging),
		verbose(verbose), lh(fg_model)
{
	// Initialize parameter order
	const std::vector<FactorType*>& factor_types = fg_model->FactorTypes();
	for (std::vector<FactorType*>::const_iterator fti = factor_types.begin();
		fti != factor_types.end(); ++fti) {
		parameter_order.push_back((*fti)->Name());
	}
}

StructuredPerceptron::~StructuredPerceptron() {
}

void StructuredPerceptron::AddPrior(const std::string& factor_type,
	Prior* prior) {
	// Priors are not supported for Perceptron training
	assert(0);
}

double StructuredPerceptron::Train(double conv_tol, unsigned int max_epochs) {
	// Initialize random number generator
	size_t N = training_data.size();
	assert(N > 0);
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_int<unsigned int> rdestd(0, static_cast<unsigned int>(N-1));
	boost::variate_generator<boost::mt19937,
		boost::uniform_int<unsigned int> > rand_n(rgen, rdestd);
	std::cout << "Perceptron training with " << N << " training instances"
		<< std::endl;

	// For all epochs
	assert(max_epochs > 0);
	boost::timer total_timer;
	unsigned int total_iter = 0;
	unsigned int violation_count = 0;
	for (unsigned int epoch = 0; max_epochs == 0 || epoch < max_epochs; ++epoch) {
		violation_count = 0;
		for (size_t n = 0; n < N; ++n) {
			// Sample an instance id uniformly at random
			unsigned int sample_id = rand_n();

			ClearParameterGradient();
			bool is_violated = ProcessSample(sample_id);
			if (is_violated) {
				UpdateFactorWeights();
				violation_count += 1;
			}

			// Average parameters
			total_iter += 1;
			if (do_averaging) {
				UpdateAveragedParameters(static_cast<double>(total_iter-1)/
					static_cast<double>(total_iter),
					1.0/static_cast<double>(total_iter));
			}
		}

		if (verbose && (epoch % 20 == 0)) {
			std::cout << std::endl;
			std::cout << "  iter     time      violations" << std::endl;
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
			std::cout << std::resetiosflags(std::ios::scientific)
				<< std::setprecision(5)
				<< std::setiosflags(std::ios::left)
				<< std::setiosflags(std::ios::showpos)
				<< std::setw(7) << violation_count << " of " << N;
			std::cout << std::endl;

			std::cout.precision(original_prec);
			std::cout.flags(original_format);
		}
		if (violation_count == 0)
			break;
	}

	// Use averaged parameters
	if (do_averaging) {
		assert(total_iter >= 1);
		SetFactorWeights();
	}

	return (violation_count);
}

bool StructuredPerceptron::ProcessSample(unsigned int sample_id) {
	FactorGraph* ts_fg = training_data[sample_id].first;
	const FactorGraphObservation* ts_obs = training_data[sample_id].second;
	InferenceMethod* ts_inf = inference_methods[sample_id];
	assert(ts_fg != 0);
	assert(ts_obs != 0);
	assert(ts_inf != 0);

	// Compute forward map: parameters (changed) to energies
	ts_fg->ForwardMap();

	// Compute: -E(y_n;x_n,w)
	/*double obj_truth = */
	lh.ComputeObservationEnergy(ts_fg, ts_obs, parameter_gradient, -1.0);

	// Compute: min_y E(y;x_n,w)
	std::vector<unsigned int> y_star;
	/*double obj_star = */
	ts_inf->MinimizeEnergy(y_star);

	//  ii) Compute gradient (already of negative sign)
	lh.ComputeObservationEnergy(ts_fg, y_star, parameter_gradient, 1.0);
	ts_inf->ClearInferenceResult();
#if 0
	std::cout << "perc, sample " << sample_id << ", true E " << obj_truth
		<< ", y* E " << obj_star << ", viol " << -(obj_truth - obj_star)
		<< std::endl;
#endif

	// Equal energy does not imply no violation (eg. w=0).  Therefore, check
	// violation
	if (ts_obs->Type() == FactorGraphObservation::ExpectationType)
		return (true);

	const std::vector<unsigned int>& true_state = ts_obs->State();
	assert(true_state.size() == y_star.size());
	for (unsigned int vi = 0; vi < true_state.size(); ++vi)
		if (true_state[vi] != y_star[vi])
			return (true);

	return (false);
}

void StructuredPerceptron::UpdateFactorWeights() {
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = fg_model->FindFactorType(*ft_name);
		std::transform(parameter_gradient[*ft_name].begin(),
			parameter_gradient[*ft_name].end(), ft->Weights().begin(),
			ft->Weights().begin(), std::plus<double>());
	}
}

void StructuredPerceptron::UpdateAveragedParameters(double old_factor,
	double new_factor) {
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = fg_model->FindFactorType(*ft_name);

		// First iteration: old average is zero
		if (parameter_averaged[*ft_name].empty()) {
			parameter_averaged[*ft_name].resize(ft->Weights().size());
			std::fill(parameter_averaged[*ft_name].begin(),
				parameter_averaged[*ft_name].end(), 0.0);
		}

		// Total average equation:
		//   v_0 = 0,
		//   v_t = ((L-1)/L) v_{t-1} + (1/L) w_L.
		std::transform(ft->Weights().begin(), ft->Weights().end(),
			parameter_averaged[*ft_name].begin(),
			parameter_averaged[*ft_name].begin(),
			old_factor * _2 + new_factor * _1);
	}
}

void StructuredPerceptron::SetFactorWeights() {
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		// Get factor type
		FactorType* ft = fg_model->FindFactorType(*ft_name);
		std::copy(parameter_averaged[*ft_name].begin(),
			parameter_averaged[*ft_name].end(), ft->Weights().begin());
	}
}

void StructuredPerceptron::ClearParameterGradient() {
	parameter_gradient.clear();
	for (std::vector<std::string>::const_iterator
		ft_name = parameter_order.begin();
		ft_name != parameter_order.end(); ++ft_name) {
		FactorType* ft = fg_model->FindFactorType(*ft_name);
		parameter_gradient[*ft_name] =
			std::vector<double>(ft->WeightDimension(), 0.0);
	}
}

}

