
#include <cmath>
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "ParameterEstimationMethod.h"

namespace Grante {

ParameterEstimationMethod::ParameterEstimationMethod(
	FactorGraphModel* fg_model)
	: fg_model(fg_model) {
}

ParameterEstimationMethod::~ParameterEstimationMethod() {
	for (std::multimap<std::string, Prior*>::iterator pi = priors.begin();
		pi != priors.end(); ++pi) {
		delete (pi->second);
	}
}

void ParameterEstimationMethod::AddPrior(const std::string& factor_type,
	Prior* prior) {
	priors.insert(std::multimap<std::string, Prior*>::value_type(
		factor_type, prior));
}

void ParameterEstimationMethod::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	assert(inference_methods.empty() ||
		training_data.size() == inference_methods.size());
	this->training_data = training_data;
	this->inference_methods = inference_methods;
}

void ParameterEstimationMethod::PrintProblemStatistics() const {
	size_t sample_count = training_data.size();
	size_t var_count = 0;	// Total variables in factor graphs
	double log2_states = 0.0;	// Total log_2 of state space
	size_t factor_count = 0;		// Total number of factors
	size_t data_size = 0;	// Total data elements
	for (unsigned int si = 0; si < sample_count; ++si) {
		const FactorGraph* fg = training_data[si].first;

		// Count variables and state space
		const std::vector<unsigned int>& card = fg->Cardinalities();
		var_count += card.size();
		for (unsigned int vi = 0; vi < card.size(); ++vi)
			log2_states += std::log(static_cast<double>(card[vi]))
				/ std::log(2.0);

		const std::vector<Factor*>& facs = fg->Factors();
		factor_count += facs.size();
		for (unsigned int fi = 0; fi < facs.size(); ++fi)
			data_size += facs[fi]->Data().size();
	}

	// Count parameters
	assert(training_data.size() > 0);
	size_t param_elem_count = 0;	// Total number of scalar pars
	const std::vector<FactorType*>& ftypes =
		training_data[0].first->Model()->FactorTypes();
	for (size_t fti = 0; fti < ftypes.size(); ++fti)
		param_elem_count += ftypes[fti]->Weights().size();

	std::cout << std::endl;
	std::cout << "Learning problem statistics.  There are " << var_count
		<< " variables in " << sample_count << " factor graphs,"
		<< std::endl;
	std::cout << "taking one of 2^" << log2_states
		<< " states and interacting through " << factor_count
		<< " factors," << std::endl;
	std::cout << "accessing " << data_size << " data elements." << std::endl;
	if (ftypes.size() == 1) {
		std::cout << "Learning " << param_elem_count << " parameters in "
			<< "a single factor type." << std::endl;
	} else {
		std::cout << "Learning " << param_elem_count << " parameters in "
			<< ftypes.size() << " different factor types." << std::endl;
	}
#ifdef _OPENMP
	std::cout << "Using a maximum of " << omp_get_max_threads()
		<< " threads." << std::endl;
#endif
	std::cout << std::endl;
}

void ParameterEstimationMethod::UpdateTrainingLabeling(
	const std::vector<labeled_instance_type>& training_update) {
	assert(training_data.size() == training_update.size());
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		assert(training_data[n].first == training_update[n].first);
	}
	this->training_data = training_update;
}

}

