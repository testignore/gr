
#include <cassert>

#include "SubFactorGraph.h"
#include "NaivePiecewiseTraining.h"

namespace Grante {

// TODO: rewrite so that no explicit subgraphs are formed
NaivePiecewiseTraining::NaivePiecewiseTraining(FactorGraphModel* fg_model)
	: ParameterEstimationMethod(fg_model), mle(fg_model) {
}

NaivePiecewiseTraining::~NaivePiecewiseTraining() {
	assert(pw_training_data.size() == pw_inference_methods.size());
	for (unsigned int n = 0; n < pw_training_data.size(); ++n) {
		delete (pw_training_data[n].second);
		delete (pw_inference_methods[n]);
		delete (subfgs[n]);	// also deletes pw_training_data[n].first
	}
}

void NaivePiecewiseTraining::SetOptimizationMethod(
	MaximumLikelihood::MLEOptimizationMethod opt_method) {
	mle.SetOptimizationMethod(opt_method);
}

MaximumLikelihood::MLEProblem* NaivePiecewiseTraining::GetLearnProblem() {
	return (mle.GetLearnProblem());
}

void NaivePiecewiseTraining::SetupTrainingData(
	const std::vector<labeled_instance_type>& training_data,
	const std::vector<InferenceMethod*> inference_methods) {
	// Argument and context check
	assert(pw_training_data.size() == 0);
	assert(pw_inference_methods.size() == 0);
	assert(inference_methods.size() == training_data.size());

	// Count piecewise graphs to be created
	size_t total_pw_count = 0;
	size_t training_data_size = training_data.size();
	for (size_t n = 0; n < training_data_size; ++n)
		total_pw_count += training_data[n].first->Factors().size();

	pw_training_data.reserve(total_pw_count);
	pw_inference_methods.reserve(total_pw_count);
	subfgs.reserve(total_pw_count);

	// Decompose each factor graph into all its factor pieces
	for (size_t n = 0; n < training_data_size; ++n) {
		FactorGraph* fg = training_data[n].first;
		const FactorGraphObservation* obs = training_data[n].second;

		const std::vector<Factor*>& factors = fg->Factors();
		std::vector<unsigned int> f_set(1);
		std::vector<double> f_scale(1, 1.0);
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			f_set[0] = fi;
			SubFactorGraph* sfg = new SubFactorGraph(fg, f_set, f_scale);

			// Create observation on sub-factorgraph
			FactorGraphObservation* sobs = sfg->ConstructSubObservation(obs);

			subfgs.push_back(sfg);
			pw_training_data.push_back(labeled_instance_type(sfg->FG(), sobs));
			assert(inference_methods[n] != 0);
			pw_inference_methods.push_back(
				inference_methods[n]->Produce(sfg->FG()));
		}
	}

	// Pass pieces to normal MLE
	mle.SetupTrainingData(pw_training_data, pw_inference_methods);
}

void NaivePiecewiseTraining::UpdateTrainingLabeling(
	const std::vector<labeled_instance_type>& training_update) {
	unsigned int pw_idx = 0;
	size_t training_update_size = training_update.size();
	for (size_t n = 0; n < training_update_size; ++n) {
		FactorGraph* fg = training_update[n].first;
		const FactorGraphObservation* obs = training_update[n].second;

		const std::vector<Factor*>& factors = fg->Factors();
		for (size_t fi = 0; fi < factors.size(); ++fi) {
			FactorGraphObservation* sobs =
				subfgs[pw_idx]->ConstructSubObservation(obs);

			delete (pw_training_data[pw_idx].second);
			pw_training_data[pw_idx].second = sobs;
			pw_idx += 1;
		}
	}
	assert(pw_idx == pw_training_data.size());
	mle.UpdateTrainingLabeling(pw_training_data);
}

void NaivePiecewiseTraining::AddPrior(const std::string& factor_type,
	Prior* prior) {
	mle.AddPrior(factor_type, prior);
}

double NaivePiecewiseTraining::Train(double conv_tol, unsigned int max_iter) {
	return (mle.Train(conv_tol, max_iter));
}

}

