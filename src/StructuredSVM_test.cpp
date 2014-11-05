
#include <vector>
#include <iostream>
#include <ctime>

#include <boost/random.hpp>

#include "FactorType.h"
#include "FactorGraph.h"
#include "FactorGraphModel.h"
#include "StructuredHammingLoss.h"
#include "StructuredSVM.h"
#include "StructuredPerceptron.h"
#include "TreeInference.h"
#include "NormalPrior.h"

#define BOOST_TEST_MODULE(StructuredSVMTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(StructuredSVMSimple)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type:
	// Each energy is the inner product of a two-vector data with a
	// state-specific weight vector.
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.3);	// for (0,0)
	w.push_back(0.5);	// for (0,0)
	w.push_back(1.0);	// for (1,0)
	w.push_back(0.2);	// for (1,0)
	w.push_back(0.05);	// for (0,1)
	w.push_back(0.6);	// for (0,1)
	w.push_back(-0.2);	// for (1,1)
	w.push_back(0.75);	// for (1,1)
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);

	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);

	// Randomly set the data observations
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Create a set of factor graphs realizing this model
	unsigned int instance_count = 500;
	std::vector<Grante::FactorGraph*> instances;
	std::vector<Grante::StructuredLossFunction*> loss;
	std::vector<Grante::InferenceMethod*> inference_methods;

	// for Perceptron
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;

	for (unsigned int n = 0; n < instance_count; ++n) {
		Grante::FactorGraph* fg = new Grante::FactorGraph(&model, vc);
		// Add factors
		data[0] = 2.0*randu() - 1.0;
		data[1] = 2.0*randu() - 1.0;
		var_index[0] = 0;
		var_index[1] = 1;
		Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
		fg->AddFactor(fac1);

		data[0] = 2.0*randu() - 1.0;
		data[1] = 2.0*randu() - 1.0;
		var_index[0] = 1;
		var_index[1] = 2;
		Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
		fg->AddFactor(fac2);

		// Add instance
		instances.push_back(fg);

		// Compute the forward map
		fg->ForwardMap();

		// Perform inference
		Grante::TreeInference* tinf = new Grante::TreeInference(fg);
		inference_methods.push_back(tinf);
		std::vector<unsigned int> y_min;
		tinf->MinimizeEnergy(y_min);

		// Add ground truth
		loss.push_back(new Grante::StructuredHammingLoss(
			new Grante::FactorGraphObservation(y_min)));

		// for Perceptron (structured SVM sees the ground truth only through
		// the structured loss).
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				fg, new Grante::FactorGraphObservation(y_min)));
	}

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	double acc_loss = 0.0;
	double mean_emp_loss = 0.0;

	// Train: stochastic
	Grante::StructuredSVM ssvm_s(&model, 100.0, "bmrm");
	ssvm_s.AddPrior("pairwise", new Grante::NormalPrior(10.0, w.size()));
	ssvm_s.SetupTrainingData(instances, loss, inference_methods);
	ssvm_s.Train(1e-2, 100);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Evaluate structured loss for stochastically-trained weights
	acc_loss = 0.0;
	for (unsigned int n = 0; n < instances.size(); ++n) {
		instances[n]->ForwardMap();
		std::vector<unsigned int> y_pred;
		inference_methods[n]->MinimizeEnergy(y_pred);
		double cur_loss = loss[n]->Eval(y_pred);
		acc_loss += cur_loss;
	}
	mean_emp_loss =
		acc_loss / static_cast<double>(instances.size());
	std::cout << "BMRM: mean empirical structured loss: "
		<< mean_emp_loss << std::endl;

	// Check that we have a low empirical loss
	BOOST_CHECK_LT(mean_emp_loss, 0.125);

	// Train: structured Perceptron
	Grante::StructuredPerceptron sperc(&model);
	sperc.SetupTrainingData(training_data, inference_methods);
	sperc.Train(0.0, 100);

	// Evaluate structured loss for Perceptron-trained weights
	acc_loss = 0.0;
	for (unsigned int n = 0; n < instances.size(); ++n) {
		instances[n]->ForwardMap();
		std::vector<unsigned int> y_pred;
		inference_methods[n]->MinimizeEnergy(y_pred);
		double cur_loss = loss[n]->Eval(y_pred);
		acc_loss += cur_loss;
	}
	mean_emp_loss =
		acc_loss / static_cast<double>(instances.size());
	std::cout << "Perceptron: mean empirical structured loss: "
		<< mean_emp_loss << std::endl;
	BOOST_CHECK_LT(mean_emp_loss, 0.125);

	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].first);
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}


BOOST_AUTO_TEST_CASE(HammingLoss)
{
	Grante::FactorGraphModel model;

	// Create one simple pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);
	w.push_back(0.3);
	w.push_back(0.2);
	w.push_back(0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.1);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.3;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	// Create a factor graph from the model: 2 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt2 = model.FindFactorType("pairwise");
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt2 != 0);
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt2, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// True state
	std::vector<unsigned int> obs_state(2);
	obs_state[0] = 1;
	obs_state[1] = 0;
	Grante::FactorGraphObservation* fg_obs =
		new Grante::FactorGraphObservation(obs_state);

	// Check structured loss evaluation
	Grante::StructuredHammingLoss h_loss(fg_obs);
	BOOST_CHECK_SMALL(h_loss.Eval(obs_state), 1e-5);

	std::vector<unsigned int> obs_state_1(2);
	obs_state_1[0] = 1;
	obs_state_1[1] = 1;
	BOOST_CHECK_CLOSE_ABS(1.0, h_loss.Eval(obs_state_1), 1e-5);

	std::vector<unsigned int> obs_state_2(2);
	obs_state_2[0] = 0;
	obs_state_2[1] = 1;
	BOOST_CHECK_CLOSE_ABS(2.0, h_loss.Eval(obs_state_2), 1e-5);

	// Perform loss augmentation
	std::vector<unsigned int> state(2);
	for (unsigned int s1 = 0; s1 <= 1; ++s1) {
		state[0] = s1;
		for (unsigned int s2 = 0; s2 <= 1; ++s2) {
			state[1] = s2;
			fg.ForwardMap();
			assert(state[0] <= 1);
			assert(state[1] <= 1);
			double base_energy = fg.EvaluateEnergy(state);
			h_loss.PerformLossAugmentation(&fg);
			double augmented_energy = fg.EvaluateEnergy(state);
			BOOST_CHECK_CLOSE_ABS(augmented_energy - base_energy,
				h_loss.Eval(state), 1e-5);
		}
	}
}

