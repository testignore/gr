
#include <iostream>
#include <vector>
#include <ctime>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "LinearFactorType.h"
#include "Factor.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "Likelihood.h"
#include "GibbsSampler.h"
#include "MaximumLikelihood.h"
#include "MaximumPseudolikelihood.h"
#include "FactorGraphObservation.h"

#define BOOST_TEST_MODULE(LinearFactorTypeTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(Simple)
{
	Grante::FactorGraphModel model;

	// Create one linear pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<int> A;
	A.push_back(-1);
	A.push_back(0);
	A.push_back(0);
	A.push_back(-1);

	double a = 0.05;
	std::vector<double> w;
	w.push_back(a);
	Grante::LinearFactorType* factortype =
		new Grante::LinearFactorType("pairwise", card, w, 0, A);
	model.AddFactorType(factortype);

	// Add unary factor types
	std::vector<unsigned int> card1;
	card1.push_back(2);
	std::vector<double> w1;
	w1.push_back(0.3);
	w1.push_back(0.7);
	Grante::FactorType* factortype1a = new Grante::FactorType("unary1", card1, w1);
	model.AddFactorType(factortype1a);

	w1[0] = 0.4;
	w1[1] = 0.6;
	Grante::FactorType* factortype1b = new Grante::FactorType("unary2", card1, w1);
	model.AddFactorType(factortype1b);

	// Create factor graph
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	const Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	const Grante::FactorType* pt1a = model.FindFactorType("unary1");
	const Grante::FactorType* pt1b = model.FindFactorType("unary2");
	BOOST_REQUIRE(pt1a != 0);
	BOOST_REQUIRE(pt1b != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	std::vector<unsigned int> var_index1(1);
	var_index1[0] = 0;
	Grante::Factor* fac1a = new Grante::Factor(pt1a, var_index1, data);
	fg.AddFactor(fac1a);
	var_index1[0] = 1;
	Grante::Factor* fac1b = new Grante::Factor(pt1b, var_index1, data);
	fg.AddFactor(fac1b);

	// Compute the forward map
	fg.ForwardMap();

	std::vector<unsigned int> state(2);
	state[0] = 0;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(0.7, fg.EvaluateEnergy(state), 1.0e-6);
	state[0] = 0;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(0.9 + a, fg.EvaluateEnergy(state), 1.0e-6);
	state[0] = 1;
	state[1] = 0;
	BOOST_CHECK_CLOSE_ABS(1.1 + a, fg.EvaluateEnergy(state), 1.0e-6);
	state[0] = 1;
	state[1] = 1;
	BOOST_CHECK_CLOSE_ABS(1.3, fg.EvaluateEnergy(state), 1.0e-6);
}

BOOST_AUTO_TEST_CASE(MLETyingSimple)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(0.0);
	w.push_back(0.45);
	w.push_back(0.55);
	w.push_back(0.0);
	Grante::FactorType* factortype = new Grante::FactorType("pairwise", card, w);
	model.AddFactorType(factortype);

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(2);
	vc.push_back(2);
	vc.push_back(2);
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Get marginals
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 20000;
	tinf.Sample(states, sample_count);


	///////
	// Setup symmetric/sparsity-constrained model
	///////
	Grante::FactorGraphModel model_learn;

	// Create one linear pairwise factor type
	std::vector<int> A;
	A.push_back(-1);
	A.push_back(0);
	A.push_back(0);
	A.push_back(-1);

	std::vector<double> wl(1, 0.0);
	Grante::LinearFactorType* pw_factortype =
		new Grante::LinearFactorType("pairwise", card, wl, 0, A);
	model_learn.AddFactorType(pw_factortype);

	// Create a factor graph from the model: 3 binary variables
	Grante::FactorGraph fg_l(&model_learn, vc);

	// Add factors
	Grante::FactorType* pt_l = model_learn.FindFactorType("pairwise");
	BOOST_REQUIRE(pt_l != 0);
	var_index[0] = 0;
	var_index[1] = 1;
	fg_l.AddFactor(new Grante::Factor(pt_l, var_index, data));
	var_index[0] = 1;
	var_index[1] = 2;
	fg_l.AddFactor(new Grante::Factor(pt_l, var_index, data));

	Grante::TreeInference tinf_l(&fg_l);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg_l, new Grante::FactorGraphObservation(states[si])));

		// Push the same inference object again (graph is of fixed-structure)
		inference_methods.push_back(&tinf_l);
	}
	Grante::MaximumLikelihood mle(&model_learn);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1e-5);

	for (unsigned int wi = 0; wi < pt_l->Weights().size(); ++wi) {
		std::cout << "  dim " << wi
			<< ", learned " << pt_l->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg_l.ForwardMap();
	tinf_l.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf_l.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n)
		delete (training_data[n].second);
}

