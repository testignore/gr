
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "Likelihood.h"
#include "GibbsSampler.h"
#include "MaximumLikelihood.h"
#include "MaximumPseudolikelihood.h"
#include "FactorGraphObservation.h"
#include "NormalPrior.h"
#include "HyperbolicPrior.h"
#include "LaplacePrior.h"
#include "StudentTPrior.h"

#define BOOST_TEST_MODULE(LikelihoodTest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(Digit1)
{
	Grante::FactorGraphModel model;

	// Create one unary factor type
	std::vector<unsigned int> card(1, 10);
	std::vector<double> w(10*16*16, 0.1);
	Grante::FactorType* ft = new Grante::FactorType("digit_unary", card, w);
	model.AddFactorType(ft);

	// Get the factor type back from the model
	Grante::FactorType* pt = model.FindFactorType("digit_unary");
	std::vector<unsigned int> factor_varcard(1, 10);
	std::vector<unsigned int> factor_varindex(1, 0);
	std::vector<unsigned int> fg_label(1);

	// Setup training data
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int digit = 0; digit <= 9; ++digit) {
		fg_label[0] = digit;

		// Build a set of factor graph for each digit
		for (unsigned int n = 0; n < 10; ++n) {
			std::vector<double> data(256, 0.0);
			for (unsigned int di = 0; di < 256; ++di)
				data[di] = 1e-6 * static_cast<double>(di * 10 + n);
			data[digit] = 1.0;

			// Build factor graph
			Grante::FactorGraph* fg =
				new Grante::FactorGraph(&model, factor_varcard);
			Grante::Factor* fac =
				new Grante::Factor(pt, factor_varindex, data);
			fg->AddFactor(fac);

			Grante::ParameterEstimationMethod::labeled_instance_type
				lit(fg, new Grante::FactorGraphObservation(fg_label));
			training_data.push_back(lit);

			Grante::TreeInference* tinf = new Grante::TreeInference(fg);
			inference_methods.push_back(tinf);
		}
	}

	// Train using maximum likelihood
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.AddPrior("digit_unary", new Grante::NormalPrior(1.0e-1, w.size()));
	mle.Train(1e-4);

	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].first);
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(MPLEDataSimple)
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);
	data[0] = 0.1;
	data[1] = 1.3;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	data[0] = 0.3;
	data[1] = 0.4;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Perform inference
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 5000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumPseudolikelihood mple(&model);
	mple.SetupTrainingData(training_data, inference_methods);
	mple.Train(1e-5);

	std::cout << "### Parameters" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::cout << "### Marginals" << std::endl;
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}

	// Delete objects
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(Simple)
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);
	data[0] = 0.1;
	data[1] = 0.2;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	data[0] = 0.3;
	data[1] = 0.4;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Perform inference
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();

	std::vector<std::vector<double> > marginals;
	marginals.push_back(tinf.Marginal(0));
	marginals.push_back(tinf.Marginal(1));

	Grante::Likelihood likelihood(&model);
	std::vector<unsigned int> observed(3);
	observed[0] = 0;
	observed[1] = 1;
	observed[2] = 0;
	Grante::FactorGraphObservation obs(observed);

	std::tr1::unordered_map<std::string, std::vector<double> >
		parameter_gradient;
	parameter_gradient["pairwise"] = std::vector<double>(8, 0.0);
	double nll = likelihood.ComputeNegLogLikelihood(&fg,
		&obs, marginals, tinf.LogPartitionFunction(), parameter_gradient);

	const std::vector<double>& pwg = parameter_gradient["pairwise"];

	// Add a little of the gradient to the parameters in the model
	for (unsigned int wi = 0; wi < pwg.size(); ++wi)
		pt->Weights()[wi] += -1e-2*pwg[wi];
	fg.ForwardMap();
	tinf.PerformInference();
	marginals[0] = tinf.Marginal(0);
	marginals[1] = tinf.Marginal(1);
	parameter_gradient["pairwise"] = std::vector<double>(8, 0.0);
	double nll_new = likelihood.ComputeNegLogLikelihood(&fg,
		&obs, marginals, tinf.LogPartitionFunction(), parameter_gradient);

	BOOST_CHECK(nll_new < nll);
}

BOOST_AUTO_TEST_CASE(MLESimple)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
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

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1e-5);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(MLEDataSimple)
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(2);
	std::vector<unsigned int> var_index(2);
	data[0] = 0.1;
	data[1] = 1.3;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac1);

	data[0] = 0.3;
	data[1] = 0.4;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Perform inference
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 5000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1e-5);

	std::cout << "### Parameters" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::cout << "### Marginals" << std::endl;
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(MLEDataSimpleSparse)
{
	std::vector<std::vector<double> > marg_dense;
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
		Grante::FactorGraph fg(&model, vc);

		// Add factors
		Grante::FactorType* pt = model.FindFactorType("pairwise");
		BOOST_REQUIRE(pt != 0);
		std::vector<double> data(2);
		std::vector<unsigned int> var_index(2);
		data[0] = 0.71;
		data[1] = 0.0;
		var_index[0] = 0;
		var_index[1] = 1;
		Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data);
		fg.AddFactor(fac1);

		data[0] = 0.0;
		data[1] = 0.35;
		var_index[0] = 1;
		var_index[1] = 2;
		Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data);
		fg.AddFactor(fac2);

		// Compute the forward map
		fg.ForwardMap();

		// Perform inference
		Grante::TreeInference tinf(&fg);
		tinf.PerformInference();
		marg_dense = tinf.Marginals();
	}
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
	Grante::FactorGraph fg(&model, vc);

	// Add factors
	Grante::FactorType* pt = model.FindFactorType("pairwise");
	BOOST_REQUIRE(pt != 0);
	std::vector<double> data(1);
	std::vector<unsigned int> data_idx(1);
	std::vector<unsigned int> var_index(2);
	data_idx[0] = 0;
	data[0] = 0.71;
	var_index[0] = 0;
	var_index[1] = 1;
	Grante::Factor* fac1 = new Grante::Factor(pt, var_index, data, data_idx);
	fg.AddFactor(fac1);

	data_idx[0] = 1;
	data[0] = 0.35;
	var_index[0] = 1;
	var_index[1] = 2;
	Grante::Factor* fac2 = new Grante::Factor(pt, var_index, data, data_idx);
	fg.AddFactor(fac2);

	// Compute the forward map
	fg.ForwardMap();

	// Perform inference
	Grante::TreeInference tinf(&fg);
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_true = tinf.Marginals();

	// Sample a population from the true model
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 15000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1e-5);

	std::cout << "### Parameters" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::cout << "### Marginals" << std::endl;
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_dense[fi][wi], 1.0e-5);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(MLECRFSimple)
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
	unsigned int instance_count = 1000;
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	training_data.reserve(instance_count);
	inference_methods.reserve(instance_count);
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

		// Compute the forward map
		fg->ForwardMap();

		// Perform inference
		Grante::TreeInference* tinf = new Grante::TreeInference(fg);
		inference_methods.push_back(tinf);
		tinf->PerformInference();
		std::vector<std::vector<double> > marg_true = tinf->Marginals();

		// Use exact marginals as target distribution
		// (multiple instances should ensure identifiable parameters)
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				fg, new Grante::FactorGraphObservation(marg_true)));
	}

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.AddPrior("pairwise", new Grante::NormalPrior(10.0, w.size()));
	mle.Train(1e-5);

	std::cout << "### Parameters" << std::endl;
	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << std::setprecision(7) << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals produced by learned weights with truth
	for (unsigned int n = 0; n < instance_count; ++n) {
		training_data[n].first->ForwardMap();
		inference_methods[n]->PerformInference();
		std::vector<std::vector<double> > marg_learned =
			inference_methods[n]->Marginals();
		const std::vector<std::vector<double> >& marg_truth =
			training_data[n].second->Expectation();

		assert(marg_learned.size() == marg_truth.size());
		for (unsigned int fi = 0; fi < marg_learned.size(); ++fi) {
			assert(marg_learned[fi].size() == marg_truth[fi].size());
			for (unsigned int ei = 0; ei < marg_learned[fi].size(); ++ei) {
#if 0
				std::cout << "instance " << n << ", factor " << fi
					<< ", element " << ei << ": learned " << marg_learned[fi][ei]
					<< ", truth " << marg_truth[fi][ei] << std::endl;
#endif
				BOOST_CHECK_CLOSE_ABS(marg_truth[fi][ei], marg_learned[fi][ei], 0.025);
			}
		}
		delete (inference_methods[n]);
		delete (training_data[n].first);
		delete (training_data[n].second);
	}
}

BOOST_AUTO_TEST_CASE(RegularizedMLESimple)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
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
	unsigned int sample_count = 25000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);

	// Add a Normal prior
	mle.AddPrior("pairwise", new Grante::NormalPrior(1e-1, w.size()));
	mle.Train(1e-5);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}

	// Delete objects
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(RegularizedMLEL1)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.0);
	w.push_back(-0.2);
	w.push_back(1.0);
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
	unsigned int sample_count = 15000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);

	// Add a Normal prior
	mle.AddPrior("pairwise", new Grante::LaplacePrior(1.0, w.size()));
	std::cout << "FISTA minimization" << std::endl;
	mle.SetOptimizationMethod(Grante::MaximumLikelihood::FISTAMethod);
	mle.Train(1.0e-8);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}

	// Delete objects
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(StudentTRegularizedMLESimple)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
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
	unsigned int sample_count = 25000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);

	// Add a Normal prior
	mle.AddPrior("pairwise", new Grante::StudentTPrior(1e3, 1.0, w.size()));
	mle.Train(1e-5);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

BOOST_AUTO_TEST_CASE(MPLESimple)
{
	Grante::FactorGraphModel model;

	// Create one simple parametrized, data-independent pairwise factor type
	std::vector<unsigned int> card;
	card.push_back(2);
	card.push_back(2);
	std::vector<double> w;
	w.push_back(1.0);
	w.push_back(0.2);
	w.push_back(-0.2);
	w.push_back(1.0);
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
	unsigned int sample_count = 10000;
	tinf.Sample(states, sample_count);

	// Change model parameters
	std::vector<double> w_truth(pt->Weights());
	std::fill(pt->Weights().begin(), pt->Weights().end(), 0.0);

	// Reconstruct model weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(tinf.Produce(&fg));
	}
	Grante::MaximumPseudolikelihood mple(&model);
	mple.SetupTrainingData(training_data, inference_methods);
	mple.AddPrior("pairwise",
		new Grante::HyperbolicPrior(w.size(), 0.5, 1.0));
	mple.Train(1e-5);

	for (unsigned int wi = 0; wi < w_truth.size(); ++wi) {
		std::cout << "  dim " << wi << ": truth " << w_truth[wi]
			<< ", learned " << pt->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg.ForwardMap();
	tinf.PerformInference();
	std::vector<std::vector<double> > marg_mple = tinf.Marginals();
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mple[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mple[fi][wi], 0.025);
		}
	}

	// Delete objects
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
}

