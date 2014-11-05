
#include <vector>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "Likelihood.h"
#include "GibbsInference.h"
#include "BruteForceExactInference.h"
#include "MaximumLikelihood.h"
#include "MaximumTreePseudoLikelihood.h"
#include "FactorGraphObservation.h"

#define BOOST_TEST_MODULE(MTPLETest)
#include <boost/test/unit_test.hpp>
#include "Testing.h"

BOOST_AUTO_TEST_CASE(MTPLESimple)
{
	// Create this model
	//
	//      (0) ---[B]--- (2)
	//       |          /
	//       |        /
	//      [A]    [C]
	//       |    /
	//       |  /
	//      (1)
	Grante::FactorGraphModel model;

	std::vector<unsigned int> card_pw(2, 2);

	std::vector<double> w_A;
	w_A.push_back(0.0);
	w_A.push_back(0.7);
	w_A.push_back(0.4);
	w_A.push_back(0.1);
	Grante::FactorType* factortype_A =
		new Grante::FactorType("A", card_pw, w_A);
	model.AddFactorType(factortype_A);

	std::vector<double> w_B;
	w_B.push_back(0.8);
	w_B.push_back(0.5);
	w_B.push_back(0.6);
	w_B.push_back(0.2);
	Grante::FactorType* factortype_B =
		new Grante::FactorType("B", card_pw, w_B);
	model.AddFactorType(factortype_B);

	std::vector<double> w_C;
	w_C.push_back(0.2);
	w_C.push_back(0.7);
	w_C.push_back(0.3);
	w_C.push_back(0.4);
	Grante::FactorType* factortype_C =
		new Grante::FactorType("C", card_pw, w_C);
	model.AddFactorType(factortype_C);

	// Create a cyclic factor graph from the model: 3 binary variables
	std::vector<unsigned int> card_cyc(3, 2);
	Grante::FactorGraph fg_cyc(&model, card_cyc);

	// Add factors
	Grante::FactorType* pt_A = model.FindFactorType("A");
	Grante::FactorType* pt_B = model.FindFactorType("B");
	Grante::FactorType* pt_C = model.FindFactorType("C");
	std::vector<double> data;
	std::vector<unsigned int> var_index(2);
	var_index[0] = 0;
	var_index[1] = 1;
	fg_cyc.AddFactor(new Grante::Factor(pt_A, var_index, data));
	var_index[0] = 0;
	var_index[1] = 2;
	fg_cyc.AddFactor(new Grante::Factor(pt_B, var_index, data));
	var_index[0] = 1;
	var_index[1] = 2;
	fg_cyc.AddFactor(new Grante::Factor(pt_C, var_index, data));

	// Create tree-unrolled factor graph from the model
	//
	// (2') ---[C]--- (1) ---[A]--- (0) ---[B]--- (2) ---[C]--- (1')
	//
	// Note that we need to pay attention to the correct direction of the
	// copied C factors.
	std::vector<unsigned int> card_tu(5, 2);
	Grante::FactorGraph fg_tu(&model, card_tu);

	var_index[0] = 0;
	var_index[1] = 1;
	fg_tu.AddFactor(new Grante::Factor(pt_A, var_index, data));
	var_index[0] = 0;
	var_index[1] = 2;
	fg_tu.AddFactor(new Grante::Factor(pt_B, var_index, data));
	var_index[0] = 1;
	var_index[1] = 3;
	fg_tu.AddFactor(new Grante::Factor(pt_C, var_index, data));
	var_index[0] = 4;
	var_index[1] = 2;
	fg_tu.AddFactor(new Grante::Factor(pt_C, var_index, data));

	// Compute the forward map
	fg_cyc.ForwardMap();
	fg_tu.ForwardMap();

	// Get marginals
	Grante::BruteForceExactInference bfinf(&fg_cyc);
	bfinf.PerformInference();
	std::vector<std::vector<double> > marg_true = bfinf.Marginals();

	// Sample a population from the true model
	Grante::GibbsInference ginf(&fg_cyc);
	ginf.SetSamplingParameters(1000, 10, 5000);
	std::vector<std::vector<unsigned int> > states;
	unsigned int sample_count = 5000;
	ginf.Sample(states, sample_count);

	// Map the ground truth samples to the unrolled tree's variable indices
	std::vector<std::vector<unsigned int> > states_tu(sample_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		std::vector<unsigned int> smp(5);
		smp[0] = states[si][0];
		smp[1] = states[si][1];
		smp[2] = states[si][2];
		smp[3] = states[si][2];	// 3 is copy of 2
		smp[4] = states[si][1];	// 4 is copy of 1
		states_tu[si] = smp;
	}

	// Change model parameters
	std::fill(pt_A->Weights().begin(), pt_A->Weights().end(), 0.0);
	std::fill(pt_B->Weights().begin(), pt_B->Weights().end(), 0.0);
	std::fill(pt_C->Weights().begin(), pt_C->Weights().end(), 0.0);

	// Exact model: reconstruct weights from population by MLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;
	for (unsigned int si = 0; si < states.size(); ++si) {
		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg_cyc, new Grante::FactorGraphObservation(states[si])));

		inference_methods.push_back(
			new Grante::BruteForceExactInference(&fg_cyc));
	}
	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.Train(1.0e-8);

	std::cout << std::endl;
	std::cout << "MLE exact cyclic model reconstruction, parameters"
		<< std::endl;
	for (unsigned int wi = 0; wi < w_A.size(); ++wi) {
		std::cout << "  A  dim " << wi << ": truth " << w_A[wi]
			<< ", learned " << pt_A->Weights()[wi] << std::endl;
	}
	for (unsigned int wi = 0; wi < w_B.size(); ++wi) {
		std::cout << "  B  dim " << wi << ": truth " << w_B[wi]
			<< ", learned " << pt_B->Weights()[wi] << std::endl;
	}
	for (unsigned int wi = 0; wi < w_C.size(); ++wi) {
		std::cout << "  C  dim " << wi << ": truth " << w_C[wi]
			<< ", learned " << pt_C->Weights()[wi] << std::endl;
	}

	// Compare marginals
	fg_cyc.ForwardMap();
	bfinf.PerformInference();
	std::vector<std::vector<double> > marg_mle = bfinf.Marginals();
	std::cout << "MLE exact cyclic model reconstruction, marginals"
		<< std::endl;
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mle[fi][wi] << std::endl;
			BOOST_CHECK_CLOSE_ABS(marg_true[fi][wi], marg_mle[fi][wi], 0.025);
		}
	}

	///
	/// Tree-unrolled learning
	///
	std::fill(pt_A->Weights().begin(), pt_A->Weights().end(), 0.0);
	std::fill(pt_B->Weights().begin(), pt_B->Weights().end(), 0.0);
	std::fill(pt_C->Weights().begin(), pt_C->Weights().end(), 0.0);
	fg_tu.ForwardMap();

	// TU model: reconstruct weights from population by MTPLE
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data_tu;
	std::vector<Grante::InferenceMethod*> inference_methods_tu;
	std::vector<std::vector<unsigned int> > cond_var_set_tu;
	for (unsigned int si = 0; si < states_tu.size(); ++si) {
		training_data_tu.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				&fg_tu, new Grante::FactorGraphObservation(states_tu[si])));

		std::vector<unsigned int> cv_tu;
		cv_tu.push_back(3);
		cv_tu.push_back(4);
		cond_var_set_tu.push_back(cv_tu);
		inference_methods_tu.push_back(new Grante::TreeInference(&fg_tu));
	}
	Grante::MaximumTreePseudoLikelihood mtple(&model);
	mtple.SetupTrainingData(training_data_tu, cond_var_set_tu,
		inference_methods_tu);
	mtple.Train(1.0e-8);

	// Display parameters
	std::cout << std::endl;
	std::cout << "MTPLE model reconstruction, parameters"
		<< std::endl;
	for (unsigned int wi = 0; wi < w_A.size(); ++wi) {
		std::cout << "  A  dim " << wi << ": truth " << w_A[wi]
			<< ", learned " << pt_A->Weights()[wi] << std::endl;
	}
	for (unsigned int wi = 0; wi < w_B.size(); ++wi) {
		std::cout << "  B  dim " << wi << ": truth " << w_B[wi]
			<< ", learned " << pt_B->Weights()[wi] << std::endl;
	}
	for (unsigned int wi = 0; wi < w_C.size(); ++wi) {
		std::cout << "  C  dim " << wi << ": truth " << w_C[wi]
			<< ", learned " << pt_C->Weights()[wi] << std::endl;
	}

	// Compare marginals within exact model
	fg_cyc.ForwardMap();
	bfinf.PerformInference();
	std::vector<std::vector<double> > marg_mtple = bfinf.Marginals();
	std::cout << "MTPLE model reconstruction, marginals"
		<< std::endl;
	for (unsigned int fi = 0; fi < marg_true.size(); ++fi) {
		std::cout << "Factor " << fi << std::endl;
		for (unsigned int wi = 0; wi < marg_true[fi].size(); ++wi) {
			std::cout << "   true " << marg_true[fi][wi]
				<< ", learned " << marg_mtple[fi][wi] << std::endl;
		}
	}

	// Delete objects
	for (unsigned int n = 0; n < training_data.size(); ++n) {
		delete (training_data[n].second);
		delete (inference_methods[n]);
	}
	for (unsigned int n = 0; n < training_data_tu.size(); ++n) {
		delete (training_data_tu[n].second);
		delete (inference_methods_tu[n]);
	}
}
