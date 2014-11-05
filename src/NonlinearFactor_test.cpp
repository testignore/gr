
#include <vector>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <cmath>

#include <boost/random.hpp>

#include "FactorGraph.h"
#include "FactorType.h"
#include "FactorGraphModel.h"
#include "TreeInference.h"
#include "MaximumLikelihood.h"
#include "NormalPrior.h"
#include "NonlinearRBFFactorType.h"
#include "RBFNetwork.h"

#define BOOST_TEST_MODULE(NonlinearFactorTest)
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE(MLEDataSimple)
{
	// Randomly set the data observations
	boost::mt19937 rgen(148894);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// Generate data set, 2D observations
	std::vector<std::vector<double> > X;
	std::vector<std::vector<unsigned int> > Y;
	std::vector<double> x_cur(2);
	unsigned int sample_count = 250;
	unsigned int class_count = 5;
	for (unsigned int n = 0; n < sample_count; ++n) {
		x_cur[0] = randu()-0.5;
		x_cur[1] = randu()-0.5;
		double norm = std::sqrt(x_cur[0]*x_cur[0] + x_cur[1]*x_cur[1]);
		x_cur[0] /= norm;
		x_cur[0] += 0.05*randu();
		x_cur[1] /= norm;
		x_cur[1] += 0.05*randu();
		X.push_back(x_cur);

		double a2 = std::atan2(x_cur[1], x_cur[0]) + 3.14159265358979323846;
		a2 /= 2.0*3.14159265358979323846;	// -> [0;1[
#if 0
		if (randu() >= 0.95) {	// 5% noise in labels
			a2 += randu();
			while (a2 >= 1.0)
				a2 -= 1.0;
		}
#endif
		a2 *= static_cast<double>(class_count);	// -> [0;class_count[
		unsigned int y_cur = static_cast<unsigned int>(a2);
		assert(y_cur < class_count);
		std::vector<unsigned int> y_cur_v(1);
		y_cur_v[0] = y_cur;
		Y.push_back(y_cur_v);
	}

	// Learn a non-linear multi-class logistic regression classifier
	Grante::FactorGraphModel model;
	std::vector<unsigned int> card;
	card.push_back(class_count);
	unsigned int rbf_basis_count = 4;
	Grante::NonlinearRBFFactorType* nlft =
		new Grante::NonlinearRBFFactorType("rbf_unary", card,
			/*data_size*/ 2, rbf_basis_count, /*log_beta*/ 1.0);
	model.AddFactorType(nlft);
	Grante::FactorType* pt = model.FindFactorType("rbf_unary");
	BOOST_REQUIRE(pt != 0);

	// Reconstruct model weights from population
	std::vector<Grante::ParameterEstimationMethod::labeled_instance_type>
		training_data;
	std::vector<Grante::InferenceMethod*> inference_methods;

	// Create a factor graph from the model: 3 binary variables
	std::vector<unsigned int> vc;
	vc.push_back(class_count);
	for (unsigned int si = 0; si < sample_count; ++si) {
		Grante::FactorGraph* fg = new Grante::FactorGraph(&model, vc);

		// Add factors
		std::vector<unsigned int> var_index(1);
		var_index[0] = 0;
		Grante::Factor* fac1 = new Grante::Factor(pt, var_index, X[si]);
		fg->AddFactor(fac1);

		training_data.push_back(
			Grante::ParameterEstimationMethod::labeled_instance_type(
				fg, new Grante::FactorGraphObservation(Y[si])));

		// Push the same inference object again (graph is of fixed-structure)
		Grante::TreeInference* tinf = new Grante::TreeInference(fg);
		inference_methods.push_back(tinf);
	}
	nlft->InitializeUsingTrainingData(training_data);

	Grante::MaximumLikelihood mle(&model);
	mle.SetupTrainingData(training_data, inference_methods);
	mle.AddPrior("rbf_unary",
		new Grante::NormalPrior(1.0, nlft->WeightDimension()));
	mle.Train(1.0e-5);

	unsigned int wbase = 0;
	const Grante::RBFNetwork& net = nlft->Net();
	std::vector<double>& weights = nlft->Weights();
	for (unsigned int ei = 0; ei < class_count; ++ei) {
		std::cout << "ei " << ei << std::endl;
		for (unsigned int ri = 0; ri < rbf_basis_count; ++ri) {
			std::cout << "   ri " << ri << ", alpha " << weights[wbase+ri]
				<< ", center (" << weights[wbase+rbf_basis_count+ri*2+0] << ","
				<< weights[wbase+rbf_basis_count+ri*2+1] << ")" << std::endl;
		}
		wbase += net.ParameterDimension();
	}

	// Perform prediction
	unsigned int train_err = 0;
	for (unsigned int si = 0; si < sample_count; ++si) {
		training_data[si].first->ForwardMap();
		inference_methods[si]->PerformInference();
		const std::vector<double>& marg = inference_methods[si]->Marginal(0);
#if 0
		std::cout << "sample " << si << ", y_truth " << Y[si][0] << ":";
		for (unsigned int ci = 0; ci < class_count; ++ci)
			std::cout << " " << marg[ci];
		std::cout << std::endl;
#endif
		unsigned int y_pred = std::max_element(marg.begin(), marg.end()) -
			marg.begin();
		if (y_pred != Y[si][0])
			train_err += 1;
	}
	std::cout << train_err << " of " << sample_count << " training errors"
		<< std::endl;
	BOOST_CHECK_LT(train_err, 20u);

	for (unsigned int si = 0; si < sample_count; ++si) {
		delete (inference_methods[si]);
		delete (training_data[si].first);
		delete (training_data[si].second);
	}
}

