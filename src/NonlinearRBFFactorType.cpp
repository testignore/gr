
#include <ctime>

#include <boost/random.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

#include "NonlinearRBFFactorType.h"

namespace Grante {

NonlinearRBFFactorType::NonlinearRBFFactorType(const std::string& name,
	const std::vector<unsigned int>& card,
	unsigned int data_size, unsigned int rbf_basis_count, double log_beta)
	: FactorType(name, card, data_size), rbfnet(rbf_basis_count, data_size),
		rbf_basis_count(rbf_basis_count) {
	InitializeProdCard();
	assert(rbf_basis_count > 0);

	assert((boost::math::isnan)(log_beta) == false);
	rbfnet.FixBeta(log_beta);
	size_t wdim = prod_card * rbfnet.ParameterDimension();

	// Initialize weight vector randomly
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	// FIXME: better initialization concepts
	w.resize(wdim);
	std::fill(w.begin(), w.end(), 0.0);
	size_t wbase = 0;
	for (unsigned int ri = 0; ri < prod_card; ++ri) {
		// Initialize alpha_n
		for (unsigned int wi = 0; wi < rbf_basis_count; ++wi)
			w[wbase + wi] = randu() - 0.5;

		// Initialize c_n
		for (unsigned int wi = 0; wi < (data_size*rbf_basis_count); ++wi)
			w[wbase + rbf_basis_count + wi] = randu() - 0.5;

		wbase += rbfnet.ParameterDimension();
	}
}

NonlinearRBFFactorType::~NonlinearRBFFactorType() {
}

void NonlinearRBFFactorType::InitializeUsingTrainingData(const std::vector<
	ParameterEstimationMethod::labeled_instance_type>& training_data) {
	boost::mt19937 rgen(static_cast<const boost::uint32_t>(std::time(0))+1);
	boost::uniform_real<double> rdestu;	// range [0,1]
	boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu(rgen, rdestu);

	size_t wbase = 0;
	for (unsigned int ri = 0; ri < prod_card; ++ri) {
		// 1. Collect all factors that are labeled with the corresponding
		// ground truth label
		std::vector<Factor*> m_factors;
		for (unsigned int n = 0; n < training_data.size(); ++n) {
			const FactorGraph* fg = training_data[n].first;
			const FactorGraphObservation* obs = training_data[n].second;
			assert(obs->Type() == FactorGraphObservation::DiscreteLabelingType);
			const std::vector<Factor*>& factors = fg->Factors();
			for (unsigned int fi = 0; fi < factors.size(); ++fi) {
				if (factors[fi]->Type()->Name() != Name())
					continue;

				unsigned int ei_obs =
					factors[fi]->ComputeAbsoluteIndex(obs->State());
				if (ei_obs != ri)
					continue;

				m_factors.push_back(factors[fi]);
			}
		}
		// Need to be sure there is at least one observation
		assert(m_factors.size() >= 1);
		std::cout << m_factors.size() << " samples for statepair "
			<< ri << std::endl;

		// Initialize alpha_n
		for (unsigned int wi = 0; wi < rbf_basis_count; ++wi)
			w[wbase + wi] = 1.0;

		// Initialize c_n as sample from the training set
		size_t wbi_base = 0;
		for (unsigned int bi = 0; bi < rbf_basis_count; ++bi) {
			unsigned int mi = static_cast<unsigned int>(
				randu() * static_cast<double>(m_factors.size()));
			assert(mi < m_factors.size());
			const std::vector<double>& H = m_factors[mi]->Data();

			// Copy selected training instance, perturbed
			assert(H.size() == data_size);
			for (unsigned int wi = 0; wi < data_size; ++wi) {
				w[wbase + rbf_basis_count + wbi_base + wi] =
					H[wi] + randu()*1.0e-8;
			}
			wbi_base += data_size;
		}
		wbase += rbfnet.ParameterDimension();
	}
	assert(wbase == w.size());
}

void NonlinearRBFFactorType::InitializeWeights(
	const std::vector<double>& weights) {
	assert(weights.size() == (prod_card * rbfnet.ParameterDimension()));
	this->w = weights;
}

bool NonlinearRBFFactorType::IsDataDependent() const {
	return (true);
}

void NonlinearRBFFactorType::ForwardMap(const Factor* factor,
	std::vector<double>& energies) const {
	const std::vector<double>& H = factor->Data();
	assert(H.size() == data_size);
	assert(energies.size() == prod_card);
	size_t wbase = 0;
	for (size_t ei = 0; ei < prod_card; ++ei) {
		energies[ei] = rbfnet.Evaluate(H, w, wbase);
		wbase += rbfnet.ParameterDimension();
	}
}

void NonlinearRBFFactorType::BackwardMap(const Factor* factor,
	const std::vector<double>& marginals,
	std::vector<double>& parameter_gradient, double mult) const {
	const std::vector<double>& H = factor->Data();
	assert(H.size() == data_size);
	size_t wbase = 0;
	for (size_t ei = 0; ei < prod_card; ++ei) {
		rbfnet.EvaluateGradient(H, w, parameter_gradient, wbase,
			mult * marginals[ei]);
		wbase += rbfnet.ParameterDimension();
	}
}

const RBFNetwork& NonlinearRBFFactorType::Net() const {
	return (rbfnet);
}

}

