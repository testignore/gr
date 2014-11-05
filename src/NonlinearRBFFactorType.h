
#ifndef GRANTE_NONLINEARRBF_FACTORTYPE_H
#define GRANTE_NONLINEARRBF_FACTORTYPE_H

#include "FactorType.h"
#include "RBFNetwork.h"
#include "ParameterEstimationMethod.h"

namespace Grante {

/* A factor type that produces energies as non-linear responses of an RBF
 * network applied to the factor data.
 */
class NonlinearRBFFactorType : public FactorType {
public:
	// name, card: Same as in FactorType.
	// data_size: The size of the data vector stored in each factor of this
	//    type.
	// rbf_basis_count: Number of RBF basis functions per adjacent variable
	//    configuration.  Must be >= 1.
	// log_beta: Bandwidth parameter to basis function in log-domain.
	//    log_beta=0 is beta=1.
	NonlinearRBFFactorType(const std::string& name,
		const std::vector<unsigned int>& card,
		unsigned int data_size, unsigned int rbf_basis_count, double log_beta);

	virtual ~NonlinearRBFFactorType();

	// Heuristically initialize RBF basis from given labeled training data.
	// Each configuration of the factor must appear at least once in the
	// training data.  Right now, only discrete observations are supported.
	void InitializeUsingTrainingData(const std::vector<
		ParameterEstimationMethod::labeled_instance_type>& training_data);
	void InitializeWeights(const std::vector<double>& weights);

	virtual bool IsDataDependent() const;

	virtual void ForwardMap(const Factor* factor,
		std::vector<double>& energies) const;

	virtual void BackwardMap(const Factor* factor,
		const std::vector<double>& marginals,
		std::vector<double>& parameter_gradient, double mult = 1.0) const;

	const RBFNetwork& Net() const;

private:
	// rbfnet[ei] is the network producing energies[ei] from data and a subset
	// of the parameter vector.
	RBFNetwork rbfnet;
	unsigned int rbf_basis_count;
};

}

#endif

