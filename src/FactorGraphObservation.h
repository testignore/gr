
#ifndef GRANTE_FACTORGRAPHOBSERVATION_H
#define GRANTE_FACTORGRAPHOBSERVATION_H

#include <vector>

namespace Grante {

/* A 'label' or expected distribution of labels.  This class represents truth
 * our training samples.
 */
class FactorGraphObservation {
public:
	enum ObservationType {
		DiscreteLabelingType = 0,
		ExpectationType,
	};

	// Construct a discrete labeling observation.  The labeling is copied.
	//
	// observed_state: Discrete labeling of a set of variables.
	explicit FactorGraphObservation(const std::vector<unsigned int>&
		observed_state);

	// Construct an marginally expected labeling observation.  The
	// distributions are copied.
	//
	// observed_expectations: for each factor a marginal distribution has to
	//    be provided.
	explicit FactorGraphObservation(const std::vector<std::vector<double> >&
		observed_expectation);

	// Return the type of observation:
	//    DiscreteLabelingType: state vector is available,
	//    ExpectationType: marginal expected distribution for each factor is
	//       available.
	ObservationType Type() const;

	// Obtain state vector.  Use only in case Type()==DiscreteLabelingType
	const std::vector<unsigned int>& State() const;

	// Obtain marginal distributions.  Use only in case Type()=ExpectationType
	const std::vector<std::vector<double> >& Expectation() const;
	std::vector<std::vector<double> >& Expectation();

private:
	ObservationType type;
	std::vector<unsigned int> observed_state;
	std::vector<std::vector<double> > observed_expectation;
};

}

#endif

