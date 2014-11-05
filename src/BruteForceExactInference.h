
#ifndef GRANTE_BRUTEFORCE_H
#define GRANTE_BRUTEFORCE_H

#include "InferenceMethod.h"

namespace Grante {

/* Inference by exhaustive enumeration.  This class is only useful for very
 * small instances and to debug approximate inference methods.
 */
class BruteForceExactInference : public InferenceMethod {
public:
	BruteForceExactInference(const FactorGraph* fg);

	virtual ~BruteForceExactInference();

	virtual InferenceMethod* Produce(const FactorGraph* fg) const;

	virtual void PerformInference();
	virtual void ClearInferenceResult();

	virtual const std::vector<double>& Marginal(
		unsigned int factor_id) const;
	virtual const std::vector<std::vector<double> >& Marginals() const;

	virtual double LogPartitionFunction() const;

	// TODO: not implemented yet
	virtual void Sample(std::vector<std::vector<unsigned int> >& states,
		unsigned int sample_count);

	virtual double MinimizeEnergy(std::vector<unsigned int>& state);

private:
	std::vector<std::vector<double> > marginals;
	double log_z;

	unsigned int StateCount() const;
	void InitializeState(std::vector<unsigned int>& state);
	bool AdvanceState(std::vector<unsigned int>& state);
};

}

#endif

