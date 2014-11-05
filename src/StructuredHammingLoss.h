
#ifndef GRANTE_STRUCTUREDHAMMINGLOSS_H
#define GRANTE_STRUCTUREDHAMMINGLOSS_H

#include <vector>

#include "StructuredLossFunction.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* Hamming-type loss: deviation from target label is penalized independently
 * across variables by a variable-specific constant.
 */
class StructuredHammingLoss : public StructuredLossFunction {
public:
	// XXX: right now, only discrete observations are supported
	// y_truth: this object takes ownership.
	// penalty_weights: The total penalty distributed over all variable
	//    labels.
	StructuredHammingLoss(const FactorGraphObservation* y_truth,
		const std::vector<double>& penalty_weights);
	// Assume a constant penalty of one.
	explicit StructuredHammingLoss(const FactorGraphObservation* y_truth);

	virtual ~StructuredHammingLoss();

	virtual double Eval(const std::vector<unsigned int>& y1_state) const;
	virtual void PerformLossAugmentation(FactorGraph* fg,
		double scale = 1.0) const;

private:
	std::vector<double> penalty_weights;
};

}

#endif

