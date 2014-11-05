
#ifndef GRANTE_STRUCTUREDLOSSFUNCTION_H
#define GRANTE_STRUCTUREDLOSSFUNCTION_H

#include <vector>

#include "FactorGraph.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* Structured loss function \Delta interface.  The structured loss is used for
 * MAP-based learning methods such as the structured SVM.
 *
 * We restrict the loss functions to symmetric, semi-metric,
 * factor-decomposable loss functions.
 */
class StructuredLossFunction {
public:
	// The object does not take ownership of the FactorGraphObservation
	explicit StructuredLossFunction(const FactorGraphObservation* y_truth);
	virtual ~StructuredLossFunction();

	// Evaluate \Delta(y1,y_truth), where y_truth is given by an observation
	// (discrete or expectation) and y1 is a discrete labeling.  If the loss
	// function is factorgraph-specific, such as when reweighting is
	// incorporated, this must be hidden from this interface.
	virtual double Eval(const std::vector<unsigned int>& y1_state) const = 0;

	// Perform linear loss augmentation of the energies of the factor graph.
	// The resulting MAP problem is the following loss-augmented MAP problem:
	//   argmin_y E(y;x_n,w)-E(y_n;x_n,w)-\Delta(y,y_n)
	virtual void PerformLossAugmentation(FactorGraph* fg,
		double scale) const = 0;

	const FactorGraphObservation* Truth() const;

protected:
	const FactorGraphObservation* y_truth;
};

}

#endif

