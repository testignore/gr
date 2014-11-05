
#ifndef GRANTE_CONDITIONING_H
#define GRANTE_CONDITIONING_H

#include <vector>

#include "FactorGraph.h"
#include "FactorConditioningTable.h"
#include "FactorGraphPartialObservation.h"

namespace Grante {

class Conditioning {
public:
	// Condition a factor graph on some discrete observations
	//
	// ftab: The storage object for conditioning factor types created.  To
	//    make conditioning completely transparent to inference and learning
	//    algorithms, factor types that are reduced by conditioning are stored
	//    in one place, a so called FactorConditioningTable.  When multiple
	//    Factor's are conditioned in the same way only one conditioned
	//    FactorType is created.
	// fg_base: The factor graph to be conditioned.  Note: you can only
	//    condition once and the factor graph returned by this method must not
	//    be conditioned again.
	// pobs: The partial observation describing the conditioned-on variables.
	// var_new_to_orig: (Output) an index map such that
	//    var_new_to_orig[new_variable_index] = original_variable_index.  The
	//    vector is resized accordingly.
	// fac_new_to_orig: (Output, optional) an index map such that
	//    fac_new_to_orig[new_factor_index] = original_factor_index.  The
	//    vector is resized accordingly.
	//
	// Return the conditioned factor graph.
	static FactorGraph* ConditionFactorGraph(FactorConditioningTable* ftab,
		const FactorGraph* fg_base, const FactorGraphPartialObservation* pobs,
		std::vector<unsigned int>& var_new_to_orig);
	static FactorGraph* ConditionFactorGraph(FactorConditioningTable* ftab,
		const FactorGraph* fg_base, const FactorGraphPartialObservation* pobs,
		std::vector<unsigned int>& var_new_to_orig,
		std::vector<unsigned int>& fac_new_to_orig);
};

}

#endif

