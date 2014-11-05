
#ifndef GRANTE_RANDOMFACTORGRAPHGENERATOR_H
#define GRANTE_RANDOMFACTORGRAPHGENERATOR_H

#include <vector>

#include <boost/random.hpp>

#include "FactorGraphModel.h"
#include "FactorGraph.h"

namespace Grante {

/* Utility class to generate random factor graphs with special properties.
 * This is useful to testing and development of inference and learning
 * algorithms.
 */
class RandomFactorGraphGenerator {
public:
	explicit RandomFactorGraphGenerator(const FactorGraphModel* fg_model);

	/* Generate a tree-structured, connected factor graph with an expected
	 * factor type distribution 'ft_dist' and a total number of 'factor_count'
	 * factor nodes.  The number of variable nodes can vary.  Each variable is
	 * connected to at least one factor node.
	 */
	FactorGraph* GenerateTreeStructured(const std::vector<double>& ft_dist,
		unsigned int factor_count) const;

private:
	const FactorGraphModel* fg_model;

	// Random number generation, for the sampler
	boost::mt19937 rgen;
	boost::uniform_real<double> rdestu;	// range [0,1]
	mutable boost::variate_generator<boost::mt19937,
		boost::uniform_real<double> > randu;
};

}

#endif

