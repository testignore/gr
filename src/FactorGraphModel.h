
#ifndef GRANTE_FACTORGRAPHMODEL_H
#define GRANTE_FACTORGRAPHMODEL_H

#include <vector>
#include <string>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "FactorType.h"

namespace Grante {

/* Description of a generic factor graph template, with different factor
 * types.  This is not a factor graph instance (FactorGraph).
 *
 * The model stores: factortypes, learnable/learned parameters
 */
class FactorGraphModel {
public:
	FactorGraphModel();
	~FactorGraphModel();

	// The model takes ownership of the factortype
	void AddFactorType(FactorType* ft);
	FactorType* FindFactorType(const std::string& name);
	const std::vector<FactorType*>& FactorTypes() const;

	// Serialization: load and save a factor graph model
	void Save(const std::string& filename) const;
	static FactorGraphModel* Load(const std::string& filename);

private:
	std::vector<FactorType*> factortypes;

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & factortypes;
	}
};

}

#endif

