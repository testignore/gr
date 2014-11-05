
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "FactorGraphModel.h"

namespace Grante {

FactorGraphModel::FactorGraphModel() {
}

FactorGraphModel::~FactorGraphModel() {
	for (unsigned int n = 0; n < factortypes.size(); ++n)
		delete (factortypes[n]);
}

void FactorGraphModel::AddFactorType(FactorType* ft) {
	factortypes.push_back(ft);
}

FactorType* FactorGraphModel::FindFactorType(
	const std::string& name) {
	for (unsigned int n = 0; n < factortypes.size(); ++n) {
		if (factortypes[n]->Name() == name)
			return (factortypes[n]);
	}
	return (0);
}

const std::vector<FactorType*>& FactorGraphModel::FactorTypes() const {
	return (factortypes);
}

void FactorGraphModel::Save(const std::string& filename) const {
	std::ofstream ofs(filename.c_str());
	{
		boost::archive::text_oarchive oa(ofs);
		oa << *this;
	}
}

FactorGraphModel* FactorGraphModel::Load(const std::string& filename) {
	FactorGraphModel* fgm = new FactorGraphModel();
	{
		std::ifstream ifs(filename.c_str());
		boost::archive::text_iarchive ia(ifs);
		ia >> *fgm;
	}
	return (fgm);
}

}

