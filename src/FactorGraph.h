
#ifndef GRANTE_FACTORGRAPH_H
#define GRANTE_FACTORGRAPH_H

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

#include "FactorGraphModel.h"
#include "FactorDataSource.h"
#include "FactorGraphObservation.h"

namespace Grante {

/* One specific factor graph, instantiating a FactorGraphModel.
 * A FactorGraph stores: the number and cardinalities of variables,
 *    the factors.
 */
class FactorGraph {
public:
	// model: the factor graph model.  We do not assume ownership but the
	//    pointer has to remain valid during the lifetime of this object.
	// card: The number of variables in the model and their cardinalities.
	FactorGraph(const FactorGraphModel* model,
		const std::vector<unsigned int>& card);
	~FactorGraph();

	const FactorGraphModel* Model() const;
	const std::vector<Factor*>& Factors() const;
	const std::vector<unsigned int>& Cardinalities() const;

	// Perform forward map: update energies upon model change
	void ForwardMap();

	void EnergiesRelease();

	// Evaluate energy for a given fully observed configuration
	double EvaluateEnergy(const std::vector<unsigned int>& state) const;
	double EvaluateEnergy(const std::vector<std::vector<double> >& exp) const;
	double EvaluateEnergy(const FactorGraphObservation* obs) const;

	// Multiply all current energies with the given factor
	void ScaleEnergies(double factor);

	// Add a factor.  The factor's factortype must be looked up through
	// model->FindFactorType.  The FactorGraph takes ownership of the passed
	// object.
	void AddFactor(Factor* factor);

	// Add a data source.  The FactorGraph takes ownership of the passed
	// object, but will never change its content.
	void AddDataSource(const FactorDataSource* datasource);

	void Save(const std::string& filename) const;
	static FactorGraph* Load(const std::string& filename);

	void Print() const;

private:
	// The model this factorgraph instantiates
	const FactorGraphModel* model;

	// Cardinalities of variables.  The number of variables is implicitly
	// given.
	std::vector<unsigned int> cardinalities;

	// The factors of the model
	std::vector<Factor*> factors;

	// Data source objects if data is shared among multiple factors
	std::vector<const FactorDataSource*> datasources;

	FactorGraph();

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & const_cast<FactorGraphModel* &>(model);
		ar & cardinalities;
		ar & factors;
		// TODO: is this correctly handled?
		// ar & datasources;
	}
};

}

#endif

