
#ifndef GRANTE_FACTORDATASOURCE_H
#define GRANTE_FACTORDATASOURCE_H

#include <vector>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace Grante {

/* Source for factor data.  This class has a number of purposes:
 *  1. For data arising from conditioning, a single object can provide the
 *     data for many factors, avoiding duplication and preserving memory,
 * (2. Factor data can be produced on the fly.)  TODO
 */
class FactorDataSource {
public:
	explicit FactorDataSource(const std::vector<double>& data);
	FactorDataSource(const std::vector<double>& data,
		const std::vector<unsigned int>& data_sparse_index);
	virtual ~FactorDataSource();

	virtual bool IsSparse() const;

	virtual const std::vector<double>& Data() const;
	virtual const std::vector<unsigned int>& DataSparseIndex() const;

private:
	std::vector<double> H;
	// If H_index.empty() == false, then H is specified as sparse vector
	std::vector<unsigned int> H_index;

	FactorDataSource();

	friend class boost::serialization::access;
	template<class Archive>
	void serialize(Archive& ar, const unsigned int version) {
		ar & H;
		ar & H_index;
	}
};

}

#endif

