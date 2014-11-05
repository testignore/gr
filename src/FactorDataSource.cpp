
#include "FactorDataSource.h"

namespace Grante {

FactorDataSource::FactorDataSource(const std::vector<double>& data)
	: H(data) {
}

FactorDataSource::FactorDataSource() {
}

FactorDataSource::FactorDataSource(const std::vector<double>& data,
	const std::vector<unsigned int>& data_sparse_index)
	: H(data), H_index(data_sparse_index) {
}

FactorDataSource::~FactorDataSource() {
}

bool FactorDataSource::IsSparse() const {
	return (H_index.empty() == false);
}

const std::vector<double>& FactorDataSource::Data() const {
	return (H);
}

const std::vector<unsigned int>& FactorDataSource::DataSparseIndex() const {
	return (H_index);
}

}

