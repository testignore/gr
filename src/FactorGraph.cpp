
#include <iostream>
#include <algorithm>
#include <numeric>
#include <functional>
#include <fstream>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

#include "FactorGraph.h"

namespace Grante {

// private
FactorGraph::FactorGraph() {
}

FactorGraph::FactorGraph(const FactorGraphModel* model,
	const std::vector<unsigned int>& card)
	: model(model), cardinalities(card) {
}

FactorGraph::~FactorGraph() {
	for (unsigned int fi = 0; fi < factors.size(); ++fi)
		delete (factors[fi]);

	for (unsigned int dsi = 0; dsi < datasources.size(); ++dsi)
		delete (datasources[dsi]);
}

const FactorGraphModel* FactorGraph::Model() const {
	return (model);
}

const std::vector<Factor*>& FactorGraph::Factors() const {
	return (factors);
}

const std::vector<unsigned int>& FactorGraph::Cardinalities() const {
	return (cardinalities);
}

void FactorGraph::ForwardMap() {
	#pragma omp critical
	{
		// Simply perform forward map for each factor
		for (std::vector<Factor*>::iterator fi = factors.begin();
			fi != factors.end(); ++fi) {
			(*fi)->ForwardMap();
		}
	}
}

void FactorGraph::EnergiesRelease() {
	for (std::vector<Factor*>::iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		(*fi)->EnergiesRelease();
	}
}

double FactorGraph::EvaluateEnergy(
	const std::vector<unsigned int>& state) const {
	assert(state.size() == cardinalities.size());

	// Sum energy of all factors
	double energy = 0.0;
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		energy += (*fi)->EvaluateEnergy(state);
	}
	return (energy);
}

double FactorGraph::EvaluateEnergy(
	const std::vector<std::vector<double> >& exp) const {
	assert(exp.size() == factors.size());
	// Sum energy of all factors
	double energy = 0.0;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		assert(factors[fi]->Energies().size() == exp[fi].size());
		energy += std::inner_product(exp[fi].begin(), exp[fi].end(),
			factors[fi]->Energies().begin(), 0.0);
	}
	return (energy);
}

double FactorGraph::EvaluateEnergy(const FactorGraphObservation* obs) const {
	if (obs->Type() == FactorGraphObservation::DiscreteLabelingType)
		return (EvaluateEnergy(obs->State()));

	assert(obs->Type() == FactorGraphObservation::ExpectationType);
	return (EvaluateEnergy(obs->Expectation()));
}

void FactorGraph::ScaleEnergies(double factor) {
	for (std::vector<Factor*>::const_iterator fi = factors.begin();
		fi != factors.end(); ++fi) {
		std::vector<double>& fac_energies = (*fi)->Energies();
		// E[n] *= factor
		std::transform(fac_energies.begin(), fac_energies.end(),
			fac_energies.begin(),
			std::bind2nd(std::multiplies<double>(), factor));
	}
}

void FactorGraph::AddFactor(Factor* factor) {
	factors.push_back(factor);
}

void FactorGraph::AddDataSource(const FactorDataSource* datasource) {
	datasources.push_back(datasource);
}

void FactorGraph::Save(const std::string& filename) const {
	std::ofstream ofs(filename.c_str());
	{
		boost::archive::text_oarchive oa(ofs);
		oa << *this;
	}
}

FactorGraph* FactorGraph::Load(const std::string& filename) {
	FactorGraph* fg = new FactorGraph();
	{
		std::ifstream ifs(filename.c_str());
		boost::archive::text_iarchive ia(ifs);
		ia >> *fg;
	}
	return (fg);
}

void FactorGraph::Print() const {
	std::cout << "FACTORGRAPH BEGIN" << std::endl;
	std::cout << cardinalities.size() << " variables, "
		<< factors.size() << " factors" << std::endl;
	std::cout << "VARIABLES BEGIN" << std::endl;
	for (unsigned int vi = 0; vi < cardinalities.size(); ++vi) {
		std::cout << "  " << vi << " (" << cardinalities[vi]
			<< ") appears in factors:";
		for (unsigned int fi = 0; fi < factors.size(); ++fi) {
			if (std::find(factors[fi]->Variables().begin(),
				factors[fi]->Variables().end(), vi) ==
				factors[fi]->Variables().end())
				continue;
			std::cout << " " << fi;
		}
		std::cout << std::endl;
	}
	std::cout << "VARIABLES END" << std::endl;
	std::cout << "FACTORS BEGIN" << std::endl;
	for (unsigned int fi = 0; fi < factors.size(); ++fi) {
		std::cout << "  " << fi << " ("
			<< factors[fi]->Variables().size() << ", '"
			<< factors[fi]->Type()->Name() << "'), data size "
			<< factors[fi]->Data().size() << " acts on var:";
		for (unsigned int vi = 0; vi < factors[fi]->Variables().size(); ++vi) {
			std::cout << " " << factors[fi]->Variables()[vi];
		}
		std::cout << std::endl;
	}
	std::cout << "FACTORS END" << std::endl;
	std::cout << "FACTORGRAPH END" << std::endl;
}

}

