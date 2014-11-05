function [states] = grante_sample(model, fg, method, sample_count, options);
% GRANTE_SAMPLE Produce one or more exact or approximate samples from a
% discrete distribution specified by a factor graph.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 19th March 2010
%
% Input
%    model: Factor graph model structure (Grante::FactorGraphModel) containing
%       the following fields,
%       .factor_types: (1,T) array of structures with elements
%          .name: string containing a factor type identifier.
%          .card: (1,K) array containing cardinalities of adjacent variables.
%             K denotes the order of this factor (>= 1).
%          .weights: (optional) (1,D) double array containing parameters of
%             this factor type.  If there is no .weights element or it is
%             empty, then this factor type does not have parameters.
%        Tying, symmetry-constraints, etc. are supported by adding the fields
%          .data_size: (1,1) scalar containing the data size of factors
%             instantiating this type.  Must be >=1.
%          .A: (optional) (1,prod(card)) double array containing integer
%             elements.  An element of -1 fixes that energy element to be
%             zero.  An integer element >=0 indexes a tied group of energies.
%             For example, for a pairwise factor between two variables with
%             three states each, we can enforce symmetric factors by setting
%                A=[0 1 2;
%                   1 3 4; 
%                   2 4 5],
%             We could enforce a Potts type of energy by setting
%                A=[-1  0  0;
%                    0 -1  0;
%                    0  0 -1],
%             forcing the diagonal to zero.  When using a non-empty .A field,
%             the .weights size changes.  It becomes
%             numel(unique(A(find(A(:)>=0))))*data_size.  Currently the use of
%             .A prevents using nonlinear factor types.
%        Nonlinear factor types are supported as follows, by setting the
%        additional fields
%          .type: string, one of
%             'rbfnet', Radial Basis Function network with exp basis
%                functions of fixed bandwidth, but variable weights and
%                centers.  For learning (grante_learn) the basis will be
%                initialized by a random training sample subsample.
%          .data_size: (1,1) scalar containing the data size of factors
%             instantiating this type.  Must be >=1.
%          .options: additional options for non-linear functions, for
%             different non-linear maps this differs as follows.
%             rbfnet: this is a (1,2) vector [rbf_basis_count, log_beta] where
%                rbf_basis_count >= 1 is the number of RBF basis functions per
%                factor energy index and log_beta is explained in
%                NonlinearRBFFactorType.h.
%    fg: structure describing a single factor graph (Grante::FactorGraph).
%       The structure contains the following fields:
%       .card: (1,V) array containing cardinalities of all V variables in the
%          factor graph.  For example, in the case of three binary variables,
%          this would be [2, 2, 2].
%       .factors: (1,F) array of structures, each containing the following
%          fields:
%          .type: either an index between 1 and T indexing into
%             model.factor_types, or a string with the name of the factor
%             type.
%          .vars: (1,K) array containing absolute variable indices mapping to
%             the K adjacent factor variables.  The variables must have the
%             same cardinalities as specified in model.factor_types(t).card.
%          .data: (optional) (M,1) data vector related to this factor
%             instance.  This is required only if the factor type has a
%             non-empty weight vector.  Refer to the documentation of
%             Grante::FactorType.  This element is allowed to be sparse.
%          .dsrc: (optional) (1,1) id indexing a datasource of this factor.
%             In case .dsrc is provided, .data must not be provided or must be
%             empty.
%       .datasources: (optional) (1,DS) array of structures containing data
%          vectors that are shareable across multiple factors.  Each structure
%          contains the following fields:
%          .id: An arbitrary index used to identify the data source within
%             this factor graph.
%          .data: (M,1) data vector.
%    method:
%       'treeinf' (for tree-structured factor graphs).
%       'gibbs' naive Gibbs sampling (for general factor graphs).
%       'mcgibbs', multiple chain naive Gibbs sampling with convergence
%          diagnostics (for general factor graphs).
%    sample_count: (1,1) double with integer value >0.  The number of samples
%       to produce.
%    options: (optional) structure with optional fields,
%       .gibbs_burnin: (1,1) scalar >=0, number of Gibbs burn in sweeps,
%          default: 100,
%       .gibbs_spacing: (1,1) scalar >=0, number of Gibbs spacing sweeps
%          between samples, default: 0.
%
%       .mcgibbs_maxpsrf: (1,1) scalar >1, maximum tolerated potential scale
%          reduction factor, where values closer to one are more stringend.
%          Default: 1.01.
%       .mcgibbs_chains: (1,1) integer >=2, number of Gibbs chains to run in
%          parallel.  Default: 5.
%       .mcgibbs_samples: (1,1) scalar >=1, number of Gibbs samples to obtain,
%          default: 1000,
%       .mcgibbs_spacing: (1,1) scalar >=0, number of Gibbs spacing sweeps
%          between samples, default: 0.
%
% Output
%    states: (V,sample_count) double matrix with discrete values, each column
%       being one sample.
if nargin < 5
	options = [];
end
[states] = mex_grante_sample(model, fg, method, sample_count, options);

