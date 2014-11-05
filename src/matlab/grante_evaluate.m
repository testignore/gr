function [E]=grante_evaluate(model, factor_graph, states_or_marginals);
%GRANTE_EVALUATE Evaluate energy values of given samples under the model.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>
% Date: 25th January 2011
%
% Input
%    model: Factor graph model structure, as in grante_infer.
%    factor_graph: A single factor graph with V variables.
%    states_or_marginals:
%       either
%
%       states: (V,sample_count) double matrix with discrete values, each
%          column being one sample.  Values are in {1,2,...,L_i} where L_i is
%          the maximum label of the i'th variable.
%
%       or
%
%       marginals: (1,F) cellarray of arrays.  marginals{fi} contains the
%          marginal distribution for the factor facto_graph.factors(fi).  The
%          array marginals{fi} is a (Y_1,...,Y_K) double array.  This can be
%          obtained from the probabilistic inference result returned by
%          grante_infer.
%
% Output
%    E: for states syntax:
%       (1,sample_count) energy values of samples under the model.
%       for marginals syntax:
%       (1,1) expected energy under the model.
E=mex_grante_evaluate(model,factor_graph,states_or_marginals);

