
% Test: learn from observational data

clear;

% Setup model of factor graph:
%   1. 'unary' factor type,
%   2. 'pairwise' factor type.
model=[];
model.factor_types=struct();
model.factor_types(1).name = 'unary';
model.factor_types(1).card = [2];
model.factor_types(1).weights = [0.0 0.0];
model.factor_types(2).name = 'pairwise';
model.factor_types(2).card = [2 2];
model.factor_types(2).weights = [0.0 0.0 0.0 0.0];

% Setup factor graph with three unary and two pairwise factors:
%
%    #--(1)--#--(2)
%        |       |
%        #       #
%        |
%       (3)--#
%
factor_graphs=struct();
factor_graphs(1).card=[2 2 2];
factor_graphs(1).factors=struct();
factor_graphs(1).factors(1).type='unary';
factor_graphs(1).factors(1).vars=[1];
factor_graphs(1).factors(2).type='unary';
factor_graphs(1).factors(2).vars=[2];
factor_graphs(1).factors(3).type='unary';
factor_graphs(1).factors(3).vars=[3];
factor_graphs(1).factors(4).type='pairwise';
factor_graphs(1).factors(4).vars=[1 2];
factor_graphs(1).factors(4).data=[];
factor_graphs(1).factors(5).type='pairwise';
factor_graphs(1).factors(5).vars=[1 3];
factor_graphs(1).factors(5).data=[];

% Six observations
observations=struct();
observations(1).id = 1;
observations(1).labels = [1 2 1];
observations(2).id = 1;
observations(2).labels = [2 2 1];
observations(3).id = 1;
observations(3).labels = [2 1 1];
observations(4).id = 1;
observations(4).labels = [1 1 2];
observations(5).id = 1;
observations(5).labels = [2 1 2];
observations(6).id = 1;
observations(6).labels = [1 1 2];

% Place a student-t prior on the unary weights and a Normal prior on the
% pairwise weights.
priors=struct();
priors(1).factor_type = 'unary';
priors(1).prior_name = 'studentt';
priors(1).prior_opt = [100.0 1.0];
priors(2).factor_type = 'pairwise';
priors(2).prior_name = 'normal';
priors(2).prior_opt = [1.0];
method='mle';

% Learn the model
[fg_model_trained] = grante_learn(model, factor_graphs, observations, priors, method);
[fg_infer, logz_result] = grante_infer(fg_model_trained, factor_graphs, 'treeinf');

