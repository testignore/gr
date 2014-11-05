
% Test: inference on a simple model using data-specified factors

clear;

model=[];
model.factor_types=struct();
model.factor_types(1).name='unary1';
model.factor_types(1).card = [2];
model.factor_types(1).weights = [];
model.factor_types(2).name='unary2';
model.factor_types(2).card = [2];
model.factor_types(2).weights = [];
model.factor_types(3).name='pairwise';
model.factor_types(3).card = [2 2];
model.factor_types(3).weights = [];

%  #--(1)--#--(2)--#
factor_graphs=struct();
factor_graphs(1).card=[2 2];
factor_graphs(1).factors=struct();
factor_graphs(1).factors(1).type='unary1';
factor_graphs(1).factors(1).vars=[1];
factor_graphs(1).factors(1).data=[0.1, 0.7];
factor_graphs(1).factors(2).type='unary2';
factor_graphs(1).factors(2).vars=[2];
factor_graphs(1).factors(2).data=[0.3, 0.6];
factor_graphs(1).factors(3).type='pairwise';
factor_graphs(1).factors(3).vars=[1 2];
factor_graphs(1).factors(3).data=[0.0, 0.2; 0.3, 0.0];

% Infer
[fg_infer, logz_result] = grante_infer(model, factor_graphs, 'treeinf');

assert(abs(logz_result(1) - 0.4836311) < 1e-6);
assert(norm(fg_infer.marginals{1} - [0.6639461; 0.3360538]) < 1e-6);
assert(norm(fg_infer.marginals{2} - [0.5813064; 0.4186935]) < 1e-6);
assert(norm(fg_infer.marginals{3} - [0.4132795, 0.2506666; 0.1680269, 0.1680269]) < 1e-6);

