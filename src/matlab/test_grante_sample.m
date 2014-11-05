
% Test: sampling from a simple model using shared factors

clear;

model=[];
model.factor_types=struct();
model.factor_types(1).name='unary1';
model.factor_types(1).card = [2];
model.factor_types(1).weights = [0.1, 0.7];
model.factor_types(2).name='unary2';
model.factor_types(2).card = [2];
model.factor_types(2).weights = [0.3, 0.6];
model.factor_types(3).name='pairwise';
model.factor_types(3).card = [2 2];
model.factor_types(3).weights = [0.0, 0.2; 0.3, 0.0];

%  #--(1)--#--(2)--#
factor_graphs=struct();
factor_graphs(1).card=[2 2];
factor_graphs(1).factors=struct();
factor_graphs(1).factors(1).type='unary1';
factor_graphs(1).factors(1).vars=[1];
factor_graphs(1).factors(2).type='unary2';
factor_graphs(1).factors(2).vars=[2];
factor_graphs(1).factors(3).type='pairwise';
factor_graphs(1).factors(3).vars=[1 2];

% Sample using exact tree inference
[states] = grante_sample(model, factor_graphs, 'treeinf', 50000);
M1 = [mean(states(1,:)==1), mean(states(1,:)==2)];
M2 = [mean(states(2,:)==1), mean(states(2,:)==2)];
M12 = [mean(states(1,:)==1 & states(2,:)==1), ...
	mean(states(1,:)==1 & states(2,:)==2); ...
	mean(states(1,:)==2 & states(2,:)==1), ...
	mean(states(1,:)==2 & states(2,:)==2)];

% Check sample approximation against exact marginals
assert(norm(M1' - [0.6639461; 0.3360538]) < 1e-2);
assert(norm(M2' - [0.5813064; 0.4186935]) < 1e-2);
assert(norm(M12 - [0.4132795, 0.2506666; 0.1680269, 0.1680269]) < 1e-2);

% Sample using Gibbs sampling
[states] = grante_sample(model, factor_graphs, 'gibbs', 50000);
M1 = [mean(states(1,:)==1), mean(states(1,:)==2)];
M2 = [mean(states(2,:)==1), mean(states(2,:)==2)];
M12 = [mean(states(1,:)==1 & states(2,:)==1), ...
	mean(states(1,:)==1 & states(2,:)==2); ...
	mean(states(1,:)==2 & states(2,:)==1), ...
	mean(states(1,:)==2 & states(2,:)==2)];

M1
M2
M12
assert(norm(M1' - [0.6639461; 0.3360538]) < 1e-2);
assert(norm(M2' - [0.5813064; 0.4186935]) < 1e-2);
assert(norm(M12 - [0.4132795, 0.2506666; 0.1680269, 0.1680269]) < 1e-2);

% Sample using multi-chain Gibbs sampling
options=[];
options.verbose=1;
[states] = grante_sample(model, factor_graphs, 'mcgibbs', 50000, options);
M1 = [mean(states(1,:)==1), mean(states(1,:)==2)];
M2 = [mean(states(2,:)==1), mean(states(2,:)==2)];
M12 = [mean(states(1,:)==1 & states(2,:)==1), ...
	mean(states(1,:)==1 & states(2,:)==2); ...
	mean(states(1,:)==2 & states(2,:)==1), ...
	mean(states(1,:)==2 & states(2,:)==2)];

M1
M2
M12
assert(norm(M1' - [0.6639461; 0.3360538]) < 1e-2);
assert(norm(M2' - [0.5813064; 0.4186935]) < 1e-2);
assert(norm(M12 - [0.4132795, 0.2506666; 0.1680269, 0.1680269]) < 1e-2);

