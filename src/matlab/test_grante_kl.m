
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

%P=-2.0:0.05:2.0;
P=-4.0:0.5:2.0;
PKL=zeros(1,numel(P));
PKL_inf=zeros(1,numel(P));
PES=zeros(1,numel(P));
PEC=zeros(1,numel(P));
model_perturb=model;
for i=1:numel(P)
	model_perturb.factor_types(3).weights(1)=...
		model.factor_types(3).weights(1) + P(i);
	PKL(i)=grante_compute_kl(model, model_perturb, ...
		factor_graphs, factor_graphs);
	PKL_inf(i)=grante_compute_kl_inf(model, model_perturb, ...
		factor_graphs, factor_graphs, 'treeinf', []);
	[PES(i), PEC(i)]=grante_compute_estatistic(model, model_perturb, ...
		factor_graphs, factor_graphs);
end
figure;

% KL
subplot(1,4,1);
plot(P, PKL);
title('D_{KL}(p,p_{+e})');
xlabel('Perturbation e');
ylabel('KL');

% KL_inf
subplot(1,4,2);
plot(P, PKL_inf);
title('D_{KLinf}(p,p_{+e})');
xlabel('Perturbation e');
ylabel('KLinf');

% ES
subplot(1,4,3);
plot(P, PES);
title('D_{ES}(p,p_{+e})');
xlabel('Perturbation e');
ylabel('E-statistic');

% EC
subplot(1,4,4);
plot(P, PEC);
title('D_{EC}(p,p_{+e})');
xlabel('Perturbation e');
ylabel('E-coefficient');

