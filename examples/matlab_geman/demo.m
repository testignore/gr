
% Exact replication of the first denoising experiment in the classic
% Geman & Geman reference on Gibbs sampling.
%
% Author: Sebastian Nowozin <Sebastian.Nowozin@microsoft.com>

num_class=5;    % 5 labels for each pixel
addpath('../../src/matlab');

% Define model as a single factor type
model=[];
model.factor_types(1).name = 'pw';
model.factor_types(1).card = [num_class, num_class];

% Define pairwise energy of the form
strength = 1/3;
model.factor_types(1).weights = strength*ones(num_class,num_class);
model.factor_types(1).weights = model.factor_types(1).weights - ...
    2.0*diag(strength*ones(num_class,1));

% Original size by Geman and Geman
%ydim = 128;
%xdim = 128;

% Smaller size for this demo
xdim = 64;
ydim = 64;


clc;
disp(' ');
disp('Welcome to the image denoising demo.  This demo reproduces a famous experiment');
disp('of the 1983 Geman&Geman paper.');
disp(' ');
disp('In this demo we use a single factor type with attractive potentials.');
disp(['Each variable has ', num2str(num_class), ' states, and ', ...
	'there are ', num2str(xdim*ydim), ' variables on a ', ...
	num2str(xdim), '-by-', num2str(ydim), ' grid.']);

disp(['The factor type structure is:']);
model.factor_types(1)

disp(' ');
disp(['The factor weights are:']);
model.factor_types(1).weights

disp(' ');
disp('As you can see the diagonal energies are negative, the off-diagonal ones positive.');
disp('This is known as a "Potts" potential.');
disp(' ');
input('Please press <return> to continue...');

% Build factor graph
fg=[];
fg.card=num_class*ones(1, ydim*xdim);
%           |               --              \, /
num_factors=(ydim-1)*xdim + (xdim-1)*ydim + 2*(ydim-1)*(xdim-1);
vfg=cell(1,num_factors);
fg.factors=struct('type',vfg,'vars',vfg,'data',vfg);

fi=1;
for yi=1:ydim
	for xi=1:xdim
		var_id=(xi-1)*ydim+yi;	% column-major
		if yi > 1 && xi > 1
			% \
			fg.factors(fi).type='pw';
			fg.factors(fi).vars=[var_id-ydim-1, var_id];
			fg.factors(fi).data=[];
			fi=fi+1;

			% /
			fg.factors(fi).type='pw';
			fg.factors(fi).vars=[var_id-1, var_id-ydim];
			fg.factors(fi).data=[];
			fi=fi+1;
		end
		if yi > 1
			% |
			fg.factors(fi).type='pw';
			fg.factors(fi).vars=[var_id-1, var_id];
			fg.factors(fi).data=[];
			fi=fi+1;
		end
		if xi > 1
			% -
			fg.factors(fi).type='pw';
			fg.factors(fi).vars=[var_id-ydim, var_id];
			fg.factors(fi).data=[];
			fi=fi+1;
		end
	end
end
assert(num_factors == (fi-1));

disp(' ');
disp('We have just created a factor graph by instantiating the factor type on ');
disp(['a regular grid.  There are ', num2str(num_factors), ' factors.  ', ...
	'The factor graph defines a']);
disp('probability distribution over all possible labelings, assigning a');
disp('probability to each.');
disp(' ');
input('Please press <return> to continue...');

options=[];
options.gibbs_burnin = 200;
options.gibbs_spacing = 1;
[states] = grante_sample(model, fg, 'gibbs', 1, options);

% Display sample from the distribution
figure;imagesc(reshape(states, ydim, xdim)); axis image; colormap gray;
title('Gibbs sample from the distribution');

disp(' ');
disp('The figure displays a sample from the model distribution.');
disp('As you can see, the attractive factors make nearby pixels take the same state.');
disp(' ');
input('Please press <return> to continue...');

% Additive noise
noise_sigma = 1.5;
states_n = states + noise_sigma*randn(numel(states),1);

% Display sample with noise
figure;imagesc(reshape(states_n, ydim, xdim), [1,num_class]); axis image; colormap gray;
title('Noisy image');

disp(' ');
disp('We added independent Gaussian noise to each pixel intensity.');
disp('From this noisy image we will try to recover the original sample by');
disp('solving an inference problem using simulated annealing and the Gibbs');
disp('sampler.  This is possible because we know both the generating model');
disp('and the noise distribution.');
disp(' ');
input('Please press <return> to continue...');

% Setup posterior energy by adding unary terms
model.factor_types(2).name = 'u';
model.factor_types(2).card = [num_class];
model.factor_types(2).weights = [];
for yi=1:ydim
	for xi=1:xdim
		var_id=(xi-1)*ydim+yi;	% column-major
		fg.factors(fi).type='u';
		fg.factors(fi).vars=[var_id];
		fg.factors(fi).data=(((1:num_class)-states_n(var_id)).^2) ...
			./ (2.0*noise_sigma^2);	% (8.2) in Geman
		fi=fi+1;
	end
end
options.sa_steps = 25;
[result25, z_result] = grante_infer(model, fg, 'mapsa', options);
options.sa_steps = 300;
[result300, z_result] = grante_infer(model, fg, 'mapsa', options);

figure;
subplot(2,2,1);
imagesc(reshape(states, ydim, xdim)); axis image; colormap gray;
title('Gibbs sample from the distribution');

subplot(2,2,2);
imagesc(reshape(states_n, ydim, xdim), [1,num_class]); axis image; colormap gray;
title('Noisy image');

subplot(2,2,3);
imagesc(reshape(result25.solution, ydim, xdim)); axis image; colormap gray;
title('Denoised image (SA=25)');

subplot(2,2,4);
imagesc(reshape(result300.solution, ydim, xdim)); axis image; colormap gray;
title('Denoised image (SA=300)');

disp(' ');
disp('The figure shows the process and two denoising results using 25 and');
disp('300 simulated annealing sweeps, respectively.  As you can see, the');
disp('second result (SA=300) is quite good given how noisy the input is.');
disp(' ');
disp('This is the end of the demo.');

