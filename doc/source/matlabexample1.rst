
Matlab Example: First steps
---------------------------

.. code-block:: matlab

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

	% Infer
	[fg_infer, logz_result] = grante_infer(model, factor_graphs, 'treeinf');

The above example shows how to construct a simple factor graph with two
variables and three factors.  The meaning of the properties is described in
detail in the documentation of the "grante_infer\", "grante_learn\", and
"grante_sample\" functions.

