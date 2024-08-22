%% Problem specific inputs
% Six (g-series) CEC 2006 benchmarks and one engineering design problem for the demo run.
param.allprobs = {'g1','g6','g7','g9','g10','g19','helical_spring',};
% Number of trials for each problem.
param.no_runs = [25,25,25,25,25,25,25];
%% General inputs
% Probability of crossover
param.prob_crossover = 1;
% Probability of mutation
param.prob_mutation = 0.1;
% Distribution index of crossover
param.distribution_crossover = 20;
% Distribution index of mutation
param.distribution_mutation = 20;
% Ratio of infeasible solutions in the population
param.infeasibility_ratio = 0.1;
% Surrogate variants
param.surr_type = {'dace' 'rbf' 'rsm2' 'rsm1'};