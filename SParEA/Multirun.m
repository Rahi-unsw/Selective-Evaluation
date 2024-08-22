function Multirun
%------------------------------- Copyright ------------------------------------
% Copyright (c) MDO Group, UNSW, Australia. You are free to use the SParEA for
% research purposes. All publications which use this code should acknowledge the 
% use of "SParEA" and reference "K.H.Rahi, H.K.Singh, T.Ray, Partial Evaluation 
% Strategies for Expensive Evolutionary Constrained Optimization, IEEE
% Transactions on Evolutionary Computation, 2021 (Accepted for publication)".
%------------------------------------------------------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Multirun: This function is used to run specified algorithms and problems across multiple trials. 
%  Set the parameters first at the Params.m file properly.
%  Check that matlab path is added properly for all functions required (Check Startup.m file)
%  Now run this script.
%% Coded by 
%  Kamrul Hasan Rahi
%  k.rahi@student.adfa.edu.au; kamrulhasanme038@gmail.com
%  Last updated: May 05, 2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parallel Computing
clc;clear all;close all;warning off;
startup;Params;
path = cd;count = 1;

for i = 1:numel(param.allprobs)
    param.problem_name = param.allprobs{i};
    prob = feval(param.problem_name);
    for k = 1:param.no_runs(i)
        disp(strcat('SParEA:- Problem Name:- ',param.problem_name,', Trial:- ',num2str(k)));
        pth = strcat(cd,filesep,'Data',filesep,'SParEA',filesep,param.problem_name,filesep,'Trial-',num2str(k));
        mkdir(pth);
        cd(pth);  
        param.mnfe = 1000*(prob.nf+prob.ng);
        param.pop_size = 2*(prob.nx+1);
        param.seed = 100+k;
        save('Parameters.mat', 'param');
        path1{count} = pth;
        count = count+1;
        cd(path);
    end
end
cd(path);

clear param;
% parfor: utilize multiple core of PC.
numcores = feature('numcores');
parpool(numcores);
parfor i = 1:length(path1)
    cd(path1{i});
    disp(strcat('Running -> ',path1{i}));
    tic;
    SParEA(path1{i});
    toc;
end
delete(gcp);
cd(path);
return