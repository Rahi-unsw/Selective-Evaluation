%%%%%%%%%Algorithm Name: SParEA (Surrogate-assisted Partial-evaluation based EA)%%%%%%%%%%%%%%%%%%%%

Please cite the following papers if you use this algorithm.

%------------------------------- Copyright ------------------------------------
% Copyright (c) MDO Group, UNSW, Australia. You are free to use the SParEA for
% research purposes. All publications which use this code should acknowledge the 
% use of "SParEA" and reference "K.H.Rahi, H.K.Singh, T.Ray, Partial Evaluation 
% Strategies for Expensive Evolutionary Constrained Optimization, IEEE
% Transactions on Evolutionary Computation, 2021 (Accepted for publication)".
%------------------------------------------------------------------------------ 

%%%%%Steps to run the code%%%%%%%%%%%%

1. Go to the Startup.m just to check both 'Problems' and 'Methods' folders are added correctly. If you just keep 'SParEA' folder name
   unchanged, you don't need to change anything.
2. Go to Params.m script. You can specify all the parameters. The demo parameters for 6 G-series and 1 engineering design problems are 
   given just for example. Please, write the problem scripts (g1.m, g6.m in the 'Problems' folder accordingly). All the parameters are kept in the 
   'param' structure.
3. Open the 'Method->mex' folder as your current folder. In the matlab command prompt, write mex('ind_sort2.c'). If mex file is built
   successfully, return to the SParEA folder.
4. You are now fully set to run the code. Run the Multirun.m script to run the algorithm. Note that, 'parfor' command will run 
   multiple trails parallelly according to 'number of workers' in your PC. Check PC system configuration->number of cores. If
   you don't want to use 'parfor', just replace it with 'for' command and comment 'delete(gcp)' command. That's it. Just run SParEA now.
5. All the data will be saved in a newly generated 'Data' folder. The format to store data is arranged as follows:
   Data->SParEA->Problem name->Trial number->Stored data (Archive.m, Parameters.m).


%%%%%Stored data format%%%%%%%%%%%%%%

1. Archive.m stores all data raw data in a (N*M) matrix during the whole optimization process. The format is as follows:
   N = Total evaluated solutions
   M = [Generation no	nth solution in current generation	variables (x1...xD)	objective values (f1-single objective)	constraint values (G1...GP)	Sum of constraint violation	Each individual function evaluation count	function evaluation so far]
3. Parameters.m stores all specified parameters.

 