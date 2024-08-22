function  SParEA(path)
%% =========================================================================================================================================
%% Code Name: SParEA - Surrogate-assisted Partial-evaluation based EA.
%------------------------------- Copyright ------------------------------------
% Copyright (c) MDO Group, UNSW, Australia. You are free to use the SParEA for
% research purposes. All publications which use this code should acknowledge the 
% use of "SParEA" and reference "K.H.Rahi, H.K.Singh, T.Ray, Partial Evaluation 
% Strategies for Expensive Evolutionary Constrained Optimization, IEEE
% Transactions on Evolutionary Computation, 2021 (Accepted for publication)".
%------------------------------------------------------------------------------ 
%% Developed and coded by 
%  Kamrul Hasan Rahi, MRes, UNSW, Canberra, Australia;
%  E-mail: kamrul.rahi@student.unsw.edu.au; kamrulhasanme038@gmail.com;
%  Contact no: +61 444 517 342; Web: www.mdolab.net/coremembers.html
%  Last updated: May 05, 2021.
%% ========================================================================================================================================
%%%%%%%%%%%%%%%%%%%%%%%  Loading the parameters  %%%%%%%%%%%%%%%%%%%%
load('Parameters.mat');
prob = feval(param.problem_name);
rng(param.seed,'twister');

%%%%%%%%%%%%%%%%%%%%%%%  Initializing the population  %%%%%%%%%%%%%%%%%%%%%%%
LB = prob.range(:,1)';UB = prob.range(:,2)';
X_pop = repmat(LB,param.pop_size,1)+(repmat(UB,param.pop_size,1)-repmat(LB,param.pop_size,1)).*(lhsdesign(param.pop_size,prob.nx));

%%%%%%%%%%%%%%%%%%%%%%%  Full evaluation of initial population  %%%%%%%%%%%%%%%%%%%%%%
[F_pop,G_pop] = feval(param.problem_name,X_pop);

%%%%%%%%%%%%%%%%%%%%%%%  Starting the evaluation Counter  %%%%%%%%%%%%%%%%%%%%%%%
[counter, cost_so_far] = cost_counter(F_pop,G_pop);
tapping_counter = [counter, cost_so_far];
if(~isempty(G_pop))
    CVpop = nansum(nanmax(G_pop,0),2);
else
    G_pop=[];CVpop = zeros(size(G_pop,1),1);
end
sol_id = 1:size(tapping_counter,1);
Archive = [zeros(param.pop_size,1) sol_id' X_pop F_pop G_pop CVpop tapping_counter];

%%%%%%%%%%%%%%%%%%%%%%%  Ranking the Initial Population  %%%%%%%%%%%%%%%%%%%%%%%
[X_pop, F_pop, G_pop,~] = IF(X_pop, F_pop, G_pop, param);

%%%%%%%%%%%%%%%%%%%%%%%  Setting the surrogate parameters and starting  generation loop  %%%%%%%%%%%%%%%%%%%%%%%
tempsurr = Surrogate(param);surr = [];k = 1;

while Archive(end,end) < param.mnfe
    %%%%%%%%%%%%%%%%%%%%%%%  Generate offspring  %%%%%%%%%%%%%%%%%%%%%%%
    X_child = genetic_operator(LB,UB,param,prob,X_pop);
    
    %%%%%%%%%%%%%%%%%%%%%%%  Uniqueness test of Offspring  %%%%%%%%%%%%%%%%%%%%%%%
    [X_child] = validity(Archive(:,3:2+prob.nx),X_child,param,prob,LB,UB);
    
    %%%%%%%%%%%%%%%%%%%%%%%  Train the surrogate model for each function  %%%%%%%%%%%%%%%%%%%%%%%
    if k == 1
        model_data = [];
        model_data.type = cell(1,prob.nf+prob.ng);
        model_data.model = cell(1,prob.nf+prob.ng);
        model_data.error = Inf * ones(1,prob.nf+prob.ng);
        for r=1:(prob.ng+prob.nf)
            surr_init = tempsurr;  % Assign blank surr
            surrset = set_range(surr_init, prob.range);  % Set range for surrogates;
            surradd = add_pop(surrset, prob, Archive(:,3:2+prob.nx+prob.nf+prob.ng),r);   % Add data to surrogate model
            [surr,model_data.type{r},model_data.model{r},model_data.error(r)] = train(surradd, param, r);  % Train the surrogate based on archive
            surr.model_data = model_data;
        end
    else
        repeat = find(counter(end,:)>0); % Preventing repeatation of model construction without updated solution
        for r = 1:size(repeat,2)
            surr_init = tempsurr;
            surrset = set_range(surr_init, prob.range);
            surradd = add_pop(surrset, prob, Archive(:,3:2+prob.nx+prob.nf+prob.ng),repeat(r));
            [surr,model_data.type{repeat(r)},model_data.model{repeat(r)},model_data.error(repeat(r))] = train(surradd, param, repeat(r));
        end
        surr.model_data = model_data;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%  Predict child's fitness and local improvement  %%%%%%%%%%%%%%%%%%%%%%%
    [F_child_s,G_child_s,X_child_star,F_child_star,G_child_star] = LocalImpr(X_child,surr,param,prob);    
    
    %%%%%%%%%%%%%%%%%%%%%%%  Select the candidate solutions  %%%%%%%%%%%%%%%%%%%%%%%
    [X_child,G_child_pred] = select_candidate(X_child,F_child_s,G_child_s,X_child_star,F_child_star,G_child_star,Archive(:,3:2+prob.nx),prob,surr);
    
    %%%%%%%%%%%%%%%%%%%%%%%  Sequencing based on constraint prediction in descending order  %%%%%%%%%%%%%%%%%%%%%%%
    [~, id_seq] = sort(G_child_pred,2,'descend');
    
    %%%%%%%%%%%%%%%%%%%%%%%  Partial Evaluation  %%%%%%%%%%%%%%%%%%%%%%%
    [F_child,G_child] = feval(param.problem_name,X_child);
    [F1_child,G1_child] = partial_eval(F_child,G_child,id_seq,prob);
    
    %%%%%%%%%%%%%%%%%%%%%%%  Deciding whether switch to full evaluation  %%%%%%%%%%%%%%%%%%%%%%%
    X1_pop = [X_pop;X_child];F1_pop = [F_pop;F1_child];G1_pop = [G_pop;G1_child];
    [~,~,~,rank1] = IF(X1_pop, F1_pop, G1_pop, param);
    [~,did] = intersect((1:param.pop_size)',rank1(1:param.pop_size),'rows','stable');
    if ~isequal(length(did),param.pop_size)
        F_child = F1_child;
        G_child = G1_child;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%  Update the evaluation counter and archive  %%%%%%%%%%%%%%%%%%%%%%%
    [counter, cost_so_far] = cost_counter(F_child,G_child);
    tapping_counter_1 = [counter, cost_so_far];
    tapping_counter = tapping_counter(end,:);
    tapping_counter = repmat(tapping_counter,size(tapping_counter_1,1),1);
    tapping_counter = tapping_counter+tapping_counter_1;
    if(~isempty(G_child))
        CVchild = nansum(nanmax(G_child,0),2);
    else
        G_child = [];CVchild = zeros(size(G_child,1),1);
    end
    sol_id = 1:size(tapping_counter,1);
    tmp = [X_child F_child G_child CVchild tapping_counter];
    Archive = [Archive; k*ones(size(tapping_counter,1),1) sol_id' tmp];
    
    %%%%%%%%%%%%%%%%%%%%%%%  Environmental selection  %%%%%%%%%%%%%%%%%%%%%%%
    [X_pop,F_pop,G_pop] = Reduce(X_pop,F_pop,G_pop,X_child,F_child,G_child,param);
    
    %%%%%%%%%%%%%%%%%%%%%%%  Display current state  %%%%%%%%%%%%%%%%%%%%%%%
    disp(strcat(path,'\generation-',num2str(k)));k = k+1;
end
%% Save Necessary Data
save('Archive.mat','Archive','-mat');
return
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%      THE END      %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%% This function is used to tap the number of function evaluation and it's associated cost in every generation.
function [counter,cost_so_far] = cost_counter(F_pop,G_pop)
counter = update_evaluation(F_pop,G_pop);
cost_so_far = sum(counter,2);
return

%% Counting Number of function evaluation
function [counter] = update_evaluation(F_pop,G_pop)
[N, M]=size(F_pop);[O, P]=size(G_pop);
G_counter=[];F_counter=[];
c=zeros(1,P);f=zeros(1,M);
for i=1:O
    for j=1:P
        if isinf(G_pop(i,j)) || isnan(G_pop(i,j))
            c(:,j) = c(:,j);
        else
            c(:,j) = c(:,j)+1;
        end
    end
    G_counter = [G_counter;c];
end
for i=1:O
    for k=1:M
        if  isinf(F_pop(i,k)) || isnan(F_pop(i,k))
            f(:,k)=f(:,k);
        else
            f(:,k)=f(:,k)+1;
        end
        F_counter=[F_counter;f];
    end
end

counter=[F_counter, G_counter];
return



%% Infeasibility driven ranking
function  [X_pop, F_pop, G_pop, rankfinal] = IF(X_pop, F_pop, G_pop,param)
N=size(X_pop,1);
overallid=(1:N);
G1_pop = G_pop;
if(~isempty(G_pop))
    G1_pop(G1_pop <0) = 0;
    cvsum = nansum(G1_pop,2);
    cvsum(~isreal(cvsum)) = Inf;
    feasible=find(cvsum == 0);
    infeasible=setdiff(overallid',feasible);
    infcvsum = cvsum(infeasible);
    [~,mincv] = min(infcvsum);
    mincvid = overallid(infeasible(mincv));
    NS = (-1).*sum((G1_pop(infeasible,:)==0),2);
else
    feasible=overallid';
    infeasible=[];
    cvsum=zeros(N,1);
end

% For the feasible solutions
ranks1 = [];
if ~isempty(feasible)
    feasiblepop = F_pop(feasible,:);
    %[front,id] = nd_sort_1(feasiblepop,(1:numel(feasible))'); % For Multi-objective Problems
    [~,id] = sort(feasiblepop); % For Single Objective Problems
    appended_list1 = feasible(id);
    ranks1 = appended_list1;
end

% For the infeasible solutions
ranks2 = [];
if ~isempty(infeasible)
    infeassize=round(param.infeasibility_ratio*param.pop_size);
    G_all = [NS infcvsum];
    [~,front_clm] = nd_sort(G_all,(1:size(G_all,1))');
    ranks2 = infeasible(front_clm);
    if length(ranks2)<2
        ranks2include=ranks2;
    else
        if length(ranks2)>max(infeassize,2)
            ranks2include = ranks2(1:max(infeassize,2));
        else
            ranks2include = ranks2;
        end
    end
end

% Assigned percentage of infeasible solutions on top
if ~isempty(infeasible)
    if (length(ranks1) + length(ranks2include)<=param.pop_size)
        rankfinal=[ranks2(1:(param.pop_size-length(ranks1))) ;ranks1];
    else
        rankfinal=[ranks2include ;ranks1(1:(param.pop_size-length(ranks2include)))];
    end
    rankfinal=rankfinal(1:param.pop_size);
else
    rankfinal = ranks1;
end

X_pop=X_pop(rankfinal,:);
F_pop=F_pop(rankfinal,:);

if(~isempty(G_pop))
    G_pop=G_pop(rankfinal,:);
else
    G_pop=[];
end
return


%% Infeasibility driven ranking
function  [pop_rank] = best_sort_IF(F_pop, G_pop)
N=size(F_pop,1);
overallid=(1:N);
G1_pop = G_pop;
if(~isempty(G_pop))
    G1_pop(G1_pop <0) = 0;
    cvsum = nansum(G1_pop,2);
    cvsum(~isreal(cvsum)) = Inf;
    feasible=find(cvsum == 0);
    infeasible=setdiff(overallid',feasible);
    infcvsum = cvsum(infeasible);
    NS = (-1).*sum((G1_pop(infeasible,:)==0),2);
else
    feasible=overallid';
    infeasible=[];
    cvsum=zeros(N,1);
end

% For the feasible solutions
ranks1 = [];
if ~isempty(feasible)
    feasiblepop = F_pop(feasible,:);
    %[front,id] = nd_sort_1(feasiblepop,(1:numel(feasible))'); % For Multi-objective Problems
    [~,id] = sort(feasiblepop); % For Single Objective Problems
    appended_list1 = feasible(id);
    ranks1 = appended_list1;
end

% For the infeasible solutions
ranks2 = [];
if ~isempty(infeasible)
    G_all = [NS infcvsum];
    [~,front_clm] = nd_sort(G_all,(1:size(G_all,1))');
    ranks2 = infeasible(front_clm);
end

if (length(ranks1) + length(ranks2)~=N)
    disp('Dimensional error occurs');
end
pop_rank = [ranks1 ; ranks2];
return


%% Simulated binary crossover and polynomial mutation
function [X_child] = genetic_operator(LB,UB,param,prob,X_parent_ordered)
[X_child] = crossover_SBX_matrix(LB,UB,param,prob,X_parent_ordered);
[X_child] = mutation_POLY_matrix(LB,UB,param,prob,X_child);
return

% Simulated Binary Crossover (SBX) operator
function [y1, y2] = op_SBX_matrix(l_limit,u_limit, x1, x2, eta)
y1 = x1;
y2 = x2;
ipos = find(abs(x1-x2) > 1e-6);
if ~isempty(ipos)
    x1_op = x1(ipos);
    x2_op = x2(ipos);
    l_op = l_limit(ipos);
    u_op = u_limit(ipos);
    pos_swap = x2_op < x1_op;
    tmp = x1_op(pos_swap);
    x1_op(pos_swap) = x2_op(pos_swap);
    x2_op(pos_swap) = tmp;
    r = rand(size(x1_op));
    beta = 1 + (2*(x1_op - l_op)./(x2_op - x1_op));
    alpha = 2 - beta.^-(eta+1);
    betaq = (1./(2-r.*alpha)).^(1/(eta+1));
    betaq(r <= 1./alpha) = (r(r <= 1./alpha).*alpha(r <= 1./alpha)).^(1/(eta+1));
    y1_op = 0.5 * (x1_op + x2_op - betaq.*(x2_op - x1_op));
    
    beta = 1 + 2*(u_op - x2_op)./(x2_op - x1_op);
    alpha = 2 - beta.^-(eta+1);
    betaq = (1./(2-r.*alpha)).^(1/(eta+1));
    betaq(r <= 1./alpha) = (r(r <= 1./alpha).*alpha(r <= 1./alpha)).^(1/(eta+1));
    y2_op = 0.5 * (x1_op + x2_op + betaq.*(x2_op - x1_op));
    
    y1_op(y1_op < l_op) = l_op(y1_op < l_op);
    y1_op(y1_op > u_op) = u_op(y1_op > u_op);
    
    y2_op(y2_op < l_op) = l_op(y2_op < l_op);
    y2_op(y2_op > u_op) = u_op(y2_op > u_op);
    
    pos_swap = (rand(size(x1_op)) <= 0.5);
    tmp = y1_op(pos_swap);
    y1_op(pos_swap) = y2_op(pos_swap);
    y2_op(pos_swap) = tmp;
    
    y1(ipos) = y1_op;
    y2(ipos) = y2_op;
end
return

function [c, fn_evals] = crossover_SBX_matrix(LB,UB,param,prob,p)
eta=param.distribution_mutation;
c = p; % parent size = no_solution*no_variable.
fn_evals = 0;
A = rand(size(p,1)/2,1);
is_crossover =[(A <= param.prob_crossover)';(A <= param.prob_crossover)'];
p_cross = p(is_crossover,:);
[N,m] = size(p_cross);
c_cross = p_cross;
p1_cross = p_cross(1:2:N,:);
p2_cross = p_cross(2:2:N,:);
B = rand(size(p_cross,1)/2,prob.nx);
l_limit = repmat(LB,size(p_cross,1)/2,1);
u_limit = repmat(UB,size(p_cross,1)/2,1);
cross_pos = (B <= 0.5);
l_cross = l_limit(cross_pos);
u_cross = u_limit(cross_pos);
p1 = p1_cross(cross_pos);
p2 = p2_cross(cross_pos);
c1 = p1_cross;
c2 = p2_cross;
[y1, y2] = op_SBX_matrix(l_cross,u_cross,p1,p2,eta);
c1(cross_pos) = y1;
c2(cross_pos) = y2;
c_cross(1:2:N,:) = c1;
c_cross(2:2:N,:) = c2;
c(is_crossover,:) = c_cross;
return

% Polynomial mutation operator: Matrix Form
function [x] = op_POLY_matrix(LB,UB,x,param)
param.distribution_mutation;
x_min = LB;
x_max = UB;
pos_mu = find(x_max > x_min);
if ~isempty(pos_mu)
    x_mu = x(pos_mu);
    x_min_mu = x_min(pos_mu);
    x_max_mu = x_max(pos_mu);
    delta1 = (x_mu - x_min_mu)./(x_max_mu - x_min_mu);
    delta2 = (x_max_mu - x_mu)./(x_max_mu - x_min_mu);
    mut_pow = 1/(param.distribution_mutation+1);
    rand_mu = rand(size(delta2));
    xy = 1 - delta2;
    val = 2*(1 - rand_mu) + 2*(rand_mu - 0.5).*xy.^(param.distribution_mutation+1);
    deltaq = 1 - val.^mut_pow;
    xy(rand_mu <= 0.5) = 1 - delta1(rand_mu <= 0.5);
    val(rand_mu <= 0.5) = 2*rand_mu(rand_mu <= 0.5) + (1-2*rand_mu(rand_mu <= 0.5)).* xy(rand_mu <= 0.5).^(param.distribution_mutation+1);
    deltaq(rand_mu <= 0.5) = val(rand_mu <= 0.5).^mut_pow - 1;
    
    x_mu = x_mu + deltaq.*(x_max_mu - x_min_mu);
    x_mu(x_mu < x_min_mu) = x_min_mu(x_mu < x_min_mu);
    x_mu(x_mu > x_max_mu) = x_max_mu(x_mu > x_max_mu);
    
    x(pos_mu) = x_mu;
end
return

function [p, fn_evals] = mutation_POLY_matrix(LB,UB,param,prob, p)
fn_evals = 0;
A = rand(size(p,1),prob.nx);
l_limit = repmat(LB,size(p,1),1);
u_limit = repmat(UB,size(p,1),1);
p_mut = p(A <= param.prob_mutation);
l_mut = l_limit(A <= param.prob_mutation);
u_mut = u_limit(A <= param.prob_mutation);
p_mut = op_POLY_matrix(l_mut,u_mut,p_mut,param);
p(A <= param.prob_mutation) = p_mut;
return


%% Checking similarity with archive population with current child pop.
function [v_X_child] = validity(X_pop,X_child,param,prob,LB,UB)
if ~isempty(X_child)
    X_child_nor = (X_child-repmat(LB,size(X_child,1),1))./repmat((UB-LB),size(X_child,1),1); % precision limit upto 5 digit with normalized solution
    X_child_r = round(X_child_nor,10);
    X_pop_nor = (X_pop-repmat(LB,size(X_pop,1),1))./repmat((UB-LB),size(X_pop,1),1);
    X_pop_r = round(X_pop_nor,10);
    [a,aid] = unique(X_child_r,'rows','stable');
    [b,b_id] = setdiff(a,X_pop_r,'rows','stable');
    if size(b,1)~=size(X_child,1)
        X_child_n = X_child(aid(b_id),:);
        N = param.pop_size - size(X_child_n,1);
        new_pop = repmat(LB,N,1)+repmat((UB-LB),N,1).*lhsdesign(N,prob.nx);
        v_X_child = [X_child_n;new_pop];
    else
        v_X_child = X_child;
    end
else
    v_X_child = [];
end
return


%% Candidate selection
function [new_child,new_g] = select_candidate(X_sur,F_sur,G_sur,X_loc,F_loc,G_loc,X_child,prob,surr)
LB = prob.range(:,1)';
UB = prob.range(:,2)';
[N,~] = size(X_sur);
G_sur1 = G_sur;
G_sur1(G_sur1<0) = 0;
cvsur = sum(G_sur1,2);
G_loc1 = G_loc;
G_loc1(G_loc1<0) = 0;
cvloc = sum(G_loc1,2);
combined_pop = [X_sur,F_sur,G_sur,cvsur;X_loc,F_loc,G_loc,cvloc];
F_pop = combined_pop(:,prob.nx+1:prob.nx+prob.nf);
CVpop = combined_pop(:,end);
[rank] = best_sort(F_pop,CVpop);

ranked_pop = combined_pop(rank,:);
ranked_pop_nor = (ranked_pop(:,1:prob.nx)-repmat(LB,size(ranked_pop,1),1))./repmat((UB-LB),size(ranked_pop,1),1);
ranked_pop_r = round(ranked_pop_nor,5);
[a,aid] = unique(ranked_pop_r,'rows','stable');
ranked_pop1 = ranked_pop(aid,:);
[un,un_id] = setdiff(a,X_child,'rows','stable');
if ~isempty(un)
    if size(un,1)>=N
        new_pop = ranked_pop1(un_id(1:N),:);
        new_child = new_pop(:,1:prob.nx);
        new_g = new_pop(:,prob.nx+prob.nf+1:prob.nx+prob.nf+prob.ng);
    else
        new_pop1 = ranked_pop1(un_id,:);
        new_child1 = new_pop1(:,1:prob.nx);
        new_g1 = new_pop1(:,prob.nx+prob.nf+1:prob.nx+prob.nf+prob.ng);
        required_N = N-size(un,1);
        new_child2 = repmat(LB,required_N,1)+repmat((UB-LB),required_N,1).*lhsdesign(required_N,prob.nx);
        for i=1:required_N
            new_g2(i,:) = pred_constr(new_child2(i,:),surr);
        end
        new_child = [new_child1;new_child2];
        new_g = [new_g1;new_g2];
    end
else
    new_child = repmat(LB,N,1)+repmat((UB-LB),N,1).*lhsdesign(N,prob.nx);
    for i=1:N
        new_g(i,:) = pred_constr(new_child(i,:),surr);
    end
end
return


%% Partial Evaluation
function [F_child_partial,G_child_partial] = partial_eval(F_child,G_child,id_seq,prob)
F_child_partial = [];
G_child_partial = [];
if ~isempty(G_child)
    for i = 1:size(G_child,1)
        g = NaN*ones(1,prob.ng);
        f = NaN*ones(1,prob.nf);
        for j = 1:prob.ng
            G_child_val = G_child(i,id_seq(i,j));
            if G_child_val>0
                g(:,id_seq(i,j)) = G_child_val;
                G_child_partial(i,:) = g;
                break;
            elseif G_child_val<=0
                g(:,id_seq(i,j)) = G_child_val;
                G_child_partial(i,:) = g;
            end
        end
        if sum(g(g>0)) == 0
            F_child_partial(i,:) = F_child(i,:);
        else
            F_child_partial(i,:) = f;
        end
    end
else
    G_child_partial = [];
    F_child_partial = F_child;
end
return


%% Sorting population following feasibility first rule with sumCV
function [rank] = best_sort(f,cv)
id = (1:numel(cv))';
j=1;l=1;Infeas=[];feasible=[];
for i=1:size(id,1)
    if(cv(id(i)) == 0)
        feasible(l)=id(i);
        l=l+1;
    else
        Infeas(j)=id(i);
        j=j+1;
    end
end

if(~isempty(feasible))
    if(size(f,2)==1)
        [~,I]=sort(f(feasible)); % for single objective
        feasible=feasible(I);
    else
        idfeas = (1:numel(feasible))';
        [~,I] = nd_sort(f(feasible,:),idfeas);
        feasible=feasible(I);
    end
end
if(~isempty(Infeas))
    vec=cv(Infeas);
    [~,I]=sort(vec);
    Infeas=Infeas(I);
end
rank=[feasible Infeas]';
return


%% Non-dominated sorting
function [fronts,idx] = nd_sort(f_all, id)
idx = [];
if isempty(f_all)
    fronts = [];
    return
end

if nargin == 1
    id = (1:size(f_all,1))';
end

if isempty(id)
    fronts = [];
    return
end

try
    fronts = nd_sort_c(id, f_all(id,:));
catch
    warning('ND_SORT() MEX not available. Using slower matlab version.');
    fronts = nd_sort_m(id, f_all(id,:));
end
for i = 1:size(fronts,2)
    if i == 1
        [ranks, dist] = sort_crowding(f_all, fronts(i).f);
        idx = [idx;ranks];
    else
        idx = [idx;(fronts(i).f)'];
    end
end

return


%% C-implementation of non-dominated sorting
function [F] = nd_sort_c(feasible, f_all)
[frontS, frontS_n] = ind_sort2(feasible', f_all');
F = [];
for i = 1:length(feasible)
    count = frontS_n(i);
    if count > 0
        tmp = frontS(1:count, i) + 1;
        F(i).f = feasible(tmp)';
    end
end
return


%% Matlab implementation of non-dominated sorting
function [F] = nd_sort_m(feasible, f_all)

front = 1;
F(front).f = [];

N = length(feasible);
M = size(f_all,2);

individual = [];
for i = 1:N
    id1 = feasible(i);
    individual(id1).N = 0;
    individual(id1).S = [];
end

% Assignging dominate flags
for i = 1:N
    id1 = feasible(i);
    f = repmat(f_all(i,:), N, 1);
    dom_less = sum(f <= f_all, 2);
    dom_more = sum(f >= f_all, 2);
    for j = 1:N
        id2 = feasible(j);
        if dom_less(j) == M && dom_more(j) < M
            individual(id1).S = [individual(id1).S id2];
        elseif dom_more(j) == M && dom_less(j) < M
            individual(id1).N = individual(id1).N + 1;
        end
    end
end

% identifying the first front
for i = 1:N
    id1 = feasible(i);
    if individual(id1).N == 0
        F(front).f = [F(front).f id1];
    end
end

% Identifying the rest of the fronts
while ~isempty(F(front).f)
    H = [];
    for i = 1 : length(F(front).f)
        p = F(front).f(i);
        if ~isempty(individual(p).S)
            for j = 1 : length(individual(p).S)
                q = individual(p).S(j);
                individual(q).N = individual(q).N - 1;
                if individual(q).N == 0
                    H = [H q];
                end
            end
        end
    end
    if ~isempty(H)
        front = front+1;
        F(front).f = H;
    else
        break
    end
end
return


%% Crowding distance to sort among solutions in same front
function [ranks, dist] = sort_crowding(f_all1, front_f)
f_all = f_all1;
L = length(front_f);
if L == 1
    ranks = front_f;
    dist = Inf;
else
    dist = zeros(L, 1);
    nf = size(f_all, 2);
    
    for i = 1:nf
        f = f_all(front_f, i);		% get ith objective
        [tmp, I] = sort(f);
        scale = f(I(L)) - f(I(1));
        dist(I(1)) = Inf;
        for j = 2:L-1
            id = I(j);
            id1 = front_f(I(j-1));
            id2 = front_f(I(j+1));
            if scale > 0
                dist(id) = dist(id) + (f_all(id2,i)-f_all(id1,i)) / scale;
            end
        end
    end
    dist = dist / nf;
    [tmp, I] = sort(dist, 'descend');
    ranks = front_f(I)';
end
return


%% Surrogate initialization
function [surr] = Surrogate(def)
% SURROGATE() creates surrogate model
surr.range = [];
surr.x = [];
surr.x_normal = [];
surr.y = [];
surr.nx = 0;
surr.ny = 0;
surr.count = 0;
surr.logger = [];
surr.model_data = [];

% Assign parameter values
surr.seed = def.seed;
surr.type = def.surr_type;
surr.max_traincount = 1000;
surr.train_ratio = 0.8;
surr.add_crit = 1.e-5;
return


function [surr] = set_range(surr, range)
% SET_RANGE() sets the range of decision variables
nx = size(range,1);
surr.range = range;
surr.nx = nx;
return


%% Add population to surrogate archive
function [surr] = add_pop(surr, prob, pop,r)
% ADD_POP() adds points from the population to the surrogate
eval_id = find(isnan(pop(:,prob.nx+r)));
[~,rest_id] = setdiff((1:size(pop,1))',eval_id,'rows','stable');
xpop = pop(rest_id,1:prob.nx);
FGpop = pop(rest_id,prob.nx+r);
y_pop = pop(rest_id,prob.nx+1:end);
n_total = length(xpop);
traincount = round(length(xpop) * surr.train_ratio);
if traincount > surr.max_traincount
    traincount = surr.max_traincount;
    n_max = min(n_total, round(traincount / surr.train_ratio));
    x = xpop(n_total-n_max+1:end,:);
    y = FGpop(n_total-n_max+1:end,:);
    ypop = y_pop(n_total-n_max+1:end,:);
else
    x = xpop;
    y = FGpop;
    ypop = y_pop;
end

% Need to add only solutions that are evaluated truly
surr = add_points(surr, x, y, ypop);
return


function [surr] = add_points(surr, x, y, y_pop)
% ADD_POINTS() adds observations (decision+response) for the surrogate model
if isempty(surr.range)
    error('Range of decision variables not defined.');
end

assert(size(x,1) == size(y,1));
assert(size(x,2) == surr.nx);

if surr.ny == 0
    surr.ny = size(y_pop,2);
else
    assert(size(y_pop,2) == surr.ny);
end

N = size(x,1);

% Verify that the new point is not in the neighborhood of archived points
for i = 1:N
    % ignore points with objectives/constraints set to inf
    x_normal = Normalize(surr, x(i,:));
    is_new = 1;
    if surr.count > 0
        for j = 1:surr.count
            if norm(surr.x_normal(j,:) - x_normal, 2) < surr.add_crit
                is_new = 0;
                break
            end
        end
    end
    if is_new == 1
        surr.x(surr.count+1,:) = x(i,:);
        surr.x_normal(surr.count+1,:) = x_normal;
        surr.y(surr.count+1,:) = y(i,:);
        surr.count = surr.count + 1;
    end
end
return

function [surr,type, model, error] = train(surr, param, r)
% TRAIN() trains surrogate model(s)
% Save random state
rand_state = rng(param.seed,'twister');
[type, model, error] = train_cluster(surr,r);
rng(rand_state);
return


function [type, model, error] = train_cluster(surr,r)
% identify the points used to train
N_total = size(surr.x,1);
rand_id = randperm(N_total);
row_num = N_total/10;
if row_num>1
    box = zeros(round(row_num)+1,10);
    count = 1;
    while size(rand_id,2)>0
        if size(rand_id,2)>10
            box(count,:) = rand_id(:,1:10);
            rand_id = setdiff(rand_id',box(count,:)','rows','stable')';
            count = count+1;
        else
            box(count,1:size(rand_id,2)) = rand_id;
            rand_id = setdiff(rand_id',box(count,1:size(rand_id,2))','rows','stable')';
        end
    end
else
    box = rand_id;
end

% Cross Validation
n = length(surr.type);
model_data = [];
model_data.type = cell(1,n);
model_data.model = cell(1,n);
model_data.error = Inf * ones(1,n);

for j = 1:n
    box_id = 1:size(box,2);
    train_id = setdiff(box_id,1);
    t_sol = box(:,train_id);
    t_ids = t_sol(t_sol~=0);
    v_sol = box(:,1);
    v_ids = v_sol(v_sol~=0);
    m_type = surr.type{j};
    [model_data.model{:,j}, model_data.error(:,j)] = train_model_single(surr, t_ids, v_ids, m_type, r);
    model_data.type{:,j} = m_type;
end
[~,id_error] = min(model_data.error);
type = model_data.type{:,id_error};
model = model_data.model{:,id_error};
error = model_data.error(:,id_error);
return
%%


%% Training the model
function [model, nrmse] = train_model_single(surr, t_ids, v_ids, type, r_id)
[surry,t_ids_m,v_ids_m] = modified_surr(surr,t_ids,v_ids,r_id);
x = surry.x_normal;
ttype = type;
train_func = strcat(ttype, '_train');
warning('off', 'all');
model = feval(train_func, x(t_ids_m,:), surry.y(t_ids_m));
warning('on', 'all');

% calculate normalized RMSE
if ~isempty(v_ids_m)
    [y, valid] = predict_model(surry, {type}, {model}, surry.x(v_ids_m,:));
    nrmse = normal_rms_error(surry, surry.y(v_ids_m), y);
else
    nrmse = 0;
end
return


function[surr_m,id1m,id2m] = modified_surr(surr,id1,id2,modelnum)
surry = surr.y;
surr_m = surr;
idreject = union(find(isnan(surry)),find(isinf(surry)),'stable');
id1m = setdiff(id1,idreject,'stable');
id2m = setdiff(id2,idreject,'stable');
require = surr.nx + 1 - numel(id1m);
available = numel(id2m);
if numel(id1m) < surr.nx+1 && available > 0
    leastavail = min(require,available);
    id1m = [id1m;id2m(1:leastavail,:)];
    id2m = id2m(leastavail+1:end,:);
end
return


function [y, valid] = predict_model(surr, type, model, x)
N = size(x,1);
m = length(type);
y = zeros(N, m);
valid = 1;
for i = 1:m
    if ~isempty(model{i})
        y(:,i) = predict_model_single(surr, type{i}, model{i}, x);
    else
        valid = 0;
        break
    end
end
return


function [y] = predict_model_single(surr, type, model, x)
x = Normalize(surr, x);
ptype = type;
pred_func = strcat(ptype, '_predict');
y = zeros(size(x,1), 1);
for i = 1:size(x,1)
    y(i) = feval(pred_func, x(i,:), model);
end
return


%% Calculate normalized RMS error
function [max_nrmse,varargout] = normal_rms_error(surr, y, ypred)
[N, m] = size(y);
nrmse = zeros(1,m);
for i = 1:m
    ydiff = y(:,i) - ypred(:,i);
    mse = (ydiff' * ydiff) / N;
    rmse = sqrt(mse);
    mm = minmax(y');
    delta = mm(2) - mm(1);
    if delta < 1.e-6, delta = 1; end
    nrmse(i) = rmse / delta;
end
max_nrmse = max(nrmse);
if nargout == 2
    varargout{1} = nrmse;
end
return


%% Fitness prediction of offspring and local improvement
function [F_surr,G_surr,X_star,F_star,G_star] = LocalImpr(X_child,surr,param,prob)
X_star = [];F_star = [];G_star = [];
F_surr = []; G_surr = [];
if ~isempty(X_child)
    for j = 1:size(X_child,1)
        X_child_in = X_child(j,:);
        % Prdeicted results from surrogates
        [F] = pred_obj(X_child_in,surr);
        [G,~] = pred_constr(X_child_in,surr);
        F_surr = [F_surr;F];
        G_surr = [G_surr;G];
        % Local search for improving child solution. Run SQP for local search
        [X_loc,F_loc,G_loc] = run_sqp(X_child_in,surr,param,prob.range);
        X_star = [X_star;X_loc];
        F_star = [F_star;F_loc];
        G_star = [G_star;G_loc];
    end
end
return


%% Normalization
function [x_normal] = Normalize(surr, x)
% NORMALIZE() normalizes x value between [eps,1]
eps = 1.e-4;
nx = size(surr.range,1);
x_normal = zeros(size(x));

range = surr.range;
for i = 1:nx
    if range(i,1) == range(i,2)
        x_normal(:,i) = ones(size(x,1),1);
    else
        x_normal(:,i) = eps + (x(:,i) - min(range(i,:))) * (1-eps) / (max(range(i,:)) - min(range(i,:)));
    end
    %         x_normal(:,i) = x(:,i);
end
return


%% Objective prediction
function [y] = pred_obj(x,surr)
datamodelsetup = surr.model_data;
datamodel = datamodelsetup.model;
typemodel = datamodelsetup.type;
type = typemodel{1};
model = datamodel{1};
x = Normalize(surr, x);
ptype = type;
pred_func = strcat(ptype, '_predict');
y = feval(pred_func, x, model);
return


%% Constraint prediction
function [y1,y2] = pred_constr(x,surr)
datamodelsetup = surr.model_data;
datamodel = datamodelsetup.model;
typemodel = datamodelsetup.type;
typeset = typemodel(2:end);
modelset = datamodel(2:end);
y1 = zeros(1,length(modelset));
x = Normalize(surr, x);
for i = 1:length(typeset)
    type = typeset{i};
    model = modelset{i};
    ptype = type;
    pred_func = strcat(ptype, '_predict');
    y1(:,i) = feval(pred_func, x, model);
end
y2 = [];
return


%% Local search
function [X_loc,F_loc,G_loc] = run_sqp(x0,surr,param,range)
param.ratio_ea = 0.8; % Last sqp option
options = optimoptions('fmincon','Algorithm','sqp','MaxIter',1000,'MaxFunEvals',1000,'display','off','OutputFcn',@outfun);
lb = range(:,1)';ub = range(:,2)';
tic;
x_star = fmincon(@(x)pred_obj(x,surr),x0,[],[],[],[],lb,ub,@(x)pred_constr(x,surr),options);
a = x_star-lb;b = x_star-ub;eps = 1e-3;
lb_eps = lb + eps;ub_eps = ub - eps;
if ~isempty(a(a<0)) || ~isempty(b(b>0))
    aa = a-abs(a);
    ida = find(aa~=0);
    x_star(:,ida) = lb_eps(:,ida);
    bb = ub-x_star;
    idb = find(bb<0);
    x_star(:,idb) = ub_eps(:,idb);
end
f_star = pred_obj(x_star,surr);
g_star = pred_constr(x_star,surr);
X_loc = x_star;F_loc = f_star;G_loc = g_star;
return

function stop = outfun(x,optimValues,state)
stop = false;
if toc > 300
    stop = true;
    disp('Stopping, time > 300sec')
end
return


%% Reduction
function [X,F,G] = Reduce(X_pop,F_pop,G_pop,X_child,F_child,G_child,param)
x = [X_pop;X_child];f = [F_pop;F_child];g = [G_pop;G_child];
[xx, ff, gg, ~] = IF(x, f, g, param);
X = xx(1:param.pop_size,:);F = ff(1:param.pop_size,:);G = gg(1:param.pop_size,:);
return