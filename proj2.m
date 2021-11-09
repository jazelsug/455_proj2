% Name: proj2.m
% Author: Jazel A. Suguitan
% Last Modified: Nov. 8, 2021

clc, clear, close all

%================= SET PARAMETERS ===============

maxepisodes = 6; 
num_nodes = 10;  %Randomly generate nodes
n = 2; %number of dimensions
nodes = 50.*rand(num_nodes,n)+50.*repmat([0 1],num_nodes,1);
statelist   = BuildStateList(num_nodes);  % the list of states
actionlist  = BuildActionList(); % the list of actions
nstates     = length(statelist);
nactions    = length(actionlist); 
epsilon_learning = 0.0001;   % probability of a random action selection for e-greedy policy
delta_t = 0.003; 
t = 0:delta_t:3.1;

%Note: Q-table states are indexed at 1, but actual states start at 0
Q_initial = BuildQtableStorage(num_nodes, nstates, nactions); %load('Qcell_4actions2.mat'); %FOR 4 ACTIONS CASE
Q_update = Q_initial;
%SAVE DATA FOR EVALUATION
Connectivity_episodes = cell(1, maxepisodes);
Connectivity_episodes_learning = cell(1, maxepisodes);
R_all_episodes = cell(1, maxepisodes);
A_sum_cooQ_episodes = cell(1, maxepisodes);
Topo_eva_all_epi = cell(1, maxepisodes);
mean_Delta_Q_epi = cell(1, maxepisodes);

%================= START ITERATION ===============

for i=1:maxepisodes
    nodes = 50.*rand(num_nodes,n)+50.*repmat([0 1],num_nodes,1);
    %Training
%    [Q_update, Connectivity, Connectivity_learning, R_all, A_sum_cooQ, mean_Delta_Q]  = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t);
    Q_update = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t);
%     %Save data
%     Connectivity_episodes{i} = Connectivity;
%     Connectivity_episodes_learning{i} = Connectivity_learning; %CHECK - is there a point for this??
%     R_all_episodes{i} = R_all;
%     A_sum_cooQ_episodes{i} = A_sum_cooQ;
%     mean_Delta_Q_epi{i} = mean_Delta_Q; 
%     disp(['Espisode: ',int2str(i),]) 
    %decrease probability of a random action selection (for e-greedy selection)
    epsilon_learning = epsilon_learning * 0.99;
end
% 
% %======================== PLOTS ===========================
% 
% %Plot connectivity and action selection evaluations, respectively (LAST EPISODE)
% figure(3),plot(Connectivity)
% title('Network Connectivity over the last learning episode')
% grid on
% %Plot connectivity and action selection evaluations, respectively (WHOLE EPISODES)
% Con_epi_mat = cell2mat(Connectivity_episodes); 
% figure(5), plot(Con_epi_mat)
% title('Network Connectivity over learning episode')
% grid on
% 
% [A_diff0, index_A_cooQ] = find(A_cooQ_matrix>0);
% figure(8), plot(A_cooQ_matrix(index_A_cooQ))
% title('Action Selection over learning episodes')
% grid on 
% %Plot total reward
% R_all_epi_mat = cell2mat(R_all_episodes);
% [R_all_diff0, index_R]= find(R_all_epi_mat>0);
% figure(9), plot(R_all_epi_mat(index_R));
% title('Total reward over learning episodes')
% grid on
% %Plot reward in last episode
% [R_all_diff0, index_R_all]= find(R_all>0);
% figure(10), plot(R_all(index_R_all))
% title('Total Reward in the last episode')
% grid on


%================= FUNCTIONS ===============

%EDIT - orig. fxn signature below
% function [Q_update, Connectivity, Connectivity_learning, ...
%     R_all, A_sum_cooQ, mean_Delta_Q] = Q_Learning(Q_update, ...
%     statelist, actionlist, nstates, nactions, num_nodes, ...
%     n, nodes, epsilon_learning, delta_t, t)

function Q_update = Q_Learning(Q_update, ...
    statelist, actionlist, nstates, nactions, num_nodes, ...
    n, nodes, epsilon_learning, delta_t, t)
%     The Q-Learning reinforcement learning algorithm.
%     
%     Parameters
%     -------------
%     Q_update : double matrix
%         The current Q-table
%     statelist : double array
%         The encoded states corresponding with number of neighbors (0-9, indexed at 1)
%     actionlist : double array
%         The encoded actions corresponding with the safe places (1-4)
%     nstates : double
%         Number of states
%     nactions : double
%         Number of actions
%     num_nodes : double
%         Number of nodes
%     n : double
%         Number of dimensions
%     nodes : double matrix
%         Positions of nodes (num_nodes x n)
%     epsilon_learning : double
%         Constant for epsilon-greedy action selection
%     delta_t : double
%         Time step
%     t : double array
%         Simulation time
%         
%     Returns
%     --------------
%     Q_update : double matrix
%         The updated Q-table
%     Connectivity : double array
%         Connectivity values over this episode
%     Connectivity_learning : ????
%         ??????
%     R_all : double matrix
%         Reward values for all nodes over this episode
%     A_sum_cooQ : double matrix
%         Actions taken for all nodes over this episode
%     mean_Delta_Q : double matrix
%         Changes in Q-table values for all nodes

    p_nodes = zeros(num_nodes,n);   % Set initial velocties of MSN
    nodes_old = nodes; %KEEP privious positions of MSN
    d = 15; %Set desired distance among sensor nodes
    r = 30; %Set active range of nodes
    epsilon = 0.1;  %Set a constant for sigma norm
    alpha = 0.95;
    gamma = 0.05;
    p_nodes = zeros(num_nodes,n);   % Set initial velocties of MSN
    
    %CHECK - following 2 may have to declared outside fxn
    q_nodes_all = cell(size(t,2),num_nodes);
    p_nodes_all = cell(size(t,2),num_nodes);
    
    s_t = [];   %Keep track of initial states of nodes
    a_next = []; %Keep track of actions selected by nodes
    
    [Nei_agent, A] = findNeighbors(nodes, r); %Determine neighbors for each node
    
    %Set positions of safe places
    safe_places = [];
    for act = 1:nactions
        safe_places(act,:) = actionToPoint(act);
    end

    for i = 1:num_nodes
        s_t(i) = length(Nei_agent{i}) + 1;  %Node's initial state = number of neighbors, +1 because states are indexed at 1
        a_next(i) = select_action(Q_update{i}, s_t(i), epsilon_learning, nactions); %Node selects an action
    end
    
    for iteration = 1:length(t)
        %Plot safe places
        plot(safe_places(:,1),safe_places(:,2),'ro','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r', 'MarkerSize',4.2)
        hold on
        
        [Nei_agent, A] = findNeighbors(nodes, r);
        [Ui] = inputcontrol_Algorithm2(nodes, Nei_agent, num_nodes, epsilon, r, d, p_nodes, n, a_next); %last param used to be qt1
        p_nodes = (nodes - nodes_old)/delta_t; %COMPUTE velocities of sensor nodes
        p_nodes_all{iteration} = p_nodes; %SAVE VELOCITY OF ALL NODES
        nodes_old = nodes;
        nodes = nodes_old + p_nodes*delta_t  + Ui*delta_t*delta_t /2;
        q_nodes_all{iteration} = nodes;
        %Connectivity(iteration)= (1/(num_nodes))*rank(A);
        
        %================= PLOT and LINK SENSOR TOGETHER ===============
        plot(nodes(:,1),nodes(:,2), '.')
        hold on
        plot(nodes(:,1),nodes(:,2), 'k>','LineWidth',.2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5)
%         hold off
        for node_i = 1:num_nodes
            tmp=nodes(Nei_agent{node_i},:);
            for j = 1:size(nodes(Nei_agent{node_i},1))
                line([nodes(node_i,1),tmp(j,1)],[nodes(node_i,2),tmp(j,2)]) 
            end
        end
        hold off
    end
    
    %================= ACTION TAKEN ===============

    s_next = [];    %Keep track of new states of nodes
    [Nei_agent, A] = findNeighbors(nodes, r); %Determine neighbors for each node
    for i = 1:num_nodes
        s_next(i) = length(Nei_agent{i}) + 1;  %Node's initial state = number of neighbors, +1 because states are indexed at 1
        connect = (1/(num_nodes))*rank(A);
        reward = s_next(i); %Reward correlates to number of neighbors at end of episode
        %reward = connect*10;
        newMax = max(Q_update{i}(s_next(i),:));  %get max reward of new state from Q-table
        Q_update{i}(s_t(i),a_next(i)) = Q_update{i}(s_t(i),a_next(i)) + alpha * (reward + gamma*newMax - Q_update{i}(s_t(i),a_next(i))); %Update node's q table
    end
end

function [Ui] = inputcontrol_Algorithm2(nodes, Nei_agent, num_nodes, epsilon, r, d, p_nodes, dimensions, a_nexts)
%     Function for generating the Ui controller of the MSN.
%     
%     Parameters
%     -------------
%     nodes : double matrix (100x2)
%         Matrix of node positions in x-y coordinates
%     Nei_agent : cell array (100x1)
%         A container holding the neighbor indices for each node
%     num_nodes : double
%         The number of nodes in the MSN
%     epsilon : double
%         A constant for sigma norm
%     r : double
%         The interaction range of the nodes in the MSN
%     d : double
%         The desired distance among nodes in the MSN
%     p_nodes : double matrix (100x2)
%         The velocities of nodes, given in x and y directions
%     dimensions : double
%         The number of dimensions in which the MSN is operating
%     a_nexts : double array
%         The targets (actions) for each node
%         
%     Returns
%     -------------
%     [Ui] : double matrix (100x2)
%         Controls the positions of the nodes in the MSN as time progresses
    
    % Set constants
    c1_alpha = 67; %ORIGINALLY 30
    c2_alpha = 2*sqrt(c1_alpha);
    c1_mt = 1.1;    % ORIGINALLY 1.1
    Ui = zeros(num_nodes, dimensions);  % initialize Ui matrix to all 0's
    gradient = 0.;  % Initialize gradient part of Ui equation
    consensus = 0.; % Initialize consensus part of Ui equation
    feedback = 0.;  % Initialize navigational feedback of Ui equation
    
    % Sum gradient and consensus values for each node i
    for i = 1:num_nodes
        q_mt = actionToPoint(a_nexts(i)); %Get target for node based off its a_next value
        
        % EDIT - commented section below for initial simplified Ui
%         for j = 1:length(Nei_agent{i})
%             % i refers to node i
%             % j refers to the jth neighbor of node i
%             phi_alpha_in = sigmaNorm(nodes(Nei_agent{i}(j),:) - nodes(i,:), epsilon);
%             gradient = gradient + phi_alpha(phi_alpha_in, r, d, epsilon) * nij(nodes(i,:), nodes(Nei_agent{i}(j),:), epsilon);
%             consensus = consensus + aij(nodes(i,:), nodes(Nei_agent{i}(j),:), epsilon, r) * (p_nodes(Nei_agent{i}(j),:) - p_nodes(i,:));
%         end
        feedback = nodes(i,:) - q_mt;
        %Ui(i,:) = (c1_alpha * gradient) + (c2_alpha * consensus) - (c1_mt * feedback);   % Set Ui for node i using gradient, consensus, and feedback
        Ui(i,:) = -(c1_mt * feedback);  %EDIT - simplified Ui
        gradient = 0;
        consensus = 0;
        feedback = 0;
    end
end

function q_mt = actionToPoint(action)
    q_mt = [0,0];
    switch action
        case 1
            q_mt = [540,400];
        case 2
            q_mt = [540,0];
        case 3
            q_mt = [300,400];
        case 4
            q_mt = [300,0];
    end
end

function [Nei_agent, A] = findNeighbors(nodes, range)
%     Function for determining the neighbors of each node in a collection of nodes.
%     
%     Parameters
%     ------------
%     nodes : double matrix (100x2)
%         Matrix of node positions in x-y coordinates
%     range : double
%         The interaction range of the nodes in the MSN
%         
%     Returns
%     ------------
%     Nei_agent : cell array (100x1)
%         A container holding the neighbor indices for each node
%     A : double matrix (100x100)
%         The adjacency matrix of nodes

    num_nodes = size(nodes, 1);
    Nei_agent = cell(num_nodes, 1);  % Initialize cell array to hold indices of neighbors
    
    % Iterate through each node i
    for i = 1:num_nodes
        for j = 1:num_nodes
           % Check each node j if it's a neighbor of node i
           q1 = [nodes(i,1) nodes(i,2)];    % Set q1 with node i values
           q2 = [nodes(j,1) nodes(j,2)];    % Set q2 with node j values
           dist = norm(q1-q2);  % Euclidean norm of q1 and q2
           if i~= j && dist <= range && dist ~= 0
              Nei_agent{i} = [Nei_agent{i} j];  %Add j to list of i's neighbors
           end
        end
    end
    
    A = adjMatrix(nodes, Nei_agent); % Use adjMatrix function to obtain adjacency matrix
end

function a = select_action(Q, S, epsilon, num_actions)
%     A function for selecting the action for the robot to take. Uses epsilon-greedy policy.
%     
%     Parameters
%     ------------
%     Q : double matrix
%         The current Q table
%     S : double
%         The current state
%     epsilon : double
%         A small constant used for epsilon-greedy policy
%     num_actions : double
%         The number of actions the robot can take
%         
%     Returns
%     ------------
%     a : double
%         The selected action the robot will take

    n = randi([0,1]);
    if n < epsilon
        a = randi([1,num_actions]);
    else
        [maxReward, max_actions] = max(Q(S,:));
        a = max_actions(1);
    end
end

function states = BuildStateList(n)
%     Builds the list of states for RL.
%     
%     Parameters
%     -------------
%     n : double
%         The number of nodes in the MSN
%     
%     Returns
%     -------------
%     states : double array
%         The encoded states, determined by the number of neighbors a node
%         can have
     
    states = 0:n-1;
end

function actions = BuildActionList()
%     Builds the list of actions for RL.
%     
%     Returns
%     --------------
%     actions : double array
%         The encoded actions, corresponding with safe places for robots to go to
    
    actions = 1:4;
end

function Q = BuildQtableStorage(num_nodes, nstates, nactions)
    Q = cell(1, num_nodes);
    for i = 1:num_nodes
       Q{i} = rand(nstates, nactions); %initialize Q table with arbitrary values between 0 and 1
    end
end

function [A] = adjMatrix(nodes, Nei_agent)
%     Function for obtaining the adjacency matrix for a set of nodes. Used
%     for calculating Connectivity in the MSN.
% 
%     Parameters
%     -------------
%     nodes : double matrix (100x2)
%           Matrix of node positions in x-y coordinates
%     Nei_agent : cell array
%           A container holding the neighbor indices for each node
%     
%     Returns
%     -------------
%     [A] : double matrix (100x100)
%           The adjacency matrix of nodes

    num_nodes = size(nodes, 1);
    A = zeros(num_nodes);   % Initialize matrix with 0s

    for i = 1:num_nodes
       for j = 1:num_nodes
           if ismember(j, Nei_agent{i})
               % Node i and node j are neighbors
                A(i,j) = 1; % Set value in matrix to 1
           end
       end
    end
end