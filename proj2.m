% Name: proj2.m
% Author: Jazel A. Suguitan
% Last Modified: Nov. 5, 2021

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
Q_initial = randi([1,5], nstates, nactions); %initialize Q table with arbitrary values %load('Qcell_4actions2.mat'); %FOR 4 ACTIONS CASE
Q_update = Q_initial;
%SAVE DATA FOR EVALUATION
Connectivity_episodes = cell(1, maxepisodes);
Connectivity_episodes_learning = cell(1, maxepisodes);
R_all_episodes = cell(1, maxepisodes);
A_sum_cooQ_episodes = cell(1, maxepisodes);
Topo_eva_all_epi = cell(1, maxepisodes);
mean_Delta_Q_epi = cell(1, maxepisodes);

% %================= START ITERATION ===============
% 
% for i=1:maxepisodes
%     nodes = 50.*rand(num_nodes,n)+50.*repmat([0 1],num_nodes,1);
%     %Training
%     [Q_update, Connectivity, Connectivity_learning, R_all, A_sum_cooQ, mean_Delta_Q]  = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t);
%     %Save data
%     Connectivity_episodes{i} = Connectivity;
%     Connectivity_episodes_learning{i} = Connectivity_learning; %CHECK - is there a point for this??
%     R_all_episodes{i} = R_all;
%     A_sum_cooQ_episodes{i} = A_sum_cooQ;
%     mean_Delta_Q_epi{i} = mean_Delta_Q; 
%     disp(['Espisode: ',int2str(i),]) 
%     %decrease probability of a random action selection (for e-greedy selection)
%     epsilon_learning = epsilon_learning * 0.99;
% end
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

function [Q_update, Connectivity, Connectivity_learning, ...
    R_all, A_sum_cooQ, mean_Delta_Q] = Q_Learning(Q_update, ...
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
%     R_all : double array
%         Reward values over this episode
%     A_sum_cooQ : double array
%         Actions taken over this episode
%     mean_Delta_Q : double array
%         Changes in Q-table values

end

function a = select_action(Q, S, epsilon, num_actions)
%     A function for selecting the action for the robot to take. Uses epsilon-greedy policy.
%     
%     Parameters
%     ------------
%     Q : double matrix
%         The current Q table
%     S : double array
%         The total list of states
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
%     Returns the list of states for RL.
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
%     Returns the list of actions for RL.
%     
%     Returns
%     --------------
%     actions : double array
%         The encoded actions, corresponding with safe places for robots to go to
    
    actions = 1:4;
end