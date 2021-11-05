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
nstates     = size(statelist,1);
nactions    = size(actionlist,1); 
epsilon_learning     = 0.0001;   % probability of a random action selection for e-greedy policy
delta_t = 0.003; 
t = 0:delta_t:3.1;
Q_initial = load('Qcell_4actions2.mat'); %FOR 4 ACTIONS CASE -  EDIT!!!!!
Q_update = Q_initial.Q;
%SAVE DATA FOR EVALUATION
Connectivity_episodes = cell(1, maxepisodes );
Connectivity_episodes_learning = cell(1, maxepisodes );
R_all_episodes = cell(1, maxepisodes );
A_sum_cooQ_episodes = cell(1, maxepisodes );
Topo_eva_all_epi = cell(1, maxepisodes );
mean_Delta_Q_epi = cell(1, maxepisodes );

% %================= START ITERATION ===============
% 
% for i=1:maxepisodes
%     nodes = 90.*rand(num_nodes,n)+90.*repmat([0 1],num_nodes,1);
%     %Training
%     [Q_update, Connectivity, Connectivity_learning, R_all, A_sum_cooQ, mean_Delta_Q]  = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning );
%     %Save data
%     Connectivity_episodes{i} = Connectivity;
%     Connectivity_episodes_learning{i} = Connectivity_learning;
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
