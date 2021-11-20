% Name: proj2_ind.m
% Author: Jazel A. Suguitan
% Last Modified: Nov. 19, 2021

clc, clear, close all

% ALGORITHM 1: Independent Q-Learning

%================= SET PARAMETERS ===============

maxepisodes = 6; %Set number of episodes
num_nodes = 10;  %Number of nodes
n = 2; %number of dimensions
statelist   = BuildStateList(num_nodes);  % the list of states
actionlist  = BuildActionList(); % the list of actions
nstates     = length(statelist);    %number of states
nactions    = length(actionlist);   %number of actions
epsilon_learning = 0.0001;   % probability of a random action selection for e-greedy policy

nodes = 50.*rand(num_nodes,n)+50.*repmat([0 1],num_nodes,1);    %Randomly generate initial positions of nodes
p_nodes = zeros(num_nodes,n);   % Set initial velocties of MSN
delta_t = 0.003; %Time step ORIGINALLY 0.003
t = 0:delta_t:3.1;  %Simulation time

%NOTE: Q-table states are indexed at 1, but actual states start at 0
Q_initial = rand(nstates, nactions);%BuildQtableStorage(num_nodes, nstates, nactions); %load('Qcell_4actions2.mat'); %FOR 4 ACTIONS CASE
Q_update = Q_initial;

%SAVE DATA FOR EVALUATION
Connectivity_episodes = cell(1, maxepisodes);
Connectivity_episodes_learning = cell(1, maxepisodes);
R_ind_episodes = cell(1, maxepisodes);
R_all_episodes = cell(1, maxepisodes);
A_sum_cooQ_episodes = cell(1, maxepisodes);
Topo_eva_all_epi = cell(1, maxepisodes);
mean_Delta_Q_epi = cell(1, maxepisodes);
q_nodes_epi = cell(1, maxepisodes);

%3 lines, from proj 1
nodes_old = nodes; %KEEP privious positions of MSN
% q_nodes_all = cell(size(t,2),num_nodes);
% p_nodes_all = cell(size(t,2),num_nodes);

%================= SET SAFE PLACE POSITIONS ===============
safe_places = [];
for act = 1:nactions
    safe_places(act,:) = actionToPoint(act);
end

%================= START ITERATION ===============

for i=1:maxepisodes
    nodes = 50.*rand(num_nodes,n)+50.*repmat([0 1],num_nodes,1);    %Generate new node positions each episode
    %Training
%    [Q_update, Connectivity, Connectivity_learning, R_all, A_sum_cooQ, mean_Delta_Q]  = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t, safe_places);
    [Q_update, Connectivity, R_nodes, R_sum_all, A_sum_cooQ, mean_Delta_Q, q_nodes_all]  = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t, safe_places);
%    Q_update = Q_Learning(Q_update, statelist, actionlist, nstates, nactions, num_nodes, n, nodes, epsilon_learning, delta_t, t, safe_places);
    %Save data
    Connectivity_episodes{i} = Connectivity;
    %Connectivity_episodes_learning{i} = Connectivity_learning; %CHECK - is there a point for this??
    R_ind_episodes{i} = R_nodes;
    R_all_episodes{i} = R_sum_all;
    A_sum_cooQ_episodes{i} = A_sum_cooQ;
    mean_Delta_Q_epi{i} = mean_Delta_Q; 
    q_nodes_epi{i} = q_nodes_all;
    disp(['Episode: ',int2str(i),]) 
    %decrease probability of a random action selection (for e-greedy selection)
    epsilon_learning = epsilon_learning * 0.99;
end


%======================== PLOTS ===========================

%Requirement 2 - Node trajectories

%Plot node trajectories in first episode
q_nodes_mat = cell2mat(q_nodes_epi{1});
figure(2),plot(q_nodes_mat(:,1),q_nodes_mat(:,2),'k.')
hold on
plot(q_nodes_epi{1}{length(t)}(:,1),q_nodes_epi{1}{length(t)}(:,2), 'm>','LineWidth',.2,'MarkerEdgeColor','m','MarkerFaceColor','m','MarkerSize',5)
title('Node trajectory in first episode')

%Plot node trajectories in second episode
q_nodes_mat = cell2mat(q_nodes_epi{2});
figure(3),plot(q_nodes_mat(:,1),q_nodes_mat(:,2),'k.')
hold on
plot(q_nodes_epi{2}{length(t)}(:,1),q_nodes_epi{2}{length(t)}(:,2), 'm>','LineWidth',.2,'MarkerEdgeColor','m','MarkerFaceColor','m','MarkerSize',5)
title('Node trajectory in second episode')

%Plot node trajectories in last episode
q_nodes_mat = cell2mat(q_nodes_epi{maxepisodes});
figure(4),plot(q_nodes_mat(:,1),q_nodes_mat(:,2),'k.')
hold on
plot(q_nodes_epi{maxepisodes}{length(t)}(:,1),q_nodes_epi{maxepisodes}{length(t)}(:,2), 'm>','LineWidth',.2,'MarkerEdgeColor','m','MarkerFaceColor','m','MarkerSize',5)
title('Node trajectory in last episode')


%Requirement 3 - Individual Rewards of Nodes
R_each_node = zeros(length(t)*maxepisodes, num_nodes);
for ep = 1:maxepisodes
    temp = R_ind_episodes{ep};
   for i = 1:length(t)
      for j = 1:num_nodes
          R_each_node((ep-1)*length(t) + i, j) = temp(i, j);
      end
   end
end
figure(5), plot(R_each_node);
title('Individual Reward over learning episodes')
grid on

%Requirement 4 - Total (Sum) of Rewards
R_all_epi_mat = cell2mat(R_all_episodes);
[R_all_diff0, index_R]= find(R_all_epi_mat>0);
figure(6), plot(R_all_epi_mat(index_R));
title('Total reward over learning episodes')
grid on

%Requirement 5 - Total Action Selection of Nodes
A_cooQ_matrix = cell2mat(A_sum_cooQ_episodes);
[A_diff0, index_A_cooQ] = find(A_cooQ_matrix>0);
figure(7), plot(A_cooQ_matrix(index_A_cooQ))
title('Action Selection over learning episodes')
grid on

%Requirement 6 - Average of Delta Q TODO


% %Plot connectivity and action selection evaluations, respectively (LAST EPISODE)
% figure(9),plot(Connectivity)
% title('Network Connectivity over the last learning episode')
% grid on
% %Plot connectivity and action selection evaluations, respectively (WHOLE EPISODES)
% Con_epi_mat = cell2mat(Connectivity_episodes); 
% figure(10), plot(Con_epi_mat)
% title('Network Connectivity over learning episode')
% grid on

% %Plot reward in last episode
% [R_all_diff0, index_R_all]= find(R_all>0);
% figure(11), plot(R_all(index_R_all))
% title('Total Reward in the last episode')
% grid on


%================= FUNCTIONS ===============

%EDIT - orig. fxn signature below
% function [Q_update, Connectivity, Connectivity_learning, ...
%     R_all, A_sum_cooQ, mean_Delta_Q, q_nodes_all] = Q_Learning(Q_update, ...
%     statelist, actionlist, nstates, nactions, num_nodes, ...
%     n, nodes, epsilon_learning, delta_t, t, safe_places)

function [Q_update, Connectivity, R_nodes, ...
    R_sum_all, A_sum_cooQ, mean_Delta_Q, q_nodes_all] = Q_Learning(Q_update, ...
    statelist, actionlist, nstates, nactions, num_nodes, ...
    n, nodes, epsilon_learning, delta_t, t, safe_places)
% function Q_update = Q_Learning(Q_update, ...
%     statelist, actionlist, nstates, nactions, num_nodes, ...
%     n, nodes, epsilon_learning, delta_t, t, safe_places)
%     The Q-Learning reinforcement learning algorithm.
%     
%     Parameters
%     -------------
%     Q_update : double matrix
%         The current Q-table
%     statelist : double array
%         The encoded states
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
%     safe_places : double matrix
%         Positions of safe places
%     q_nodes_all : double matrix
%         Positions of nodes over the episode
%         
%     Returns
%     --------------
%     Q_update : double matrix
%         The updated Q-table
%     Connectivity : double array
%         Connectivity values over this episode
%     Connectivity_learning : ????
%         ??????
%     R_nodes : double matrix
%         Individual reward values for all nodes over this episode
%     R_sum_all : double array
%         Sum of reward values for all nodes over this episode
%     A_sum_cooQ : double array
%         Actions taken for all nodes over this episode
%     mean_Delta_Q : double matrix
%         Changes in Q-table values for all nodes

    %================= SET PARAMETERS ===============
    
    d = 25.5; %Set desired distance among sensor nodes
    r = 30; %Set active range of nodes
    epsilon = 0.1;  %Set a constant for sigma norm
    alpha = 0.95;
    gamma = 0.05;
    
    p_nodes = zeros(num_nodes,n);   % Set initial velocties of MSN
    nodes_old = nodes; %KEEP privious positions of MSN
    Connectivity = 1:length(t); %Save connectivity of MSN
    R_sum_all = 1:length(t);    %Save sum of reward values for nodes
    R_nodes = zeros(length(t), 10);%cell(length(t),1);    %Sum individual reward values for nodes
    A_sum_cooQ = 1:length(t);   %Save action values for nodes
    q_nodes_all = cell(length(t),1); %cell(1, length(t));

%     p_nodes_all = cell(size(t,2),num_nodes);
    mean_Delta_Q = zeros(size(t,2),n);%Save positions of COM (Center of Mass)
    
    s_t = [];   %Keep track of states of nodes at start of iteration
    a_next = []; %Keep track of actions selected by nodes

    
    %================= INITIALIZE STATES ===============

    [Nei_agent, A] = findNeighbors(nodes, r);   %Determine neighbors for each node
    for i = 1:num_nodes
        s_t(i) = DiscretizeState(nodes, i, Nei_agent{i});%length(Nei_agent{i}) + 1;  %Node's initial state = number of neighbors, +1 because states are indexed at 1
%         a_next(i) = select_action(Q_update, s_t(i), epsilon_learning, nactions); %Node selects an action
    end
    
    if (1/(num_nodes))*rank(A) == 1
        %fully connected MSN
        s_t(:) = 2;
    end
    
    
    %================= START ITERATION ===============
    
    for iteration = 1:length(t)
        plot(safe_places(:,1),safe_places(:,2),'ro','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','r', 'MarkerSize',4.2)
        hold on
        
        %Initialize sum values for iteration
        A_sum_cooQ(iteration) = 0;
        R_sum_all(iteration) = 0;
        
        %Choose actions for each node
        for i = 1:num_nodes
            a_next(i) = select_action(Q_update, s_t(i), epsilon_learning, nactions); %Node selects an action
            A_sum_cooQ(iteration) = A_sum_cooQ(iteration) + a_next(i);  %Save action
        end
        
        %Each node takes action a_next(i)
        [Nei_agent, A] = findNeighbors(nodes, r);
        [Ui] = inputcontrol_Algorithm2(nodes, Nei_agent, num_nodes, epsilon, r, d, p_nodes, n, a_next); %last param used to be qt1
        p_nodes = (nodes - nodes_old)/delta_t; %COMPUTE velocities of sensor nodes
%         p_nodes_all{iteration} = p_nodes; %SAVE VELOCITY OF ALL NODES
        nodes_old = nodes;
        nodes = nodes_old + p_nodes*delta_t  + Ui*delta_t*delta_t /2;
        mean_Delta_Q(iteration,:) = mean(nodes); %Compute position of COM of MSN
        plot(mean_Delta_Q(:,1),mean_Delta_Q(:,2),'ro','LineWidth',2,'MarkerEdgeColor','k', 'MarkerFaceColor','k','MarkerSize',4.2)
        hold on
        q_nodes_all{iteration} = nodes; %q_nodes_all{iteration}(:,:) = nodes;
        Connectivity(iteration)= (1/(num_nodes))*rank(A);
        
        %Observe R and S'
        s_next = [];    %Keep track of new states of nodes
        [Nei_agent, A] = findNeighbors(nodes, r); %Determine neighbors for each node
        for i = 1:num_nodes
            s_next(i) = DiscretizeState(nodes, i, Nei_agent{i});%length(Nei_agent{i}) + 1;  %Node's initial state = number of neighbors, +1 because states are indexed at 1
            connect = (1/(num_nodes))*rank(A);
            if (1/(num_nodes))*rank(A) == 1
                %fully connected MSN
                s_next(i) = 2;
            end
            
            %assign reward
            if length(Nei_agent{i}) < 6
                reward = length(Nei_agent{i});
            else
                reward = 6;
            end
            R_nodes(iteration, i) = reward; %Save reward value
            R_sum_all(iteration) = R_sum_all(iteration) + reward;   %Add to sum reward value
            newMax = max(Q_update(s_next(i),:));  %get max reward of new state from Q-table
            Q_update(s_t(i),a_next(i)) = Q_update(s_t(i),a_next(i)) + alpha * (reward + gamma*newMax - Q_update(s_t(i),a_next(i))); %Update node's q table
        end
        
        %Set S = S'
        for i = 1:num_nodes
            s_t(i) = s_next(i);
        end
        
        %================= PLOT and LINK SENSOR TOGETHER ===============
        plot(nodes(:,1),nodes(:,2), '.')
        hold on
        plot(nodes(:,1),nodes(:,2), 'k>','LineWidth',.2,'MarkerEdgeColor','k','MarkerFaceColor','k','MarkerSize',5)
        hold off
        for node_i = 1:num_nodes
            tmp=nodes(Nei_agent{node_i},:);
            for j = 1:size(nodes(Nei_agent{node_i},1))
                line([nodes(node_i,1),tmp(j,1)],[nodes(node_i,2),tmp(j,2)]) 
            end
        end
        drawnow;
        hold off
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
    c1_alpha = 30; %ORIGINALLY 30, 67, 10
    c2_alpha = 2*sqrt(c1_alpha);
    c1_mt = 1.1;    % ORIGINALLY 1.1, 75, 500
    Ui = zeros(num_nodes, dimensions);  % initialize Ui matrix to all 0's
    gradient = 0.;  % Initialize gradient part of Ui equation
    consensus = 0.; % Initialize consensus part of Ui equation
    feedback = 0.;  % Initialize navigational feedback of Ui equation
    
    % Sum gradient and consensus values for each node i
    for i = 1:num_nodes
        q_mt = actionToPoint(a_nexts(i)); %Get target for node based off its a_next value
%         q_mt = actionToPoint(3); %Make all nodes go to point 3
        
        % EDIT - commented section below for initial simplified Ui
        for j = 1:length(Nei_agent{i})
            % i refers to node i
            % j refers to the jth neighbor of node i
            phi_alpha_in = sigmaNorm(nodes(Nei_agent{i}(j),:) - nodes(i,:), epsilon);
            gradient = gradient + phi_alpha(phi_alpha_in, r, d, epsilon) * nij(nodes(i,:), nodes(Nei_agent{i}(j),:), epsilon);
            consensus = consensus + aij(nodes(i,:), nodes(Nei_agent{i}(j),:), epsilon, r) * (p_nodes(Nei_agent{i}(j),:) - p_nodes(i,:));
        end
        feedback = nodes(i,:) - q_mt;
        
        Ui(i,:) = ((c1_alpha * gradient) + (c2_alpha * consensus) - (c1_mt * feedback));   % Set Ui for node i using gradient, consensus, and feedback
        %Ui(i,:) = -(c1_mt * feedback);  %EDIT - simplified Ui
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
%     Builds the list of states for RL (assuming 2 dimensions).
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
     
    states = zeros(n*2 +2, 2);
    for row = 1:n*2 +2
       for col = 1:2
           if col == 1
               if mod(row, 2) == 0
                   %even row
                   states(row,col) = row/2 - 1;
               else
                   %odd row
                   states(row,col) = floor(row/2);
               end
           else
              % second column
              states(row,col) = mod(row-1, 2);
           end
       end
    end
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

function result = sigmaNorm(z, epsilon)
%     Returns the sigma norm of a given value/vector z and an epsilon contant value.
%     
%     Parameters
%     -------------
%     z : double
%           Vector of which to take the sigma norm
%     epsilon : double
%           A constant used to calculate the sigma norm
%     
%     Returns
%     -----------
%     result : double
%           The sigma norm value of vector z

    result = (1/epsilon) * (sqrt(1 + epsilon*(norm(z))^2)-1);
end

function result = phi_alpha(z, r, d, epsilon)
%     The action function used to construct a smooth pairwise potential
%     with finite cut-off in the gradient-based term of the Alg.1 Ui.
%     
%     Parameters
%     -------------
%     z : double
%           Sigma norm value of two nodes
%     r : double
%           Interaction range of nodes in MSN
%     d : double
%           Desired distance of nodes in MSN
%     espilon : double
%           A constant for the sigma norm
%     
%     Returns
%     -----------
%     result : double
%           Value to be used in Ui gradient-based term

    r_alpha = sigmaNorm(r, epsilon);
    d_alpha = sigmaNorm(d, epsilon);    %CHECK - is this what d alpha is?
    result = bump(z/r_alpha) * phi(z-d_alpha);
end

function result = phi(z)
%     An uneven sigmoidal function, used in the phi_alpha function.
%     
%     Parameters
%     -------------
%     z : double
%     
%     Returns
%     -----------
%     result : double

    %Set constants
    a = 5;
    b = 5;
    c = abs(a-b) / sqrt(4*a*b);
    
    sigmaZ = sigma1(z+c);
    result = 0.5*((a+b)*sigmaZ + (a-b));
end

function result = bump(z)
%     A scalar function varying between 0 and 1. Used for construction of smooth potential functions with finite cut-offs and smooth adj. matrices.
%     
%     Parameters
%     -------------
%     z : double
%           The input to be smoothened
%     
%     Returns
%     -----------
%     result : double
%           The 0 or 1 value
    
    h = 0.2;    % Set constant h
    
    if z >= 0 && z < h
        result = 1;
    elseif z >= h && z <= 1
        result = 0.5 * (1 + cos(pi*(z-h)/(1-h)));
    else
        result = 0;
    end
end

function result = nij(i, j, epsilon)
%     Function for obtaining the vector along the line connecting two nodes.
%     Used in calculating the gradient-based term in Algorithm 1 Ui.
%     
%     Parameters
%     -------------
%     i : double array (1x2)
%           Position of node i
%     j : double array (1x2)
%           Position of node j
%     epsilon : double
%           A constant
%     
%     Returns
%     -----------
%     result : double array (1x2)
%           The vector along the line connecting node i and node j

    result = sigmaE(j-i, epsilon);
end

function result = sigmaE(z, epsilon)
%     Function to be used in nij function.
%     
%     Parameters
%     -------------
%     z : double (1x2)
%           A vector
%     epsilon : double
%           A constant
%     
%     Returns
%     -----------
%     result : double (1x2)
%           The resulting vector
    
    result = z / (1 + epsilon * sigmaNorm(z, epsilon));
end

function result = aij(i, j, epsilon, r)
%     Returns the spatial adjacency matrix given the positions of two nodes, i and j.
%     
%     Parameters
%     -------------
%     i : double array (1x2)
%           Position of node i
%     j : double array (1x2)
%           Position of node j
%     epsilon : double
%           Constant for sigma norm
%     r : double
%           Interaction range for nodes in MSN
%     
%     Returns
%     -----------
%     result : double array (1x2)
%           The spatial adjacency matrix
    
    result = zeros(size(i));    % result is a 1x2 matrix - CHECK
    
    if ~isequal(i,j)
        r_alpha = sigmaNorm(r, epsilon);
        input_to_bump = sigmaNorm(j-i, epsilon) / r_alpha;
        result = bump(input_to_bump);
    end
end

function result = sigma1(z)
%     A function to be used in the phi function.
%     
%     Parameters
%     -------------
%     z : double
%     
%     Returns
%     -----------
%     result : double

    result = z / sqrt(1+z^2);
end

function state = DiscretizeState(nodes, currNodeInd, neighborList)
%     Returns a single state for a node from that node's list of neighbors.
%     
%     Parameters
%     ------------
%     nodes : double matrix
%         Positions of nodes
%     currNodeInd : double
%         Index of node whose state is being determined
%     neighborList : double array
%         Indices of neighbor nodes for a single node
%         
%     Returns
%     ------------
%     state : double
%         Encoded state value

    if isempty(neighborList)
        state = 1;
        return;
    end
      
    minDist = realmax;  %initialize initial distance to largest floating-point value
    minInd = 1;
    for i = 1:length(neighborList)
        dist = norm(nodes(currNodeInd, :) - nodes(neighborList(i),:));
        if dist < minDist
            minDist = dist;
            minInd = neighborList(i);
        end
    end
    
    state = (minInd + 1) * 2;
end