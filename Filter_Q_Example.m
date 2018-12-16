%% Filter Process Noise Covariance Example
% 
% This script serves as a brief tutorial on tuning the Process Noise 
% Covariance for a Kalman Filter. Recall that the Kalman Filter is, for 
% linear processes, the optimal Maximum Likelihood and Minimum Mean 
% Squared Error estimator for the system described by:
% 
%   x(k+1) = F*x(k) + Gamma*v(k)        - State Transition
%   z(k+1) = H*x(k+1) + w(k+1)          - Measurement
% 
% In this example, a vehicle will be driving on a road in two dimensions.
% The vehicle state will be modeled as the x1 and x2 position, velocity,
% and acceleration components.
% 
% The point of this script is to show what it means that the Process Noise
% Covariance (Q) needs to be tuned. It's not like the Measurement Noise
% Covariance (R) that can be measured by taking the standard deviation of
% sensor errors. There are rules of thumb to guess Q, but beyond that
% there's no theoretical way to do it. I think that it would probably be a
% good problem for machine learning.
% 
% DEPENDENCIES
% zhist.mat
% xhist.mat
% 
% @author: Matt Marti
% @date: 2018-12-16

clear, clc


%% Governing System Parameters
% In this example, we'll assume we're measuring a car with a GPS. So the
% system will essentially be x and y positions, velocities, accelerations.
% The state will take the form:
% 
%   x(k) = [ x1; v1; a1; x2; v2; a2 ]
% 
% Measurements will take the form:
% 
%   z(k) = [ x1; v1; x2; v2 ]
% 
% Let the standard deviation of GPS be 5 meters for position, and 0.001 m/s
% for velocity. This is essentially a loosely coupled GPS Filter, meaning
% we are using position and velocity measurements instead of pseudorange
% and Doppler.

% Measurement Delta Time
dt = .2;

% F - State Transition Matrix
% x = [ x1; v1; a1; x2; v2; a2 ]
F = [ 1,  dt,  dt^2/2, 0,  0,   0      ; ...
      0,  1,   dt,     0,  0,   0      ; ...
      0,  0,   1,      0,  0,   0      ; ...
      0,  0,   0,      1,  dt,  dt^2/2 ; ...
      0,  0,   0,      0,  1,   dt     ; ...
      0,  0,   0,      0,  0,   1      ];

% H - Measurement Model
% z = [ x1; v1; x2; v2 ]
H = [ 1, 0, 0, 0, 0, 0 ;...
      0, 1, 0, 0, 0, 0 ;...
      0, 0, 0, 1, 0, 0 ;...
      0, 0, 0, 0, 1, 0 ];

% R - Measurement Noise Covariance
R = [ 5^2,  0,        0,    0       ; ...
      0,    0.001^2,  0,    0       ; ...
      0,    0,        5^2,  0       ; ...
      0,    0,        0,    0.001^2 ];


%% Process Noise
% These need to be chosen correctly to get better estimates. Try different
% values to see how they improve or worsen the Chi-Squared Distribution
% tests.
% 
% The way things are set up here, process noise affects velocity and
% acceleration.

% Gamma - Process Noise Gain
% This is the partial derivative of the state transition function with
% respect to the process noise terms. Essentially, it lets you add the
% process noise to the state. The purpose of this matrix is more obvious
% for non-linear functions and the Extended Kalman Filter.
Gamma = [ 0,  0,  0,  0 ; ...
          1,  0,  0,  0 ; ...
          0,  1,  0,  0 ; ...
          0,  0,  0,  0 ; ...
          0,  0,  1,  0 ; ...
          0,  0,  0,  1 ];

% Q - Process Noise Covariance
% This is the real important part. These values are assumed, and there's no
% solid theoretical way to get them. Hence, a machine learning technique
% could be used to guess them to get the filter to work right. Not knowning
% this matrix also extends to it's dimension. For example, here there is no
% prcoess noise for position, but whose to say the car isn't slipping? Is
% the noise correlated (elements off the diagonal)?
Q = [ 0.09,  0,  0,     0  ; ...
      0,     4,  0,     0  ; ...
      0,     0,  0.09,  0  ; ...
      0,     0,  0,     4  ];

% As an alternative, you can try to switch between the Q values below
% Q = 0.0001*Q;
% Q = 0.001*Q;
% Q = 0.01*Q;
% Q = 0.1*Q;
% Q = 10*Q;
% Q = 100*Q;
% Q = 1000*Q;
% Q = 10000*Q;
% Q = 0.001*eye(4);
% Q = 0.01*eye(4);
% Q = 0.1*eye(4);
% Q = 1*eye(4);
% Q = 10*eye(4);
% Q = 100*eye(4);
% Q = 1000*eye(4);


%% Run Filter on System

% Load measurement time history
load('zhist.mat');
N = size(zhist,2);

% Initial state estimate - assume we are starting from the first GPS
% measurement
xhat_0 = [ 0; 0; 0; 0; 0; 0 ];

% Initial Covariance estimate - This can be set pretty big and it will
% converge fast for a linear system. Don't want it to be too small if you
% think you're not accurate
P_0 = 10*eye(6);

% Preallocate data arrays
xhathist = zeros(6,N);
Phist = zeros(6,6,N);
epsilonhist = zeros(1,N);

% Initialize loop variables
xhat_k = xhat_0;
P_k = P_0;

% Run Filter
for kp1 = 1:N % Index by k+1
    
    % Dynamic Propagation - Predict the next state
    xbar_kp1 = F*xhat_k;
    Pbar_kp1 = F*P_k*(F') + Gamma*Q*(Gamma');
    
    % Obtain Measurement Data
    zkp1 = zhist(:,kp1);
    
    % Filter Innovation
    nu_kp1 = zkp1 - H*xbar_kp1;
    
    % Compute Kalman Gain
    S_kp1 = H*Pbar_kp1*(H') + R;
    invS_kp1 = inv(S_kp1);
    W_kp1 = Pbar_kp1*(H')*invS_kp1; %#ok
    
    % Measurement Update - Correct the prediction based on measurements
    xhat_kp1 = xbar_kp1 + W_kp1*nu_kp1;
    P_kp1 = Pbar_kp1 - W_kp1*S_kp1*(W_kp1');
    
    % Innovation Statistic
    epsilon_kp1 = (nu_kp1')*invS_kp1*nu_kp1; %#ok
    
    % Save data
    xhathist(:,kp1) = xhat_kp1;
    Phist(:,:,kp1) = P_kp1;
    epsilonhist(kp1) = epsilon_kp1;
    
    % Iterate
    xhat_k = xhat_kp1;
    P_k = P_kp1;
end


%% Chi-Squared Distribution Test for Filter Consistency
% This is what's known as a consistency test. It makes sure that the Filter
% Innovation is sampled from a Chi-Squared distribution of degree equal to 
% the number of elements of the measurement vector z. Essentially, this is
% what you use to tell that the Filter is working correctly (In the absence
% of truth data).

% "1% of points may lie outside these bounds"
alpha = 0.01;

% Chi-Squared Distribution Bounds for Innovation Statistic
% These are displayed as red lines on the Innovation Statistic Mean Time
% History. A certain percentage of points must lie within these bounds.
nz = 4; % Number of Measurements
r1nu = chi2inv(alpha/2, nz);
r2nu = chi2inv(1-alpha/2, nz);

% Chi-Squared Distribution Bounds for Innovation Statistic Mean
% These are displayed as magenta lines on the Consistency Test Time
% Hisotry. The mean value of the Innovation Statistic must lie within these
% bounds.
r1nu_mean = chi2inv(alpha/2, N*nz)/N;
r2nu_mean = chi2inv(1-alpha/2, N*nz)/N;

% Chi-squared distribution test
passcount = zeros(N,1);
for k = 1:N
    passcount(k) = (r1nu <= epsilonhist(k)) && (epsilonhist(k) <= r2nu);
end
passrate = 100*sum(passcount)/length(passcount);
pass = passrate >= 100*(1-alpha);

% Filter consistency can also be measured by running Monte-Carlo
% simulations. However, for this example, we are assuming we don't have the
% true state time history.

% Display whether filter passes consistency test
if pass
    fprintf('Filter passed consistency test\n');
else
    fprintf('Filter failed consistency test\n');
end
fprintf('Filter pass rate: %.2f\n', passrate);


%% Plot State Estimates

% Load Truth Data
load xhist.mat

% Ground Track Plot
figure(1)
hold off
plot(xhist(1,:), xhist(4,:),'k.-', 'linewidth', 1.25, 'markersize', 10);
hold on
plot(zhist(1,:), zhist(3,:),'ro', 'markersize', 5);
plot(xhathist(1,:), xhathist(4,:),'b.-', 'linewidth', 1.25, 'markersize', 10);
title('Object Ground Track');
xlabel('x1');
ylabel('y1');
legend({'True', 'GPS','Kalman'});
grid on, grid minor

% Innovation Statistic Plot
figure(2);
hold off;
semilogy(epsilonhist', 'linewidth', 1.5);
hold on;
semilogy(r1nu*ones(size(epsilonhist)), 'r--', 'linewidth', 1.75);
semilogy(r2nu*ones(size(epsilonhist)), 'r--', 'linewidth', 1.75);
semilogy(r1nu_mean*ones(size(epsilonhist)), 'm--', 'linewidth', 1.75);
semilogy(r2nu_mean*ones(size(epsilonhist)), 'm--', 'linewidth', 1.75);
semilogy(mean(epsilonhist)*ones(size(epsilonhist)), 'b-.', ...
    'linewidth', 1);
semilogy(nz*ones(size(epsilonhist)), 'k-.', 'linewidth', 1);
hold off;
title('Innovation Statistic Consistency Test Time History');
ylabel('Innovation Statistic');
xlabel('Index of Innovation Statistic k');
grid on, grid minor;

% Position Error
figure(3);
hold off
plot((xhathist([1,4],:) - xhist([1,4],:))');
title('Position Error Time History');
grid on, grid minor

% Velocity Error
figure(4);
hold off
plot((xhathist([2,5],:) - xhist([2,5],:))');
title('Velocity Error Time History');
grid on, grid minor

% Acceleration Error
figure(5);
hold off
plot((xhathist([3,6],:) - xhist([3,6],:))');
title('Acceleration Error Time History');
grid on, grid minor

% Acceleration Time History
figure(6);
hold off
plot(xhist([3,6],:)', 'b');
hold on
plot(xhathist([3,6],:)','r');
title('Acceleration Time History');
legend({'True', '', 'Kalman', ''});
grid on, grid minor