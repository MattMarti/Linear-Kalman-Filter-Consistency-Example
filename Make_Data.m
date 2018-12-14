%% Make Data
% 
% This script generates the true state time history, measurements, and
% initial state guess for the example filter in "Filter_Q_Example.m".
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
% @author: Matt Marti
% @date: 2018-12-14

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
F = [ 1,  dt,  dt^2/2, 0,  0,   0,      ; ...
      0,  1,   dt,     0,  0,   0,      ; ...
      0,  0,   1,      0,  0,   0,      ; ...
      0,  0,   0,      1,  dt,  dt^2/2, ; ...
      0,  0,   0,      0,  1,   dt,     ; ...
      0,  0,   0,      0,  0,   1,      ];
  
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


%% Process Noise Stuff
% These need to be chosen correctly to get better estimates. Try different
% values to see how they improve or worsen the Chi-Squared Distribution
% tests.
% 
% The way things are set up here, process noise affects velocity and
% acceleration.

% Gamma - Process Noise Gain
% This is usually just 1s and 0s, and essentially just let's you add the
% process noise v(k) to the state. Sometimes it has other values though,
% especially when v(k) is a scalar. If this is the case, then v(k) is
% scaled by this matrix.
Gamma = [ 0, 0, 0, 0 ; ...
          1, 0, 0, 0 ; ...
          0, 1, 0, 0 ; ...
          0, 0, 0, 0 ; ...
          0, 0, 1, 0 ; ...
          0, 0, 0, 1 ];

% Q - Process Noise Covariance
% This is the real important part. These values are assumed, and there's no
% solid theoretical way to get them. Hence, a machine learning technique
% could be used to guess them to get the filter to work right. Not knowning
% this matrix also extends to it's dimension. For example, here there is no
% prcoess noise for position, but whose to say the car isn't slipping? Is
% the noise correlated (elements off the diagonal)?
Q = [ 0.01,  0,  0,     0  ; ...
      0,     1,  0,     0  ; ...
      0,     0,  0.01,  0  ; ...
      0,     0,  0,     1  ];


%% Run Filter on System

N = 200;
nx = 6;
nv = 4;
nz = 4;

% Initial state
x0 = [ 2; .3; -0.005; -1; -.2; .002 ];

% Generate True Measurment Noise
Sr = chol(R)';
whist = Sr*randn(nz,N);

% Generate True Process Noise
Sq = chol(Q)';
v0 = Sq*randn(nv,1);
vhist = Sq*randn(nv,N);
figure(4)
plot(vhist');
title('Process Noise');

% Preallocate data arrays
xhist = zeros(nx,N);
zhist = zeros(nz,N);

% Generate data
xk = x0;
vk = v0;
for kp1 = 1:N
    
    % Current state data
    wkp1 = whist(:,kp1);
    
    % Compute process
    xkp1 = F*xk + Gamma*vk;
    zkp1 = H*xkp1 + wkp1;
    
    % Save data
    xhist(:,kp1) = xkp1;
    zhist(:,kp1) = zkp1;
    
    % Iterate state data
    xk = xkp1;
    vk = vhist(:,kp1);
end
% xhist = xhist([1:3,5:7],:);


%% Save Output

% Ground Track Plot
figure(1)
hold off
plot(xhist(1,:), xhist(4,:),'k.-', 'linewidth', 1.25, 'markersize', 5);
hold on
plot(zhist(1,:), zhist(3,:),'ro', 'linewidth', 1.25, 'markersize', 3);
title('Object Ground Track');
xlabel('x1');
ylabel('y1');
grid on, grid minor

% Velocity Hist
figure(3)
hold off
plot(xhist(2,:), xhist(5,:),'k.-', 'linewidth', 1.25, 'markersize', 5);
hold on
plot(zhist(2,:), zhist(4,:),'ro', 'linewidth', 1.25, 'markersize', 3);
title('Object Velocity Track');
xlabel('x1');
ylabel('y1');
grid on, grid minor

% Accerlation Hist
figure(5)
hold off
plot(xhist([3,6],:)');
title('Acceleration Time History');
grid on, grid minor

save zhist zhist
save xhist xhist