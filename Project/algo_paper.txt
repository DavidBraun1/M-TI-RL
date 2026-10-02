clear; clc; close all;

%% 1. System- und Designparameter
A_sys = [2 1 1; 1 -1 0; 0 0 1];
B_sys = [0 0; 0 1; 1 0]; 
Q = eye(3);
R = eye(2);
umax = [3; 20];

%% 2. Definition der Basisfunktionen & Gradient
% 21 Terme für V(x)
sigma = @(x) [x(1)^2; x(2)^2; x(3)^2; x(1)*x(2); x(1)*x(3); x(2)*x(3); ...
              x(1)^4; x(2)^4; x(3)^4; x(1)^2*x(2)^2; x(1)^2*x(3)^2; ...
              x(2)^2*x(3)^2; x(1)^2*x(2)*x(3); x(1)*x(2)^2*x(3); ...
              x(1)*x(2)*x(3)^2; x(1)^3*x(2); x(1)^3*x(3); x(1)*x(2)^3; ...
              x(1)*x(3)^3; x(2)*x(3)^3; x(2)^2*x(3)];

% Analytischer Gradient (Größe: 21x3)
grad_sigma = @(x) [ ...
    2*x(1), 0, 0;                    % w1
    0, 2*x(2), 0;                    % w2
    0, 0, 2*x(3);                    % w3
    x(2), x(1), 0;                   % w4
    x(3), 0, x(1);                   % w5
    0, x(3), x(2);                   % w6
    4*x(1)^3, 0, 0;                  % w7
    0, 4*x(2)^3, 0;                  % w8
    0, 0, 4*x(3)^3;                  % w9
    2*x(1)*x(2)^2, 2*x(2)*x(1)^2, 0; % w10
    2*x(1)*x(3)^2, 0, 2*x(3)*x(1)^2; % w11
    0, 2*x(2)*x(3)^2, 2*x(3)*x(2)^2; % w12
    2*x(1)*x(2)*x(3), x(1)^2*x(3), x(1)^2*x(2); % w13
    x(2)^2*x(3), 2*x(1)*x(2)*x(3), x(1)*x(2)^2; % w14
    x(2)*x(3)^2, x(1)*x(3)^2, 2*x(1)*x(2)*x(3); % w15
    3*x(1)^2*x(2), x(1)^3, 0;        % w16
    3*x(1)^2*x(3), 0, x(1)^3;        % w17
    x(2)^3, 3*x(1)*x(2)^2, 0;        % w18
    x(3)^3, 0, 3*x(1)*x(3)^2;        % w19
    0, x(3)^3, 3*x(2)*x(3)^2;        % w20
    0, 2*x(2)*x(3), x(2)^2];         % w21

%% 3. Training (Policy Iteration)
lim = 1.2;
[X1, X2, X3] = meshgrid(-lim:0.1:lim, -lim:0.1:lim, -lim:0.1:lim);
pts = [X1(:), X2(:), X3(:)]';
w = zeros(21, 1);
u_old = @(x) -[8.31, 2.28, 4.66; 8.57, 2.27, 2.28] * x; 

fprintf('Berechne optimale Gewichte...');
for iter = 1:30
    A_LS = zeros(21, 21); B_LS = zeros(21, 1);
    for k = 1:size(pts,2)
        xk = pts(:,k);
        uk = u_old(xk);
        uk_sat = [max(min(uk(1), 2.99), -2.99); max(min(uk(2), 19.99), -19.99)];
        phi_dot = grad_sigma(xk) * (A_sys*xk + B_sys*uk_sat);
        Wu = 0;
        for i = 1:2
            Wu = Wu + 2*umax(i)*R(i,i)*uk_sat(i)*atanh(uk_sat(i)/umax(i)) + ...
                 umax(i)^2*R(i,i)*log(1 - uk_sat(i)^2/umax(i)^2);
        end
        A_LS = A_LS + (phi_dot * phi_dot');
        B_LS = B_LS + phi_dot * (xk'*Q*xk + Wu);
    end
    w = -(A_LS + 1e-4*eye(21)) \ B_LS;
    u_old = @(x) -umax .* tanh( (1./(2.*diag(R).*umax)) .* (B_sys' * grad_sigma(x)' * w) );
end
fprintf(' Fertig!\n');

%% 4. Simulation
t_span = [0 5];
x0 = [1; 1; -1];
[t, x_out] = ode45(@(t,x) A_sys*x + B_sys*(-umax .* tanh((1./(2*diag(R).*umax)) .* (B_sys' * grad_sigma(x)' * w))), t_span, x0);

% Stellgrößen für Plot berechnen
u_out = zeros(length(t), 2);
for i = 1:length(t)
    u_out(i,:) = (-umax .* tanh((1./(2*diag(R).*umax)) .* (B_sys' * grad_sigma(x_out(i,:)')' * w)))';
end

%% 5. Plots
figure;
subplot(2,1,1); plot(t, x_out, 'LineWidth', 2); grid on;
title('Zustandsverläufe (Paper)'); legend('x1','x2','x3');
xlabel('t [s]'); ylabel('x');
subplot(2,1,2); plot(t, u_out, 'LineWidth', 2); grid on;
title('Stellgrößen'); legend('u1','u2');
xlabel('t [s]'); ylabel('u');