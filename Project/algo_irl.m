clear; clc; close all;

%% 1. System-Parameter
A_sys = [2 1 1; 1 -1 0; 0 0 1];
B_sys = [0 0; 0 1; 1 0]; 
Q = eye(3); R = eye(2); umax = [3; 20];
lim = 1.2;

%% 2. Basisfunktionen
sigma = @(x) [x(1)^2; x(2)^2; x(3)^2; x(1)*x(2); x(1)*x(3); x(2)*x(3); ...
              x(1)^4; x(2)^4; x(3)^4; x(1)^2*x(2)^2; x(1)^2*x(3)^2; ...
              x(2)^2*x(3)^2; x(1)^2*x(2)*x(3); x(1)*x(2)^2*x(3); ...
              x(1)*x(2)*x(3)^2; x(1)^3*x(2); x(1)^3*x(3); x(1)*x(2)^3; ...
              x(1)*x(3)^3; x(2)*x(3)^3; x(2)^2*x(3)];

grad_sigma = @(x) [2*x(1), 0, 0; 0, 2*x(2), 0; 0, 0, 2*x(3); x(2), x(1), 0; ...
    x(3), 0, x(1); 0, x(3), x(2); 4*x(1)^3, 0, 0; 0, 4*x(2)^3, 0; 0, 0, 4*x(3)^3; ...
    2*x(1)*x(2)^2, 2*x(2)*x(1)^2, 0; 2*x(1)*x(3)^2, 0, 2*x(3)*x(1)^2; ...
    0, 2*x(2)*x(3)^2, 2*x(3)*x(2)^2; 2*x(1)*x(2)*x(3), x(1)^2*x(3), x(1)^2*x(2); ...
    x(2)^2*x(3), 2*x(1)*x(2)*x(3), x(1)*x(2)^2; x(2)*x(3)^2, x(1)*x(3)^2, 2*x(1)*x(2)*x(3); ...
    3*x(1)^2*x(2), x(1)^3, 0; 3*x(1)^2*x(3), 0, x(1)^3; x(2)^3, 3*x(1)*x(2)^2, 0; ...
    x(3)^3, 0, 3*x(1)*x(3)^2; 0, x(3)^3, 3*x(2)*x(3)^2; 0, 2*x(2)*x(3), x(2)^2];

%% 3. Verbessertes Training: Integral RL
T_int = 0.1; dt = 0.01; n_starts = 500;
w = zeros(21, 1);
u_old = @(x) -[8.31, 2.28, 4.66; 8.57, 2.27, 2.28] * x; 

fprintf('Berechne IRL Gewichte...\n');
for iter = 1:20
    A_IRL = zeros(21, 21); B_IRL = zeros(21, 1);
    for k = 1:n_starts
        xk = (2*rand(3,1) - 1) * lim; 
        uk = u_old(xk);
        uk_sat = [max(min(uk(1), 2.99), -2.99); max(min(uk(2), 19.99), -19.99)];
        
       
        cost_int = 0; x_temp = xk;
        for s = 1:(T_int/dt)
            Wu = 0;
            for i = 1:2
                Wu = Wu + 2*umax(i)*R(i,i)*uk_sat(i)*atanh(uk_sat(i)/umax(i)) + ...
                     umax(i)^2*R(i,i)*log(1 - (uk_sat(i)/umax(i))^2);
            end
            cost_int = cost_int + (x_temp'*Q*x_temp + Wu) * dt;
            dx = A_sys * x_temp + B_sys * uk_sat;
            x_temp = x_temp + dx * dt;
        end
        
        % Bellman-Differenz
        delta_phi = (sigma(x_temp) - sigma(xk));
        A_IRL = A_IRL + (delta_phi * delta_phi');
        B_IRL = B_IRL + delta_phi * (-cost_int);
    end
    w = (A_IRL + 1e-4*eye(21)) \ B_IRL;
    u_old = @(x) -umax .* tanh( (1./(2.*diag(R).*umax)) .* (B_sys' * grad_sigma(x)' * w) );
end
fprintf('IRL Training fertig.\n');

%% 5. Simulation & Visualisierung
t_span = [0 5]; x0 = [1; 1; -1];
[t, x_out] = ode45(@(t,x) A_sys*x + B_sys*(-umax .* tanh((1./(2*diag(R).*umax)) .* (B_sys' * grad_sigma(x)' * w))), t_span, x0);

% Stellgrößen u berechnen
u_out = zeros(length(t), 2);
for i = 1:length(t)
    u_out(i,:) = (-umax .* tanh((1./(2*diag(R).*umax)) .* (B_sys' * grad_sigma(x_out(i,:)')' * w)))';
end

%% 6. Plots
% Figure 1: Zustandsverläufe
subplot(2,1,1);
plot(t, x_out, 'LineWidth', 2); grid on;
title('Zustandsverläufe (IRL Controller)'); xlabel('t [s]'); ylabel('x');
legend('x1','x2','x3');

subplot(2,1,2);
plot(t, u_out, 'LineWidth', 2); grid on;
title('Stellgrößen'); xlabel('t [s]'); ylabel('u');
legend('u1','u2');