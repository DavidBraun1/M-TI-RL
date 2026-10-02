clear; clc; close all;

%% SYSTEM
A_sys = [2 1 1; 1 -1 0; 0 0 1];
B_sys = [0 0; 0 1; 1 0]; 
Q = eye(3); 
R = eye(2); 
umax = [3; 20];

%% BASISFUNKTIONEN
sigma = @(x) [x(1)^2; x(2)^2; x(3)^2; x(1)*x(2); x(1)*x(3); x(2)*x(3); ...
              x(1)^4; x(2)^4; x(3)^4; x(1)^2*x(2)^2; x(1)^2*x(3)^2; ...
              x(2)^2*x(3)^2; x(1)^2*x(2)*x(3); x(1)*x(2)^2*x(3); ...
              x(1)*x(2)*x(3)^2; x(1)^3*x(2); x(1)^3*x(3); x(1)*x(2)^3; ...
              x(1)*x(3)^3; x(2)*x(3)^3; x(2)^2*x(3)];

%% GRADIENT
grad_sigma = @(x) [2*x(1), 0, 0; 0, 2*x(2), 0; 0, 0, 2*x(3); ...
    x(2), x(1), 0; x(3), 0, x(1); 0, x(3), x(2); ...
    4*x(1)^3, 0, 0; 0, 4*x(2)^3, 0; 0, 0, 4*x(3)^3; ...
    2*x(1)*x(2)^2, 2*x(2)*x(1)^2, 0; ...
    2*x(1)*x(3)^2, 0, 2*x(3)*x(1)^2; ...
    0, 2*x(2)*x(3)^2, 2*x(3)*x(2)^2; ...
    2*x(1)*x(2)*x(3), x(1)^2*x(3), x(1)^2*x(2); ...
    x(2)^2*x(3), 2*x(1)*x(2)*x(3), x(1)*x(2)^2; ...
    x(2)*x(3)^2, x(1)*x(3)^2, 2*x(1)*x(2)*x(3); ...
    3*x(1)^2*x(2), x(1)^3, 0; ...
    3*x(1)^2*x(3), 0, x(1)^3; ...
    x(2)^3, 3*x(1)*x(2)^2, 0; ...
    x(3)^3, 0, 3*x(1)*x(3)^2; ...
    0, x(3)^3, 3*x(2)*x(3)^2; ...
    0, 2*x(2)*x(3), x(2)^2];

%% ZEITDISKRETISIERUNG
T = 0.1;
dt = 0.01;
t_final = 40;

%% STARTWERTE
x = [1.2; -1; 0.8];
w_online = 0.1*ones(21,1);

%% RECURSIVE LEAST SQUARES
P = 1e3 * eye(21);
lambda = 0.99;
history_w = [];

fprintf('--- Online-Lernphase gestartet ---\n');

for t_curr = 0:T:(t_final-T)

    %% MODELLWECHSEL
    if t_curr >= 15
        A_now = 2.5 * A_sys;
        if abs(t_curr - 15) < 0.05
            x = [1.2; -1.2; 1.0];
            P = 1e3 * eye(21);
            fprintf('T=15: System geändert & RLS zurückgesetzt!\n');
        end
    else
        A_now = A_sys;
    end

    %% INTEGRATION
    x_t = x; % Zustand am Anfang des Intervalls
    cost_integral = 0;
    
    for sub_t = 0:dt:(T-dt)
        dV = grad_sigma(x)' * w_online;
        u = -umax .* tanh( (1./(2.*diag(R).*umax)) .* (B_sys' * dV) );
        
        % --- EXPLORATION NOISE HINZUFÜGEN (ESSENZIELL!) ---
        u_noisy = u + 2.0 * [sin(10*t_curr); cos(15*t_curr)]; 
        u_noisy = max(min(u_noisy, umax*0.99), -umax*0.99);

        dx = A_now*x + B_sys*u_noisy;
        
        % Kosten mit u_noisy berechnen
        Wu = 0;
        for i = 1:2
            Wu = Wu + 2*umax(i)*R(i,i)*u_noisy(i)*atanh(u_noisy(i)/umax(i)) + ...
                 umax(i)^2*R(i,i)*log(1 - (u_noisy(i)/umax(i))^2);
        end
        cost_integral = cost_integral + (x'*Q*x + Wu) * dt;
        x = x + dx * dt;
    end
    x_next = x; % Zustand am Ende des Intervalls

    %% INTEGRAL REINFORCEMENT LEARNING UPDATE
    % Regressor ist die Differenz der Basisfunktionen (Delta Phi)
    phi_delta = (sigma(x_next) - sigma(x_t)); 
    
    % Zielgröße ist das negative Kostenintegral
    y_k = -cost_integral; 

    %% RLS-UPDATE (Stabilisiert)
    if norm(phi_delta) > 1e-6 % Nur updaten, wenn Bewegung im System ist
        K = (P * phi_delta) / (lambda + phi_delta' * P * phi_delta);
        w_online = w_online + K * (y_k - phi_delta' * w_online);
        P = (P - K * phi_delta' * P) / lambda;
    end
end

fprintf('--- Simulation beendet ---\n');

%% PLOT
t_vec = 0:T:(t_final-T);

plot(t_vec, history_w(:,1:6), 'LineWidth', 2);
hold on;
line([15 15], [min(history_w(:)) max(history_w(:))], ...
     'Color','r','LineStyle','--');
grid on;
title('Online-Adaption (RLS + HJB-Residuum)');
xlabel('Zeit (s)');
ylabel('\omega_i');
legend('\omega_1','\omega_2','\omega_3','\omega_4','\omega_5','\omega_6');
