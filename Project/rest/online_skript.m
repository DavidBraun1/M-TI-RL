clear; clc; close all;

%% ===== SYSTEM (Environment) =====
A_sys = [2 1 1; 1 -1 0; 0 0 1];
B_sys = [0 0; 0 1; 1 0]; 
Q = eye(3); 
R = eye(2); 
umax = [3; 20];

%% ===== BASISFUNKTIONEN (Critic) =====
sigma = @(x) [x(1)^2; x(2)^2; x(3)^2; x(1)*x(2); x(1)*x(3); x(2)*x(3); ...
              x(1)^4; x(2)^4; x(3)^4; x(1)^2*x(2)^2; x(1)^2*x(3)^2; ...
              x(2)^2*x(3)^2; x(1)^2*x(2)*x(3); x(1)*x(2)^2*x(3); ...
              x(1)*x(2)*x(3)^2; x(1)^3*x(2); x(1)^3*x(3); x(1)*x(2)^3; ...
              x(1)*x(3)^3; x(2)*x(3)^3; x(2)^2*x(3)];

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

grad_V = @(x,w) (grad_sigma(x)' * w);

%% ===== ZEITPARAMETER =====
T = 0.1;
dt = 0.01;
t_final = 40;

%% ===== SCHRITT 1: INITIALIZE NN =====
x = [1.2; -1; 0.8];
w = 0.1*ones(21,1);
alpha = 5e-4;          % Grundlernrate

history_w = [];

fprintf('--- Online Actor–Critic gestartet ---\n');

for t_curr = 0:T:(t_final-T)

    %% ===== SYSTEMWECHSEL BEI t = 15 (KRITISCH) =====
    if abs(t_curr - 15) < 1e-9
        A_now = 2.5 * A_sys;   % neue Dynamik
        x = [1.5; -1.5; 1.2];  % <-- WICHTIG: aktiver Zustandssprung
        alpha = 5e-3;          % kurzzeitig starkes Lernen
        fprintf('*** SYSTEMWECHSEL BEI t=15 ***\n');
    elseif t_curr > 20
        A_now = 2.5 * A_sys;
        alpha = 5e-4;
    else
        A_now = A_sys;
    end

    %% ===== SCHRITT 2: APPLY CONTROL & ROLLOUT =====
    x_t = x;
    p_t = 0;

    for sub_t = 0:dt:(T-dt)

        % Policy (Actor)
        dV = grad_V(x,w);
        u = -umax .* tanh( (1./(2.*diag(R).*umax)) .* (B_sys' * dV) );

        % Systemdynamik
        dx = A_now*x + B_sys*u;

        % Kosten p(t)
        p_t = p_t + (x'*Q*x + u'*R*u) * dt;

        % Zeitschritt
        x = x + dx * dt;
    end

    %% ===== SCHRITT 4: BELLMAN-FEHLER =====
    V_now  = w' * sigma(x_t);
    V_next = w' * sigma(x);

    e_B = p_t + V_next - V_now;

    % Sanfte Skalierung (behält Signal)
    e_B = e_B / max(1, abs(e_B));

    %% ===== LERNBEDINGUNG =====
    if norm(x_t) < 0.2
        history_w = [history_w; w'];
        continue;
    end

    %% ===== SCHRITT 5: TRAIN NN (Critic) =====
    phi = sigma(x_t);
    phi = phi / (1 + 0.05*norm(phi));

    grad_w = e_B * phi;
    w = w - alpha * grad_w;

    % Sicherheits-Clipping
    if norm(w) > 1e4
        w = w / norm(w) * 1e4;
    end

    %% ===== SCHRITT 6: LOOP =====
    history_w = [history_w; w'];
end

fprintf('--- Simulation beendet ---\n');

%% ===== PLOT =====
t_vec = 0:T:(t_final-T);

plot(t_vec, history_w(:,1:6), 'LineWidth', 2);
hold on;
line([15 15], [min(history_w(:)) max(history_w(:))], ...
     'Color','r','LineStyle','--');
grid on;
title('Online Actor–Critic');
xlabel('Zeit (s)');
ylabel('\omega_i');
legend('\omega_1','\omega_2','\omega_3','\omega_4','\omega_5','\omega_6');
ylim([0.08, 0.12])