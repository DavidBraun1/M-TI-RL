clear; clc; close all;

%% ===== BLACK-BOX ENVIRONMENT (nur für die Simulation) =====
% Der Agent "sieht" diese Gleichungen NICHT.
A_env = [2 1 1; 1 -1 0; 0 0 1];
B_env = [0 0; 0 1; 1 0]; 

Q = eye(3);
R = eye(2);
umax = [3; 20];

env_step = @(x,u) A_env*x + B_env*u;   % wahre Dynamik (für den Agenten unbekannt)

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
T = 0.1;      % Rollout-Länge (wie auf deiner Folie)
dt = 0.01;
t_final = 40;

%% ===== SCHRITT 1: INITIALIZE NN =====
x = [1.2; -1; 0.8];      % Startzustand
w = 0.1*ones(21,1);      % Critic-Gewichte
alpha = 5e-4;            % Grundlernrate

history_w = [];

fprintf('--- Modellfreier Actor–Critic gestartet ---\n');

for t_curr = 0:T:(t_final-T)

    %% ===== SYSTEMWECHSEL (Demonstration) =====
    if abs(t_curr - 15) < 1e-9
        A_env = 2.5 * A_env;      % Umwelt ändert sich
        x = [1.5; -1.5; 1.2];     % Stoß im Zustand
        alpha = 5e-3;             % kurzzeitig stärkeres Lernen
        fprintf('*** SYSTEMWECHSEL BEI t=15 ***\n');
    elseif t_curr > 20
        alpha = 5e-4;
    end

    %% ===== SCHRITT 2: APPLY CONTROL & OBSERVE (ROLLOUT) =====
    x_t = x;
    p_t = 0;   % entspricht p(t) auf deiner Folie

    for sub_t = 0:dt:(T-dt)

        % ---- SCHRITT 2.1: POLICY (Actor) ----
        dV = grad_V(x,w);

        % Dimensionskorrigierte, modellfreie Policy
        u = -umax .* tanh( 0.1 * dV(1:2) );

        % minimale Exploration (echtes RL-Element)
        u = u + 0.02*[sin(2*t_curr); cos(1.5*t_curr)];
        u = max(min(u, 0.99*umax), -0.99*umax);

        % ---- SCHRITT 2.2: APPLY CONTROL & OBSERVE x(t) ----
        dx = env_step(x,u);   % Black-Box-Übergang
        x = x + dx * dt;

        % ---- SCHRITT 3: COMPUTE p(t) ----
        p_t = p_t + (x'*Q*x + u'*R*u) * dt;
    end

    %% ===== SCHRITT 4: BELLMAN-FEHLER =====
    V_now  = w' * sigma(x_t);
    V_next = w' * sigma(x);

    e_B = p_t + V_next - V_now;   % exakt wie auf deiner Folie

    % sanfte Skalierung (nur numerischer Schutz)
    e_B = e_B / max(1, abs(e_B));

    %% ===== SCHRITT 5: TRAIN NN (Critic) =====
    phi = sigma(x_t);
    phi = phi / (1 + 0.05*norm(phi));   % sanfte Feature-Skalierung

    w = w - alpha * (e_B * phi);

    % Sicherheitsclip (verhindert Explosion)
    if norm(w) > 1e4
        w = w / norm(w) * 1e4;
    end

    %% ===== SCHRITT 6: LOOP BACK TO STEP 2 =====
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
title('Modellfreier Actor–Critic');
xlabel('Zeit (s)');
ylabel('\omega_i');
legend('\omega_1','\omega_2','\omega_3','\omega_4','\omega_5','\omega_6');
ylim([0.07, 0.12])

%% ===== PHASE B: TEST DES GELERNTEN REGLERS (AUSREGELN) ===

fprintf('--- Starte Closed-Loop-Test (kein Lernen mehr) ---\n');

% Neue Anfangsbedingung wie im Paper
x_test = [1; -0.5; 0.8];
t_sim = 0:dt:5;

x_traj = zeros(3,length(t_sim));
u_traj = zeros(2,length(t_sim));

for k = 1:length(t_sim)

    x_traj(:,k) = x_test;

    % === WICHTIGER FIX: Paper-konsistente Policy ===
    dV = grad_V(x_test,w);               % ∇V(x)
    u = -umax .* tanh( 0.5 * (B_env' * dV) );  % <-- HJB-Form!

    % Stellgröße speichern
    u_traj(:,k) = u;

    % Systemschritt (echte Dynamik)
    dx = env_step(x_test,u);
    x_test = x_test + dx * dt;
end

%% ===== PLOTS (wie im Paper) =====
figure;

subplot(2,1,1);
plot(t_sim, x_traj', 'LineWidth', 2);
grid on;
title('Zustandsverlauf (Closed-Loop mit gelerntem Regler)');
xlabel('Zeit [s]');
ylabel('x');
legend('x_1','x_2','x_3');

subplot(2,1,2);
plot(t_sim, u_traj', 'LineWidth', 2);
grid on;
title('Stellgrößen (tanh-gesättigt)');
xlabel('Zeit [s]');
ylabel('u');
legend('u_1','u_2');
