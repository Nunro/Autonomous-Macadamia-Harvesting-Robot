%% =========================================================================
%  MCTR 474 - Macadamia Harvesting Robot - Motion Simulation
%  Nunro Brenkman 37540505 - NWU Potchefstroom
% =========================================================================
%  INSTRUCTIONS:
%    1. Save this file as mctr474_simulation.m in any folder
%    2. Open MATLAB R2025a, navigate to that folder
%    3. Press F5 or click the green Run button
%    4. Five figures appear automatically - do not close any until all load
%    5. To export: File > Save As > PDF (best for report) in each window
% =========================================================================
%  FIGURES PRODUCED:
%    Figure 1 - Autonomous Three-Zone Sequence
%    Figure 2 - Proportional Steering Convergence (trajectory + time plots)
%    Figure 3 - Manual Mode Skid-Steer Manoeuvre Trajectory
%    Figure 4 - Skid-Steer Principle: Six Control Input Scenarios
%    Figure 5 - Autonomous Run Wheel Speed and Zone Time Profile
%  TOOLBOXES REQUIRED: NONE (base MATLAB only)
% =========================================================================

clear; clc; close all;

%% ---- ROBOT PARAMETERS --------------------------------------------------

WR   = 0.125;      % Wheel radius [m]         (250 mm / 2)
L    = 0.62;       % Track width c-to-c [m]   (540 + 2x40 mm)
CL   = 0.60;       % Chassis length [m]
CW   = 0.45;       % Chassis width [m]
MASS = 25;         % Robot mass [kg]

% Speed derivation:
% Motor output ~140 RPM. Chain sprocket 2:1 reduction -> 70 RPM at wheel.
% v_max = (70/60) * 2 * pi * WR = 0.916 m/s at 100% PWM
V_MAX  = (70/60) * 2 * pi * WR;   % 0.916 m/s at 100% PWM
V_AUTO = V_MAX * 0.40;             % 0.366 m/s at 40% PWM
V_SRCH = V_MAX * (75/255);         % 0.270 m/s search pivot speed

% Controller gain (matched to firmware Kp = 0.5)
Kp = 0.18;         % [m/s per radian]

% Capture zone (from firmware constants)
CAP_DIST = 0.15;   % [m] blind spot threshold
CAP_DUR  = 1.20;   % [s] blind forward drive duration

DT = 0.02;         % [s] simulation timestep (50 Hz)

% Zone identifiers
SEARCH   = 1;
APPROACH = 2;
CAPTURE  = 3;

%% =========================================================================
%  FIGURE 1: AUTONOMOUS THREE-ZONE SEQUENCE
% =========================================================================

BALL1 = [3.5; -1.5];
s1    = [0; 0; pi/2];

N1_max   = 5000;
tr1      = zeros(3, N1_max);
zn1      = zeros(1, N1_max);
vL1_log  = zeros(1, N1_max);   % Logged for Figure 5
vR1_log  = zeros(1, N1_max);
t1log    = zeros(1, N1_max);
n1 = 0;  t1 = 0;

zone1      = SEARCH;
t_srch_end = 2.5;
t_cap_st   = 0;
vL1 = 0;  vR1 = 0;

while n1 < N1_max
    n1        = n1 + 1;
    tr1(:,n1) = s1;
    zn1(n1)   = zone1;
    t1log(n1) = t1;

    switch zone1
        case SEARCH
            vL1 =  V_SRCH;
            vR1 = -V_SRCH;
            if t1 >= t_srch_end
                zone1 = APPROACH;
            end

        case APPROACH
            dx1   = BALL1(1) - s1(1);
            dy1   = BALL1(2) - s1(2);
            dist1 = sqrt(dx1^2 + dy1^2);
            if dist1 <= CAP_DIST
                zone1    = CAPTURE;
                t_cap_st = t1;
                vL1 = V_AUTO;  vR1 = V_AUTO;
            else
                tgt1  = atan2(dy1, dx1);
                berr1 = wrap_angle(tgt1 - s1(3));
                tc1   = max(-V_AUTO*0.85, min(V_AUTO*0.85, Kp * berr1));
                vL1   = V_AUTO - tc1;
                vR1   = V_AUTO + tc1;
            end

        case CAPTURE
            vL1 = V_AUTO;  vR1 = V_AUTO;
            vL1_log(n1) = vL1;
            vR1_log(n1) = vR1;
            if t1 - t_cap_st >= CAP_DUR
                break;
            end
    end

    vL1_log(n1) = vL1;
    vR1_log(n1) = vR1;
    s1 = ddrive_step(s1, vL1, vR1, L, DT);
    t1 = t1 + DT;
end

tr1     = tr1(:, 1:n1);
zn1     = zn1(1:n1);
vL1_log = vL1_log(1:n1);
vR1_log = vR1_log(1:n1);
t1log   = t1log(1:n1);

idx1s = find(zn1 == SEARCH);
idx1a = find(zn1 == APPROACH);
idx1c = find(zn1 == CAPTURE);

fig1 = figure('Name','Three-Zone Autonomous Sequence', ...
              'NumberTitle','off','Color','w','Position',[30 80 820 620]);
ax1  = axes(fig1);
hold(ax1,'on');

if ~isempty(idx1s)
    plot(ax1, tr1(1,idx1s), tr1(2,idx1s), '-', ...
         'Color',[0.82 0.18 0.18],'LineWidth',2.8,...
         'DisplayName','Zone 1: Search (Pivot)');
end
if ~isempty(idx1a)
    plot(ax1, tr1(1,idx1a), tr1(2,idx1a), '-', ...
         'Color',[0.16 0.42 0.80],'LineWidth',2.8,...
         'DisplayName','Zone 2: Approach');
end
if ~isempty(idx1c)
    plot(ax1, tr1(1,idx1c), tr1(2,idx1c), '-', ...
         'Color',[0.08 0.60 0.28],'LineWidth',2.8,...
         'DisplayName','Zone 3: Capture Drive');
end

draw_robot(ax1, tr1(1,1), tr1(2,1), tr1(3,1), CL, CW, [0.78 0.78 0.78]);
if ~isempty(idx1a)
    p = idx1a(1);
    draw_robot(ax1, tr1(1,p), tr1(2,p), tr1(3,p), CL, CW, [0.40 0.62 0.92]);
end
if ~isempty(idx1c)
    p = idx1c(1);
    draw_robot(ax1, tr1(1,p), tr1(2,p), tr1(3,p), CL, CW, [0.08 0.60 0.28]);
end
draw_robot(ax1, tr1(1,end), tr1(2,end), tr1(3,end), CL, CW, [0.08 0.60 0.28]);

th_c = linspace(0, 2*pi, 120);
plot(ax1, BALL1(1)+CAP_DIST*cos(th_c), BALL1(2)+CAP_DIST*sin(th_c), ...
     '--','Color',[0.55 0.55 0.55],'LineWidth',1.3,'HandleVisibility','off');
plot(ax1, BALL1(1), BALL1(2), 'o','MarkerSize',18,'LineWidth',2.4,...
     'Color',[0.88 0.48 0.04],'MarkerFaceColor',[1.00 0.78 0.22],...
     'DisplayName','Target Ball');
plot(ax1, tr1(1,1), tr1(2,1), 'ks','MarkerSize',11,'LineWidth',2,...
     'MarkerFaceColor',[0.22 0.22 0.22],'DisplayName','Start');

text(ax1, tr1(1,1)+0.06, tr1(2,1)+0.12,'START',...
     'FontSize',8.5,'FontWeight','bold','Color',[0.18 0.18 0.18]);
text(ax1, BALL1(1)+0.14, BALL1(2)+0.12,'TARGET BALL',...
     'FontSize',8.5,'FontWeight','bold','Color',[0.68 0.35 0.02]);
text(ax1, BALL1(1)+CAP_DIST+0.04, BALL1(2)+0.02,...
     sprintf('Capture zone\n(r = %dmm)', round(CAP_DIST*1000)),...
     'FontSize',7.5,'Color',[0.50 0.50 0.50]);

axis(ax1,'equal'); box(ax1,'on'); grid(ax1,'on');
ax1.GridAlpha = 0.20;  ax1.FontSize = 10;
legend(ax1,'Location','northwest','FontSize',9.5,'Box','on');
title(ax1,{'Autonomous Three-Zone Navigation Sequence',...
    'Zone 1: Search  \rightarrow  Zone 2: Approach  \rightarrow  Zone 3: Capture'},...
    'FontSize',13,'FontWeight','bold');
xlabel(ax1,'X Position (m)','FontSize',11);
ylabel(ax1,'Y Position (m)','FontSize',11);
annotation(fig1,'textbox',[0.02 0.01 0.96 0.04],...
    'String',sprintf(['K_p = %.2f  |  v_{auto} = %.3f m/s (40%% PWM)  |  '...
    'Capture dist = %dmm  |  Capture drive = %.1f s  |  Search = %.1f s'],...
    Kp, V_AUTO, round(CAP_DIST*1000), CAP_DUR, t_srch_end),...
    'EdgeColor','none','FontSize',8,'Color',[0.42 0.42 0.42],...
    'HorizontalAlignment','center');

%% =========================================================================
%  FIGURE 2: PROPORTIONAL STEERING CONVERGENCE
% =========================================================================

BALL2 = [4.5; 2.5];
s2    = [0; 0; pi/4];

N2_max = 3000;
tr2    = zeros(3, N2_max);
t2log  = zeros(1, N2_max);
be2    = zeros(1, N2_max);
di2    = zeros(1, N2_max);
n2 = 0;  t2 = 0;

while n2 < N2_max
    dx2   = BALL2(1) - s2(1);
    dy2   = BALL2(2) - s2(2);
    dist2 = sqrt(dx2^2 + dy2^2);
    if dist2 <= CAP_DIST, break; end
    tgt2  = atan2(dy2, dx2);
    berr2 = wrap_angle(tgt2 - s2(3));
    tc2   = max(-V_AUTO*0.85, min(V_AUTO*0.85, Kp * berr2));
    vL2   = V_AUTO - tc2;
    vR2   = V_AUTO + tc2;
    n2        = n2 + 1;
    tr2(:,n2) = s2;
    t2log(n2) = t2;
    be2(n2)   = rad2deg(berr2);
    di2(n2)   = dist2;
    s2 = ddrive_step(s2, vL2, vR2, L, DT);
    t2 = t2 + DT;
end

tr2   = tr2(:,1:n2);
t2log = t2log(1:n2);
be2   = be2(1:n2);
di2   = di2(1:n2);

fig2 = figure('Name','Proportional Steering Convergence',...
              'NumberTitle','off','Color','w','Position',[80 80 1020 560]);

ax2L = subplot(1,2,1,'Parent',fig2);
hold(ax2L,'on'); axis(ax2L,'equal'); box(ax2L,'on'); grid(ax2L,'on');
plot(ax2L, tr2(1,:), tr2(2,:), '-','Color',[0.16 0.42 0.80],...
     'LineWidth',2.8,'DisplayName','Approach path');
draw_robot(ax2L, tr2(1,1), tr2(2,1), tr2(3,1), CL, CW, [0.78 0.78 0.78]);
draw_robot(ax2L, tr2(1,end), tr2(2,end), tr2(3,end), CL, CW, [0.08 0.60 0.28]);
ah = 1.1;
plot(ax2L,[tr2(1,1), tr2(1,1)+ah*cos(tr2(3,1))],...
         [tr2(2,1), tr2(2,1)+ah*sin(tr2(3,1))],...
     '--','Color',[0.82 0.18 0.18],'LineWidth',1.6,'HandleVisibility','off');
text(ax2L, tr2(1,1)+ah*cos(tr2(3,1))+0.06, tr2(2,1)+ah*sin(tr2(3,1))-0.12,...
     'Initial heading','FontSize',8,'Color',[0.82 0.18 0.18]);
th_c2 = linspace(0,2*pi,120);
plot(ax2L, BALL2(1)+CAP_DIST*cos(th_c2), BALL2(2)+CAP_DIST*sin(th_c2),...
     '--','Color',[0.55 0.55 0.55],'LineWidth',1.3,'HandleVisibility','off');
plot(ax2L, BALL2(1), BALL2(2),'o','MarkerSize',18,'LineWidth',2.4,...
     'Color',[0.88 0.48 0.04],'MarkerFaceColor',[1.00 0.78 0.22],...
     'DisplayName','Target Ball');
plot(ax2L, tr2(1,1), tr2(2,1),'ks','MarkerSize',11,'LineWidth',2,...
     'MarkerFaceColor',[0.22 0.22 0.22],'DisplayName','Start');
text(ax2L, tr2(1,1)+0.05, tr2(2,1)-0.22,'START','FontSize',8.5,...
     'FontWeight','bold','Color',[0.18 0.18 0.18]);
text(ax2L, BALL2(1)+0.10, BALL2(2)+0.14,'TARGET','FontSize',8.5,...
     'FontWeight','bold','Color',[0.68 0.35 0.02]);
legend(ax2L,'Location','northwest','FontSize',9,'Box','on');
title(ax2L,'Approach Trajectory','FontSize',12,'FontWeight','bold');
xlabel(ax2L,'X Position (m)','FontSize',11);
ylabel(ax2L,'Y Position (m)','FontSize',11);
ax2L.GridAlpha = 0.20;  ax2L.FontSize = 10;

ax2R = subplot(1,2,2,'Parent',fig2);
hold(ax2R,'on'); box(ax2R,'on'); grid(ax2R,'on');
yyaxis(ax2R,'left');
plot(ax2R, t2log, be2, '-','Color',[0.82 0.18 0.18],'LineWidth',2.2);
plot(ax2R, [t2log(1) t2log(end)],[0 0],'--',...
     'Color',[0.82 0.18 0.18],'LineWidth',0.9,'HandleVisibility','off');
ax2R.YColor = [0.82 0.18 0.18];
ylabel(ax2R,'Bearing Error (degrees)','FontSize',11);
yyaxis(ax2R,'right');
plot(ax2R, t2log, di2, '-','Color',[0.16 0.42 0.80],'LineWidth',2.2);
plot(ax2R, [t2log(1) t2log(end)],[CAP_DIST CAP_DIST],'--',...
     'Color',[0.08 0.60 0.28],'LineWidth',1.5,'HandleVisibility','off');
ax2R.YColor = [0.16 0.42 0.80];
ylabel(ax2R,'Distance to Target (m)','FontSize',11);
text(ax2R, t2log(max(1,end-round(0.3*n2))), CAP_DIST+0.06,...
     'Capture threshold','FontSize',8,'Color',[0.08 0.60 0.28],'FontWeight','bold');
title(ax2R,'Controller Time History','FontSize',12,'FontWeight','bold');
xlabel(ax2R,'Time (s)','FontSize',11);
ax2R.GridAlpha = 0.20;  ax2R.FontSize = 10;
legend(ax2R,{'Bearing Error','Distance to Target'},...
       'Location','northeast','FontSize',9,'Box','on');
sgtitle(fig2,'Proportional Steering: Bearing Error Convergence',...
        'FontSize',14,'FontWeight','bold');

%% =========================================================================
%  FIGURE 3: MANUAL MODE SKID-STEER MANOEUVRE TRAJECTORY
% =========================================================================

s3     = [0; 0; pi/2];
t_piv  = (pi/2) / (2*V_SRCH/L);   % Time for 90 degree pivot

seq3 = {
    V_AUTO,       V_AUTO,         2.5,     1;
   -V_SRCH,       V_SRCH,         t_piv,   2;
    V_AUTO,       V_AUTO,         2.5,     1;
    V_AUTO,       V_AUTO*0.18,    4.2,     4;
    V_AUTO,       V_AUTO,         2.2,     1;
    V_SRCH,      -V_SRCH,         t_piv,   3;
    V_AUTO,       V_AUTO,         2.5,     1;
};

n3_max = ceil(sum(cell2mat(seq3(:,3)))/DT) + 300;
tr3    = zeros(3, n3_max);
zn3    = zeros(1, n3_max);
n3     = 0;
snaps3 = s3';

for m = 1:size(seq3,1)
    vL3 = seq3{m,1};  vR3 = seq3{m,2};
    dur3 = seq3{m,3}; zid3 = seq3{m,4};
    for k = 1:round(dur3/DT)
        n3 = n3 + 1;
        if n3 > n3_max, break; end
        tr3(:,n3) = s3;
        zn3(n3)   = zid3;
        s3 = ddrive_step(s3, vL3, vR3, L, DT);
    end
    snaps3(end+1,:) = s3';
end
tr3 = tr3(:,1:n3);
zn3 = zn3(1:n3);

fig3 = figure('Name','Manual Skid-Steer Manoeuvres',...
              'NumberTitle','off','Color','w','Position',[130 80 820 640]);
ax3  = axes(fig3);
hold(ax3,'on');

msk_st = (zn3==1); msk_lp = (zn3==2); msk_rp = (zn3==3); msk_cv = (zn3==4);
if any(msk_st)
    plot(ax3, tr3(1,msk_st), tr3(2,msk_st), '-',...
         'Color',[0.16 0.42 0.80],'LineWidth',2.8,'DisplayName','Straight Drive');
end
if any(msk_lp)
    plot(ax3, tr3(1,msk_lp), tr3(2,msk_lp), '-',...
         'Color',[0.82 0.18 0.18],'LineWidth',2.8,...
         'DisplayName',sprintf('Left Pivot (90\\circ, %.1fs)', t_piv));
end
if any(msk_rp)
    plot(ax3, tr3(1,msk_rp), tr3(2,msk_rp), '-',...
         'Color',[0.88 0.48 0.04],'LineWidth',2.8,...
         'DisplayName',sprintf('Right Pivot (90\\circ, %.1fs)', t_piv));
end
if any(msk_cv)
    plot(ax3, tr3(1,msk_cv), tr3(2,msk_cv), '-',...
         'Color',[0.08 0.60 0.28],'LineWidth',2.8,'DisplayName','Gradual Right Curve');
end

sfc = {[0.78 0.78 0.78];[0.16 0.42 0.80];[0.82 0.18 0.18];...
       [0.16 0.42 0.80];[0.08 0.60 0.28];[0.16 0.42 0.80];...
       [0.88 0.48 0.04];[0.16 0.42 0.80]};
for s = 1:size(snaps3,1)
    draw_robot(ax3, snaps3(s,1), snaps3(s,2), snaps3(s,3), CL, CW, ...
               sfc{min(s,length(sfc))});
end

plot(ax3, snaps3(1,1), snaps3(1,2), 'ks','MarkerSize',11,'LineWidth',2,...
     'MarkerFaceColor',[0.22 0.22 0.22],'DisplayName','Start');
plot(ax3, snaps3(end,1), snaps3(end,2), 'k^','MarkerSize',11,'LineWidth',2,...
     'MarkerFaceColor',[0.58 0.58 0.58],'DisplayName','End');
text(ax3, snaps3(1,1)+0.05, snaps3(1,2)-0.18,'START',...
     'FontSize',8.5,'FontWeight','bold','Color',[0.18 0.18 0.18]);
text(ax3, snaps3(end,1)+0.05, snaps3(end,2)+0.12,'END',...
     'FontSize',8.5,'FontWeight','bold','Color',[0.30 0.30 0.30]);

axis(ax3,'equal'); box(ax3,'on'); grid(ax3,'on');
ax3.GridAlpha = 0.20;  ax3.FontSize = 10;
legend(ax3,'Location','best','FontSize',9.5,'Box','on');
title(ax3,{'Manual Mode: Skid-Steer Differential Drive Demonstration',...
    'Straight Drives, Pivot Turns, and Combined Curves'},...
    'FontSize',13,'FontWeight','bold');
xlabel(ax3,'X Position (m)','FontSize',11);
ylabel(ax3,'Y Position (m)','FontSize',11);
annotation(fig3,'textbox',[0.02 0.01 0.96 0.04],...
    'String',sprintf(['Wheel diam: 250mm  |  Track: 620mm  |  '...
    'Chassis: 600x450mm  |  Mass: %dkg  |  '...
    'v_{max} = %.3f m/s  |  v_{auto} = %.3f m/s'], MASS, V_MAX, V_AUTO),...
    'EdgeColor','none','FontSize',8,'Color',[0.42 0.42 0.42],...
    'HorizontalAlignment','center');

%% =========================================================================
%  FIGURE 4: SKID-STEER PRINCIPLE - SIX CONTROL INPUT SCENARIOS
%
%  Each panel shows the robot from above (facing upward) with wheel speed
%  vectors drawn as arrows. Green = forward, Red = reverse. Arrow length
%  is proportional to speed magnitude. The turning radius is derived from
%  the actual robot track width and the differential speed ratio.
%  This figure visually explains why differential wheel speeds produce turns.
% =========================================================================

% Six scenarios: {normalised_vL, normalised_vR, title_string}
% Normalised to V_AUTO where +1 = full autonomous forward speed
scen = {
    +1.00,  +1.00, 'Forward';
    +0.35,  +1.00, 'Left Curve';
    +1.00,  +0.35, 'Right Curve';
    -1.00,  +1.00, 'Left Pivot';
    +1.00,  -1.00, 'Right Pivot';
    -1.00,  -1.00, 'Reverse';
};

fig4 = figure('Name','Skid-Steer Principle: Control Scenarios',...
              'NumberTitle','off','Color','w','Position',[180 60 1100 700]);

for p = 1:6
    ax4p = subplot(2,3,p,'Parent',fig4);
    draw_skid_panel(ax4p, scen{p,1}, scen{p,2}, scen{p,3}, L, V_AUTO);
end

sgtitle(fig4,...
    {'Skid-Steer Differential Drive Principle',...
     'Green arrows = forward   |   Red arrows = reverse   |   Arrow length proportional to wheel speed'},...
    'FontSize',13,'FontWeight','bold');

annotation(fig4,'textbox',[0.02 0.01 0.96 0.04],...
    'String',sprintf(['Track width L = %.0fmm  |  v_{auto} = %.3f m/s (40%% PWM)  |  '...
    'Turning radius R = L(v_R+v_L) / [2(v_R-v_L)]  |  '...
    'Robot always turns toward the slower wheel'], L*1000, V_AUTO),...
    'EdgeColor','none','FontSize',8,'Color',[0.42 0.42 0.42],...
    'HorizontalAlignment','center');

%% =========================================================================
%  FIGURE 5: AUTONOMOUS RUN - WHEEL SPEED AND ZONE TIME PROFILE
%
%  Shows the left and right wheel speeds (m/s) over time during the
%  Figure 1 three-zone simulation, with zone-coloured background shading.
%  Directly connects the software state machine decisions to the physical
%  motor commands sent to the BTS7960 H-bridge drivers.
% =========================================================================

zone_fc = {[0.98 0.88 0.88]; [0.86 0.92 0.98]; [0.86 0.96 0.88]};
zone_lbl = {'Zone 1: Search','Zone 2: Approach','Zone 3: Capture'};

fig5 = figure('Name','Autonomous Run - Wheel Speed Profile',...
              'NumberTitle','off','Color','w','Position',[230 80 920 560]);

ax5a = subplot(2,1,1,'Parent',fig5);
hold(ax5a,'on'); box(ax5a,'on'); grid(ax5a,'on');

% Zone shading
t_lim = [t1log(1) t1log(end)];
v_lim = max(abs([vL1_log, vR1_log])) * 1.30;
for z = [SEARCH, APPROACH, CAPTURE]
    idz = find(zn1 == z);
    if ~isempty(idz)
        ts = t1log(idz(1));  te = t1log(idz(end));
        patch(ax5a, [ts te te ts], [-v_lim -v_lim v_lim v_lim],...
              zone_fc{z},'EdgeColor','none','FaceAlpha',1.0,...
              'HandleVisibility','off');
    end
end

% Zone boundary vertical lines
for z = [APPROACH, CAPTURE]
    idz = find(zn1 == z, 1, 'first');
    if ~isempty(idz)
        plot(ax5a, [t1log(idz) t1log(idz)], [-v_lim v_lim],...
             '-','Color',[0.60 0.60 0.60],'LineWidth',1.2,...
             'HandleVisibility','off');
    end
end

% Wheel speed lines
plot(ax5a, t1log, vL1_log, '-','Color',[0.16 0.42 0.80],...
     'LineWidth',2.2,'DisplayName','Left wheel v_L');
plot(ax5a, t1log, vR1_log, '--','Color',[0.82 0.18 0.18],...
     'LineWidth',2.2,'DisplayName','Right wheel v_R');
plot(ax5a, t_lim, [0 0], '-','Color',[0.55 0.55 0.55],...
     'LineWidth',0.8,'HandleVisibility','off');

% Zone labels
for z = [SEARCH, APPROACH, CAPTURE]
    idz = find(zn1 == z);
    if ~isempty(idz)
        t_mid = (t1log(idz(1)) + t1log(idz(end))) / 2;
        text(ax5a, t_mid, v_lim*0.88, zone_lbl{z},...
             'HorizontalAlignment','center','FontSize',8.5,...
             'FontWeight','bold','Color',[0.30 0.30 0.30]);
    end
end

xlim(ax5a, t_lim);
ylim(ax5a, [-v_lim v_lim]);
ax5a.GridAlpha = 0.20;  ax5a.FontSize = 10;
legend(ax5a,'Location','southeast','FontSize',9.5,'Box','on');
title(ax5a,'Left and Right Wheel Speeds','FontSize',11,'FontWeight','bold');
ylabel(ax5a,'Wheel Speed (m/s)','FontSize',11);
xlabel(ax5a,'Time (s)','FontSize',11);

% Lower panel: linear and angular velocity derived quantities
ax5b = subplot(2,1,2,'Parent',fig5);
hold(ax5b,'on'); box(ax5b,'on'); grid(ax5b,'on');

v_lin  = (vR1_log + vL1_log) / 2;
v_ang  = (vR1_log - vL1_log) / L;

for z = [SEARCH, APPROACH, CAPTURE]
    idz = find(zn1 == z);
    if ~isempty(idz)
        ts = t1log(idz(1));  te = t1log(idz(end));
        omega_lim = max(abs(v_ang)) * 1.30;
        lin_lim   = max(abs(v_lin)) * 1.30;
        patch(ax5b, [ts te te ts],...
              [-max(omega_lim, lin_lim) -max(omega_lim, lin_lim)...
               max(omega_lim, lin_lim)  max(omega_lim, lin_lim)],...
              zone_fc{z},'EdgeColor','none','FaceAlpha',1.0,...
              'HandleVisibility','off');
    end
end

for z = [APPROACH, CAPTURE]
    idz = find(zn1 == z, 1, 'first');
    if ~isempty(idz)
        y_lim_b = max(abs(v_ang)) * 1.30;
        plot(ax5b, [t1log(idz) t1log(idz)], [-y_lim_b y_lim_b],...
             '-','Color',[0.60 0.60 0.60],'LineWidth',1.2,...
             'HandleVisibility','off');
    end
end

yyaxis(ax5b,'left');
plot(ax5b, t1log, v_lin, '-','Color',[0.08 0.60 0.28],'LineWidth',2.2);
ax5b.YColor = [0.08 0.60 0.28];
ylabel(ax5b,'Linear Speed v (m/s)','FontSize',11);

yyaxis(ax5b,'right');
plot(ax5b, t1log, v_ang, '-','Color',[0.60 0.15 0.75],'LineWidth',2.2);
ax5b.YColor = [0.60 0.15 0.75];
ylabel(ax5b,'Angular Velocity \omega (rad/s)','FontSize',11);

plot(ax5b, t_lim, [0 0], '-','Color',[0.55 0.55 0.55],...
     'LineWidth',0.8,'HandleVisibility','off');

xlim(ax5b, t_lim);
ax5b.GridAlpha = 0.20;  ax5b.FontSize = 10;
legend(ax5b,{'Linear speed v','Angular velocity \omega'},...
       'Location','southeast','FontSize',9.5,'Box','on');
title(ax5b,'Derived Motion Quantities','FontSize',11,'FontWeight','bold');
xlabel(ax5b,'Time (s)','FontSize',11);

sgtitle(fig5,'Autonomous Run: Motor Command and Motion Profiles',...
        'FontSize',14,'FontWeight','bold');

%% ---- Console summary ---------------------------------------------------
fprintf('\n=============================================================\n');
fprintf('MCTR 474 Simulation complete. Five figures generated.\n');
fprintf('  V_MAX    = %.4f m/s  (100%% PWM, 70 RPM at wheel)\n', V_MAX);
fprintf('  V_AUTO   = %.4f m/s  ( 40%% PWM)\n',                  V_AUTO);
fprintf('  V_SRCH   = %.4f m/s  ( 29%% PWM)\n',                  V_SRCH);
fprintf('  t_pivot  = %.2f s    (90 degree pivot)\n',             t_piv);
fprintf('  Track L  = %.0f mm\n',                                 L*1000);
fprintf('  Chassis  = %.0f x %.0f mm\n',                          CL*1000, CW*1000);
fprintf('  Mass     = %d kg\n',                                   MASS);
fprintf('To export: File > Save As > PDF in each figure window.\n');
fprintf('=============================================================\n\n');

%% =========================================================================
%  LOCAL FUNCTIONS  (must remain at the bottom of the file)
% =========================================================================

function s = ddrive_step(s, vL, vR, L, dt)
    % Euler integration of differential drive kinematics.
    % omega = (vR-vL)/L: positive omega = CCW = left turn (vR > vL).
    v     = (vR + vL) / 2;
    omega = (vR - vL) / L;
    s(1)  = s(1) + v * cos(s(3)) * dt;
    s(2)  = s(2) + v * sin(s(3)) * dt;
    s(3)  = s(3) + omega * dt;
end

function draw_robot(ax, x, y, theta, cl, cw, fc)
    % Draw a filled rectangle (robot body) with heading arrow.
    corners = [-cl/2, -cw/2; cl/2, -cw/2; cl/2, cw/2; -cl/2, cw/2];
    R  = [cos(theta), -sin(theta); sin(theta), cos(theta)];
    rc = (R * corners')';
    patch(ax, x+rc(:,1), y+rc(:,2), fc, 'FaceAlpha',0.52,...
          'EdgeColor',[0.14 0.14 0.14],'LineWidth',1.2,...
          'HandleVisibility','off');
    alen = cl * 0.48;
    quiver(ax, x, y, alen*cos(theta), alen*sin(theta), 'AutoScale','off',...
           'Color',[0.05 0.05 0.05],'LineWidth',1.8,'MaxHeadSize',0.7,...
           'HandleVisibility','off');
end

function draw_skid_panel(ax, vL_n, vR_n, panel_title, L, V_AUTO)
    % Draw one skid-steer diagram panel showing wheel speed vectors.
    % vL_n, vR_n: normalised speeds (-1 to +1) relative to V_AUTO.
    % Arrow direction: up = forward (green), down = reverse (red).
    % Robot shown facing upward in each panel.

    cl  = 0.30;    % Half chassis length in panel units
    cw  = 0.225;   % Half chassis width
    wl  = 0.11;    % Wheel half-length
    wt  = 0.038;   % Wheel half-thickness
    wx  = cw + wt + 0.012;   % Wheel centre x-offset from chassis centre
    as  = 0.24;    % Max arrow length at |v_n| = 1

    hold(ax,'on');
    axis(ax, [-0.52 0.52 -0.68 0.68]);
    axis(ax, 'equal');
    axis(ax, 'off');

    % Chassis body
    patch(ax, [-cw -cw cw cw], [-cl cl cl -cl],...
          [0.82 0.82 0.82],'EdgeColor',[0.18 0.18 0.18],...
          'LineWidth',1.5,'FaceAlpha',0.88);

    % Front chevron indicator
    patch(ax, [-cw*0.45 cw*0.45 0], [cl*0.55 cl*0.55 cl*0.88],...
          [0.35 0.35 0.35],'EdgeColor','none','FaceAlpha',0.82);
    text(ax, 0, cl*0.70,'F','HorizontalAlignment','center',...
         'FontSize',8,'FontWeight','bold','Color','w');

    % Four wheels and speed arrows (left then right)
    for side = [-1, 1]
        wxi = side * wx;
        vn  = vL_n;
        if side > 0, vn = vR_n; end

        % Wheel body (dark rectangle)
        patch(ax, [wxi-wt, wxi+wt, wxi+wt, wxi-wt],...
                  [-wl, -wl, wl, wl],...
              [0.12 0.12 0.12],'EdgeColor',[0.05 0.05 0.05],'LineWidth',0.8);

        % Speed arrow (green forward, red reverse)
        if abs(vn) > 0.04
            ay = vn * as;
            if vn > 0
                acolor = [0.08 0.62 0.22];
            else
                acolor = [0.82 0.18 0.18];
            end
            quiver(ax, wxi, 0, 0, ay,'AutoScale','off','Color',acolor,...
                   'LineWidth',2.8,'MaxHeadSize',0.55);
        end

        % Speed value label
        if side < 0
            text(ax, wxi, -cl-0.12, sprintf('v_L = %+.2f', vn*V_AUTO),...
                 'HorizontalAlignment','center','FontSize',8.0,...
                 'Color',[0.18 0.18 0.18],'FontWeight','bold');
        else
            text(ax, wxi, -cl-0.12, sprintf('v_R = %+.2f', vn*V_AUTO),...
                 'HorizontalAlignment','center','FontSize',8.0,...
                 'Color',[0.18 0.18 0.18],'FontWeight','bold');
        end
    end

    % Rotation direction symbol and turning radius text
    omega_n = vR_n - vL_n;  % Proportional to angular velocity
    if abs(vL_n - vR_n) < 0.02
        if vL_n >= 0
            r_str = 'R = \infty  (Straight)';
        else
            r_str = 'R = \infty  (Reverse)';
        end
        rot_sym = '';
    elseif abs(vL_n + vR_n) < 0.02
        r_str   = 'R = 0  (Pivot)';
        if omega_n > 0
            rot_sym = char(8634);   % Unicode left rotation arrow
        else
            rot_sym = char(8635);   % Unicode right rotation arrow
        end
    else
        R_m = L * (vR_n + vL_n) / (2 * (vR_n - vL_n));
        if R_m > 0
            r_str = sprintf('Left Turn  |R| = %.2f m', abs(R_m));
            rot_sym = char(8634);
        else
            r_str = sprintf('Right Turn  |R| = %.2f m', abs(R_m));
            rot_sym = char(8635);
        end
    end

    if ~isempty(rot_sym)
        text(ax, 0, 0, rot_sym,'HorizontalAlignment','center',...
             'FontSize',26,'Color',[0.50 0.50 0.50],'FontWeight','bold');
    end

    text(ax, 0, cl+0.20, r_str,'HorizontalAlignment','center',...
         'FontSize',8.0,'Color',[0.38 0.38 0.38],'FontStyle','italic');

    title(ax, panel_title,'FontSize',11,'FontWeight','bold');
end

function a = wrap_angle(a)
    % Wrap angle to [-pi, pi]
    a = mod(a + pi, 2*pi) - pi;
end
