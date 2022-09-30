hold off
%state = x.soln.grid.state;
state = res.states;
plot(state(1,:), state(2,:));
hold on;
angles = linspace(0, 2*pi, 100);
plot(cos(angles)*params.kEarthRadius, sin(angles)*params.kEarthRadius);
hold off;

clear Traj;
clear RocketPhys Traj;
params = rocketDef();
params.drag.coeff = 0;
solparams.endtime = 500;
res = Traj.FindInit(params, solparams)
res.states(3:4,end)
norm(res.states(1:2,end))-params.kEarthRadius
res.tl(end)

clear Traj;

ev = res.states(:,end);
ev(3:4)
an = Traj.AnalyseEll(ev(1:2), ev(3:4), params.kGM)
an.a * (1+an.e)-params.kEarthRadius
ev
Traj.EllGetSpeed(an, an.a*(1-an.e))

res.controls(:,end-10:end)

clear Traj;

