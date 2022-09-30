cd /home/benoit/programmation/science/control

params.drag

clear Traj Defs;
clear RocketPhys Traj;
params = Defs.ssSRBDef();
params.drag.coeff = 0.5;
params.thrust_mul = 1;
solparams.endtime = 120;
solparams.altmindist = params.kEarthRadius + 50;
solparams.mindist = params.kEarthRadius + 150;
solparams.maxdist = params.kEarthRadius + 2000;
solparams.def = params;
traj0 = Traj.FindInit(solparams);
ev = traj0.states(:,end);
ev
[a,b] =Traj.Endfunc(ev, solparams)
Traj.ObjFunc1(ev, solparams)
norm(traj0.states(1:2,end)) - params.kEarthRadius



clear Traj;
hold off

hold on;
Traj.PlotIt(params, traj0.tl, traj0.states);
%Traj.PlotIt(params, res.soln.grid.time, res.soln.grid.state);
angles = linspace(0, 2*pi, 100);
xlim([-30 30])
ylim(params.kEarthRadius + [-30 30])
plot(cos(angles)*params.kEarthRadius, sin(angles)*params.kEarthRadius);
hold off;

clear Traj;
clear RocketPhys Defs;
res = Traj.Test2(solparams, traj0)

%ev = res.states(:,end);
ev = res.soln.grid.state(:,end);
ev
an = Traj.AnalyseEll(ev(1:2), ev(3:4), params.kGM)
an.a * (1+an.e)-params.kEarthRadius
ev
Traj.EllGetSpeed(an, an.a*(1-an.e))

res.controls(:,end-10:end)

clear Traj;

