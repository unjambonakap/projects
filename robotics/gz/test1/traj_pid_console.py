from __future__ import annotations
from chdrft.config.env import init_jupyter

init_jupyter(run_app=True)
from pydantic import Field
import typing
from chdrft.sim.traj import tgo
from chdrft.sim.gz import helper
from chdrft.sim.rb import rb_gen
from chdrft.sim.rb import base as rb_base
from chdrft.sim.rb.base import Vec3, Transform
from chdrft.sim.rb import scenes
import seaborn as sns
import datetime
from gz import sim7, math7
from chdrft.gnc import ctrl
import chdrft.dsp.utils as dsp_utils
from chdrft.geo.satsim import gl
from chdrft.daq.stream import influx_connector

try:
  get_ipython().run_line_magic('matplotlib', 'qt')
except:
  ...

from astropy import constants as const
from astropy import coordinates
from astropy import units as u
from chdrft.sim.rb.scenes import *
from chdrft.sim.gz.helper import *
from chdrft.sim.traj.lander import *



class BaseSim(helper.SimInstance):
  v0: Vec3
  wl0: Transform

  def _setup(self, sh: SimHelper):
    self.runner.set_vel_l(sh.model_link, self.v0)
    sh.runner.set_wl(sh.model, self.wl0)


class BalanceSim(BaseSim):
  box_link: sim7.Link = None
  root_link: rb_gen.RigidBodyLink = None
  box_link: rb_gen.RigidBodyLink = None

  def _setup(self, sh: SimHelper):
    super()._setup(sh)

    sh.stats.requests.extend(
        [
            helper.GZDatas.l_velocity_l.make_request('root', self.sh.model_link.entity()),
            helper.GZDatas.l_velocity_w.make_request('root', self.sh.model_link.entity()),
        ]
    )
    self.root_link = sim7.Link(sh.model.link_by_name(sh.runner.ecm, 'root'))
    self.box_link = sim7.Link(
        sh.model.link_by_name(sh.runner.ecm, 'root.RigidBodyLinkType.RIGID_SolidSpecType.BOX')
    )


class GravitySim(BaseSim):
  func: object
  factor: float = 1

  def _proc(self):
    t = self.runner.info.sim_time.total_seconds()
    g = Vec3(self.func(t))
    self.runner.set_gravity(g)
    self.runner.set_force(self.sh.model_link, -g * self.factor)

    if self.stats.active_record is not None:
      self.stats.active_record['g'] = g


class BalancePIDSim(BalanceSim):
  cx: PIDZController

  def _proc(self):
    t = self.runner.info.sim_time.total_seconds()
    model_tsf = helper.GZDatas.tsf.query(self.runner, self.model.entity())
    dir = self.cx.proc(model_tsf)
    self.runner.set_force(self.box_link, model_tsf @ dir * 200)
    self.runner.set_vel_l(self.sh.model_link, Vec3.Zero())
    model_tsf.pos_v = Vec3.Zero()
    self.runner.set_wl(self.model, model_tsf)
    if self.stats.active_record is not None:
      self.stats.active_record['pid_err'] = self.cx.err


class BalanceTGOPidSim(BalanceSim):
  lc: LanderController
  gspec: tgo.GravitySpec

  def _proc(self):
    t = self.runner.info.sim_time.total_seconds()
    model_tsf = helper.GZDatas.tsf.query(self.runner, self.sh.model.entity())
    model_vl = helper.GZDatas.l_velocity_l.query(self.runner, self.sh.model_link.entity())
    model_vw = helper.GZDatas.l_velocity_w.query(self.runner, self.sh.model_link.entity())

    if self.id < 3:
      return

    gx = self.gspec(model_tsf.pos_v)
    self.runner.set_gravity(gx)

    target_force = self.lc.process(t, model_tsf, model_vw, gx)
    if self.stats.active_record is not None:
      self.stats.active_record['pid_err'] = self.lc.state.ctrl.err
      self.stats.active_record['force'] = target_force
      self.stats.active_record['g'] = gx
    self.runner.set_force(self.box_link, target_force)

    #sim7.InertialCmd.GetOrCreate(runner.ecm, link.entity()).set_data(a)


gspec = tgo.GravitySpec(center=Vec3.ZeroPt(), mass=const.M_earth.value)
sd = balance_scene()

ang = np.pi / 30
step = 0.01

alt0 = 300
alte = 6

p = gl.geocode('Paris')
ll_p = Z.deg2rad(np.array([p.longitude, p.latitude]))
ll0 = ll_p + [3e-5, 1e-6]
lle = ll_p
c0 = coordinates.SphericalRepresentation(
    lon=ll0[0] * u.rad, lat=ll0[1] * u.rad, distance=(const.R_earth.value + alt0) * u.m
)
p0 = Vec3.Pt(c0.to_cartesian().xyz.value)
v0 = Vec3(
    coordinates.SphericalDifferential(d_lon=-1e-5 * u.rad, d_lat=0e-4 * u.rad,
                                      d_distance=0 * u.m).to_cartesian(base=c0).xyz.value
)
ce = coordinates.SphericalRepresentation(
    lon=lle[0] * u.rad,
    lat=lle[1] * u.rad,
    distance=(const.R_earth.value + alte) * u.m,
)
pe = rb_base.Vec3.Pt(ce.to_cartesian().xyz.value)

t_tgo = 30
sx = tgo.TGOSolver(np.ones(3, dtype=int) * 3, [pe.vdata, np.zeros(3)])

rl = sd.sctx.roots[0].self_link
d0 = Vec3(sx.get(xp=p0.vdata, vp=v0.vdata, tgo=t_tgo)) - gspec(p0)
wl0 = rb_base.make_rot_tsf(z=d0.uvec)
wl0.pos_v = p0
p0, v0, pe

if 1:

  lc = LanderController(tgo=sx, t_tgo=t_tgo, rl=rl, pidz_params=dict(kp=10, kd=3), step_sec=step)

  sim_time = t_tgo * 1.01

  lc.reset()
  s1 = BalanceTGOPidSim(v0=v0, wl0=wl0, gspec=gspec, lc=lc)
  sh = helper.SimHelper(sd=sd, si=s1, step=step)
  sh.init()
  sh.stats.connector.connect_to(influx_connector(runid='test1'))

  dfx = sh.run(sim_time)

  dfx['pid_err']
  dfx['sim_time_s'] = dfx.sim_time.apply(lambda x: x.total_seconds())
  tt = dfx.sim_time_s.values
  a_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 0)(tt).T
  v_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 1)(tt).T
  p_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 2)(tt).T
  dfx['pid_err0'] = dfx.pid_err.apply(lambda x: x[0])
  dfx['pid_err1'] = dfx.pid_err.apply(lambda x: x[1])
  xp = np.stack(dfx['model.tsf'].apply(lambda x: x.pos).values)
  ap = np.stack(dfx['force'].apply(lambda x: x.vdata).values) / rl.mass
  gg = np.stack(dfx['g'].apply(lambda x: x.vdata).values)
  for i in range(3):
    dfx[f'p_planned{i}'] = p_planned[:, i]
    dfx[f'a_planned{i}'] = a_planned[:, i]
    dfx[f'p_act{i}'] = xp[:, i]
    dfx[f'a_act{i}'] = ap[:, i]
    dfx[f'g{i}'] = gg[:, i]

  df1 = dfx.melt('sim_time_s')
  sns.lineplot(
      data=df1[df1.variable.isin(
          (
              'p_act1',
              'p_planned1',
              #'a_act0', 'a_planned0', 'g0',
              #'a_act1', 'a_planned1', 'g1',
              #'a_act2', 'a_planned2', 'g2',
          )
      )],
      x='sim_time_s',
      y='value',
      hue='variable'
  )

  ss = SceneSaver(sd=sd)
  rx = sd.sctx.roots[0]
  for i in dfx.index.values:
    tsf = dfx.at[i, 'model.tsf']
    rx.self_link.link_data.q_joint.free_joint = FreeJoint.From(tsf=tsf)
    print(rx.self_link.link_data.q_joint.q, tsf)
    ss.push_state(dfx.at[i, 'sim_time_s'])
  cmisc.pickle.dump(ss.dump_obj(), open('/tmp/state.pickle', 'wb'))

  assert 0

  tt = dfx.sim_time.apply(lambda x: x.total_seconds())
  tf = tt < 30
  sns.lineplot(x=tt[tf], y=xp[:, 0][tf])

  tt = np.linspace(0, t_tgo)
  np.max(tgo.analyze_tgo(sx, p0, v0, 300, gspec).ang)

  fx = sx.func(p0.vdata, v0.vdata, t_tgo)
  tt = np.linspace(0, t_tgo, 100)
  pl = fx(tt) - gspec(p0).vdata.reshape((-1, 3)).T
  xl = pl[0]

  sns.lineplot(x=tt, y=xl)

if 0:
  sim_time = 5
  target_z = rb_base.Transform.From(rot=rb_base.R.from_euler('yx', np.ones(2) * ang)) @ Vec3.Z()
  v0 = Vec3.Zero()
  wl0 = Transform()
  cx = PIDZController(target_z=target_z, kp=10, kd=3, step_sec=step)
  s1 = BalancePIDSim(v0=v0, wl0=wl0, gspec=gspec, cx=cx)
  sh = helper.SimHelper(sd=sd, si=s1, step=step)
  sh.init()

  dfx = sh.run(sim_time)

  xp = np.stack(dfx['model.tsf'].apply(lambda x: (x @ Vec3.Z()).vdata).values)
  dfx
  dfx['sim_time_s'] = dfx.sim_time.apply(lambda x: x.total_seconds())
  dfx['pid_err0'] = dfx.pid_err.apply(lambda x: x[0])
  dfx['pid_err1'] = dfx.pid_err.apply(lambda x: x[1])
  df1 = dfx.melt('sim_time_s', value_vars=['pid_err0', 'pid_err1'])

  sns.lineplot(data=df1, x='sim_time_s', y='value', hue='variable')

if 0:
  sim_time = t_tgo * 1.1
  sd = trivial_scene()
  rl = sd.sctx.roots[0].self_link
  s1 = GravitySim(v0=v0, wl0=wl0, func=sx.func(p0.vdata, v0.vdata, t_tgo), factor=rl.mass)
  sh = helper.SimHelper(sd=sd, si=s1, step=step)
  sh.init()

  dfx = sh.run(sim_time)

  dfx['sim_time_s'] = dfx.sim_time.apply(lambda x: x.total_seconds())
  tt = dfx.sim_time_s.values
  a_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 0)(tt).T
  v_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 1)(tt).T
  p_planned = sx.dpfunc(p0.vdata, v0.vdata, t_tgo, 2)(tt).T
  xp = np.stack(dfx['model.tsf'].apply(lambda x: x.pos).values)
  gg = np.stack(dfx['g'].apply(lambda x: x.vdata).values)
  for i in range(3):
    dfx[f'p_planned{i}'] = p_planned[:, i]
    dfx[f'a_planned{i}'] = a_planned[:, i]
    dfx[f'p_act{i}'] = xp[:, i]
    dfx[f'g{i}'] = gg[:, i]

  df1 = dfx.melt('sim_time_s')
  sns.lineplot(
      data=df1[df1.variable.isin((
          'p_act0',
          'p_planned0',
      ))],
      x='sim_time_s',
      y='value',
      hue='variable'
  )
