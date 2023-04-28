#!/usr/bin/env python
# coding: utf-8

# In[1]:


from chdrft.config.env import init_jupyter

init_jupyter()
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


runner = None


# In[6]:


from astropy import constants as const
from astropy import coordinates
from astropy import units as u
from chdrft.sim.rb.scenes import *
assert 0

def balance_scene() -> SceneData:
  sctx = SceneContext()
  tx = RBTree(sctx=sctx)
  root = tx.add(
      RBDescEntry(
          data=RBData(base_name='root'),
          spec=SolidSpec.Cylinder(10, 1, 10),
          link_data=LinkData(spec=LinkSpec(type=RigidBodyLinkType.FREE)),
      )
  )
  box = tx.add(
      RBDescEntry(
          data=RBData(),
          spec=SolidSpec.Box(1, 1, 1, 1),
          link_data=LinkData(
              spec=LinkSpec(
                  type=RigidBodyLinkType.RIGID,
                  wr=Transform.From(pos=[0, 0, -8]),
              )
          ),
          parent=root,
      )
  )

  res = tx.create(root)
  fm = ForceModel.Id()
  return SceneData(sctx=sctx, fm=fm, tree=tx)


class GravitySpec(cmisc.PatchedModel):
    center: Vec3
    mass: float

    def __call__(self, pos: Vec3) -> Vec3:
        return (pos - self.center).uvec * -10
        diff = self.center - pos
        const.G.value
        norm = diff.norm
        if norm < 1e-6: return Vec3.Zero()
        k = const.G.value * self.mass / norm**3
        return diff * k


class Controller(cmisc.PatchedModel):
    target_z: Vec3
    max_ang: float = np.pi / 8
    err: object = None

    @cmisc.cached_property
    def pid(self) -> ctrl.PIDController:
        return ctrl.PIDController(kp=6,
                                  kd=1,
                                  control_range=ctrl.Range1D(
                                      -self.max_ang, self.max_ang))

    def proc(self, wl: rb_base.Transform) -> rb_base.Vec3:
        target_local = wl.inv @ self.target_z
        proj = -target_local[:2]
        self.err = proj
        action = self.pid.push(proj)
        rotx = rb_base.Transform.From(
            rot=rb_base.R.from_euler('yx', action * [1, -1]))
        return rotx @ Vec3.Z()


class TGOSnapshot(cmisc.PatchedModel):
    tgo: tgo.TGOSolver
    p: Vec3
    dp: Vec3
    tgo_t0: float
    t0: float

    def get_p(self, t: float) -> Vec3:
        f = self.tgo.func(self.p.vdata, self.dp.vdata, self.tgo_t0)
        res = Vec3(f(t - self.t0))
        return res

    def get_dp(self, t: float) -> Vec3:
        df = self.tgo.dfunc(self.p.vdata, self.dp.vdata, self.tgo_t0)
        return Vec3(df(t - self.t0))


class LanderControllerTGOState(cmisc.PatchedModel):
    snapshot: TGOSnapshot
    ctrl: Controller


class LanderController(cmisc.PatchedModel):
    tgo: tgo.TGOSolver
    t_tgo: float
    rl: rb_gen.RigidBodyLink
    state: LanderControllerTGOState = None
    refresh_t_seconds: int = 1

    def process(self, t: float, p: Transform, dp: Vec3, gravity: Vec3) -> Vec3:
        if self.state is None or t - self.state.snapshot.t0 > self.refresh_t_seconds:
            self.state = LanderControllerTGOState(
                snapshot=TGOSnapshot(tgo=self.tgo,
                                     p=p.pos_v,
                                     dp=dp,
                                     tgo_t0=self.t_tgo - t,
                                     t0=t),
                ctrl=Controller(target_z=p @ Vec3.Z(), ),
            )
        target_acc = self.state.snapshot.get_p(t) - gravity
        target_norm = target_acc.norm
        self.state.ctrl.target_z = target_acc.uvec
        action = p @ self.state.ctrl.proc(p)
        return action * target_norm * self.rl.mass


gspec = GravitySpec(center=Vec3.ZeroPt(), mass=const.M_earth.value)

alt = 300
c0 = coordinates.SphericalRepresentation(lon=0 * u.rad,
                                         lat=0 * u.rad,
                                         distance=(const.R_earth.value + alt) *
                                         u.m)
p0 = Vec3.Pt(c0.to_cartesian().xyz.value)
v0 = Vec3(
    coordinates.SphericalDifferential(d_lon=1e-6 * u.rad,
                                      d_lat=0e-4 * u.rad,
                                      d_distance=0 *
                                      u.m).to_cartesian(base=c0).xyz.value)
ce = coordinates.SphericalRepresentation(
    lon=3e-6 * u.rad,
    lat=0 * u.rad,
    distance=(const.R_earth.value) * u.m,
)
pe = rb_base.Vec3.Pt(ce.to_cartesian().xyz.value)

t_tgo = 30

sd = scenes.balance_scene()
rl = sd.sctx.roots[0].self_link
d0 = Vec3(sx.get(xp=p0.vdata, vp=v0.vdata, tgo=t_tgo)) - gspec(p0)
print('QO ', p0, v0, d0)
#d0 = -v0
wl0 = rb_base.make_rot_tsf(z=d0.uvec)
wl0.pos_v = p0
p0,v0,pe


# In[ ]:


t_tgo= 10
sx = tgo.TGOSolver(np.ones(3, dtype=int)*3, [pe.vdata, np.zeros(3)])
fx = sx.func(p0.vdata, v0.vdata, t_tgo)
tt = np.linspace(0, t_tgo, 100)
pl = fx(tt) - gspec(p0).vdata.reshape((-1,3)).T
xl = pl[0]
sns.lineplot(x=tt, y=xl)


# In[ ]:


pe.vdata


# In[ ]:


lc = LanderController(tgo=sx, t_tgo=t_tgo, rl=rl)
base = './traj.sdf'

step = 0.01
sim_time=  t_tgo*0.6

with cmisc.tempfile.NamedTemporaryFile() as tf:
    conv = helper.SDFConverter(base, world_name='scene1')
    conv.fill_with_rbtree(sd.tree)
    conv.write(tf.name)

    if runner is not None:
        runner.server.stop()
        runner.fixture.release()
        del runner
        
    runner = helper.GZRunner.Build(tf.name)
    runner.set_physics(step)
    model = runner.model(helper.SDFConverter.MODEL_NAME)

    model_link = runner.model_link(model)
    root_link = sim7.Link(model.link_by_name(runner.ecm, 'root'))
    box_link = sim7.Link(model.link_by_name(runner.ecm, 'root.RigidBodyLinkType.RIGID_SolidSpecType.BOX'))
   
    t0 = 0
    runner.set_vel_l(model_link, v0)
    print('qq', v0, wl0.inv @ v0)
    runner.set_wl(model, wl0)
    data = A(id=0)

    def conf_cb(*args):
        data.id += 1
        t = runner.info.sim_time.total_seconds()
        model_tsf = helper.GZDatas.tsf.query(runner, model.entity())
        model_vl = helper.GZDatas.l_velocity_l.query(runner, model_link.entity())
        model_vw = helper.GZDatas.l_velocity_w.query(runner, model_link.entity())
        print('STTEEEP ', model_tsf.pos, model_vl, model_vw, model_tsf @ model_vl)
        print(model_tsf.rot_xyzw)
        
        if data.id < 3:
            return
        

        gx = gspec(model_tsf.pos_v)
        sim7.GravityCmd.GetOrCreate(runner.ecm,
                                    runner.world.entity()).set_data(gx.to_gz())
        
        
        
        target_force = lc.process(t, model_tsf, model_vw, gx)
        if stats.active_record is not None:
            stats.active_record['pid_err'] = lc.state.ctrl.err
        print('foroce >> ', t, target_force)
        print()
        runner.set_force(box_link, target_force)

        #sim7.InertialCmd.GetOrCreate(runner.ecm, link.entity()).set_data(a)

    runner.callbacks[helper.GZCallbackMode.PRE].append(conf_cb)

    from chdrft.sim.gz.helper import *
    stats = StatsGatherer(requests=[
        helper.GZDatas.tsf.make_request('model', model.entity()),
    ], iter_downsample=10)
    stats.register(runner)

    #runner.reset()
    runner.server.run(True, int(sim_time / step), False)
    dfx = pd.DataFrame.from_records(stats.records)

#try_spec(spec)i


# In[ ]:


dfx['pid_err']


# In[ ]:


xp = np.stack(dfx['model.tsf'].apply(lambda x: x.pos - pe.vdata).values)
tt = dfx.sim_time.apply(lambda x:x.total_seconds())
tf = tt < 30

sns.lineplot(x=tt[tf], y=xp[:,1][tf])

