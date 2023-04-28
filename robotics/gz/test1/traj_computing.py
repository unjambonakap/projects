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


# In[8]:


a = tgo.TGOSolver([3,3,3], [
    np.ones(3),
    np.zeros(3),
])
fx = a.func(np.array([-10, 5, 3]), np.array([-5, 7, -8]), 10)


# In[13]:


import seaborn as sns
tt = np.linspace(0, 10, 100)
xl = fx(tt)[0]
sns.lineplot(x=tt, y=xl)


# In[15]:


sns.lineplot(x=tt, y=np.cumsum(np.cumsum(xl)))


# In[2]:


from astropy import constants as const
from astropy import coordinates
from astropy import units as u


class GravitySpec(cmisc.PatchedModel):
    center: Vec3
    mass: float

    def __call__(self, pos: Vec3) -> Vec3:
        diff = self.center - pos
        const.G.value
        norm = diff.norm
        if norm < 1e-6: return Vec3.Zero()
        k = const.G.value * self.mass / norm**3
        return diff * k


gspec = GravitySpec(center=Vec3.ZeroPt(), mass=const.M_earth.value)

spec = rb_base.SolidSpec.Box(1, 1, 1, 1)

sctx = rb_gen.SceneContext()
tx = rb_gen.RBTree(sctx=sctx)
root = tx.add(
    rb_gen.RBDescEntry(
        data=rb_gen.RBData(base_name='root'),
        spec=spec,
        link_data=rb_gen.LinkData(
            spec=rb_gen.LinkSpec(type=rb_gen.RigidBodyLinkType.FREE, )),
    ))

alt = 300
c0 = coordinates.SphericalRepresentation(lon=0 * u.rad,
                                         lat=0 * u.rad,
                                         distance=(const.R_earth.value + alt) *
                                         u.m)
p0 = c0.to_cartesian().xyz.value
v0 = Vec3(
    coordinates.SphericalDifferential(d_lon=0 * u.rad,
                                      d_lat=1e-4 * u.rad,
                                      d_distance=0 *
                                      u.m).to_cartesian(base=c0).xyz.value)
rbl = tx.create(root, wl=rb_base.Transform.From(pos=p0))
ce = coordinates.SphericalRepresentation(lon=2e-5 * u.rad , lat=0 * u.rad, distance=(const.R_earth.value) * u.m)
pe = rb_base.Vec3(ce.to_cartesian().xyz.value)

sx = tgo.TGOSolver([3,3,3], [pe.vdata])

t_tgo = 100

sx.get(xp=p0, vp=v0.vdata, tgo=t_tgo)


# In[13]:


import chdrft.dsp.utils as dsp_utils

base = './traj.sdf'
tf = cmisc.tempfile.NamedTemporaryFile()

step = 0.1

with cmisc.tempfile.NamedTemporaryFile() as tf2:
    conv = helper.SDFConverter(base, world_name='scene1')
    conv.fill_with_rbtree()
    conv.write(tf.name)

    runner = helper.GZRunner.Build(tf.name)
    runner.set_physics(step)
    model = runner.model(helper.SDFConverter.MODEL_NAME)
    link = runner.model_link(model)

    runner.set_vel(model, v0)
    t0 = 0

    def conf_cb(*args):
        t = runner.info.sim_time.total_seconds()
        model_tsf = helper.GZDatas.tsf.query(runner, model.entity())
        model_v = helper.GZDatas.l_velocity.query(runner, link.entity())

        gx = gspec(model_tsf.pos_v)
        sim7.GravityCmd.GetOrCreate(runner.ecm,
                                    runner.world.entity()).set_data(gx.to_gz())

        dt= t_tgo-t
        if dt<0: dt=2
        target_acc = Vec3(
            sx.get(xp=model_tsf.pos, vp=model_v.vdata, tgo=dt)) - gx
        target_acc = Vec3(dsp_utils.linearize_clamp(target_acc.vdata, -50, 50, -50, 50))
        runner.set_force(link, target_acc * spec.inertial.mass)

        #sim7.InertialCmd.GetOrCreate(runner.ecm, link.entity()).set_data(a)

    runner.callbacks[helper.GZCallbackMode.PRE].append(conf_cb)

    from chdrft.sim.gz.helper import *
    stats = StatsGatherer(requests=[
        helpers.GZDatas.l_velocity.make_request('link', link.entity()),
        helpers.GZDatas.tsf.make_request('model', model.entity())
    ],
                          iter_downsample=10)
    stats.register(runner)

    #runner.reset()
    runner.server.run(True, 1000, False)
    dfx = pd.DataFrame.from_records(stats.records)
    runner.server.stop()
    runner.fixture.release()
    #del runner

#try_spec(spec)


# In[14]:


dfx['model.pos'] = dfx['model.tsf'].apply(lambda x: x.pos)
dfx[dfx.sim_time <= datetime.timedelta(seconds=t_tgo)]


# In[16]:


xp = dfx['model.tsf'].apply(lambda x: x.pos[0])
yp = dfx['model.tsf'].apply(lambda x: x.pos[1])
zp = dfx['model.tsf'].apply(lambda x: x.pos[2])
t = dfx.iterations
tt = dfx.sim_time.apply(lambda x:x.total_seconds())
sns.lineplot(x=tt, y=zp)


# In[ ]:


tgo.

