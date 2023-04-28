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


# In[64]:


ang = np.pi/30
target_z = Vec3([0.01, -0.01, 1]).uvec
target_z = rb_base.Transform.From(rot=rb_base.R.from_euler('yx', np.ones(2) *ang)) @ Vec3.Z()


# In[113]:


class Controller:
    def __init__(self, target_z):
        max_ang  = np.pi/4
        self.pid = ctrl.PIDController(kp=3, kd=3, control_range=ctrl.Range1D(-max_ang, max_ang))
        self.target_z = target_z
        
    def proc(self, wl : rb_base.Transform) -> rb_base.Vec3:
        target_local = wl.inv @ target_z
        proj = -target_local[:2]
        action = self.pid.push(proj)
        rotx = rb_base.Transform.From(rot=rb_base.R.from_euler('yx', action * [1, -1]))
        return rotx @ Vec3.Z()
cx = Controller(target_z)

base = './traj.sdf'
sd = scenes.balance_scene()

step = 0.01
sim_time=  5

with cmisc.tempfile.NamedTemporaryFile() as tf:
    conv = helper.SDFConverter(base, world_name='scene1')
    conv.fill_with_rbtree(sd.tree)
    conv.write(tf.name)

    runner = helper.GZRunner.Build(tf.name)
    runner.set_physics(step)
    model = runner.model(helper.SDFConverter.MODEL_NAME)

    root_link = sim7.Link(model.link_by_name(runner.ecm, 'root'))
    box_link = sim7.Link(model.link_by_name(runner.ecm, 'root.RigidBodyLinkType.RIGID_SolidSpecType.BOX'))
   
    t0 = 0

    def conf_cb(*args):
        t = runner.info.sim_time.total_seconds()
        model_tsf = helper.GZDatas.tsf.query(runner, model.entity())
        dir = cx.proc(model_tsf)
        runner.set_force(box_link, model_tsf @ dir*200)
        runner.set_vel(model, Vec3.Zero())
        model_tsf.pos_v = Vec3.Zero()
        sim7.WorldPoseCmd.GetOrCreate(runner.ecm, model.entity()).set_data(model_tsf.to_gz())

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
    runner.server.stop()
    runner.fixture.release()
    #del runner

#try_spec(spec)i


# In[114]:


xp = np.stack(dfx['model.tsf'].apply(lambda x: (x @ Vec3.Z()).vdata).values)
tt = dfx.sim_time.apply(lambda x:x.total_seconds())
sns.lineplot(x=tt, y=xp[:,0])


# In[115]:


123

