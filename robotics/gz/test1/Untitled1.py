#!/usr/bin/env python
# coding: utf-8

# In[1]:


#need
#export LD_LIBRARY_PATH=/usr/local/lib


from __future__ import annotations
init_jupyter()
import sys
import enum
sys.path.append('/usr/lib/python/')
sys.path.append('/usr/local/lib')
sys.path.append('/usr/local/lib/python')

import os
os.environ['GZ_DEBUG_COMPONENT_FACTORY'] = 'true'
from gz.common import set_verbosity
from gz.sim7 import TestFixture, World, world_entity
from gz.math7 import Vector3d
from gz import sim7, math7
import chdrft.sim.rb.base as sim_base
import pydantic

from gz.msgs.wrench_pb2 import Wrench
from gz.msgs.vector3d_pb2 import Vector3d
from gz.msgs.world_control_pb2 import WorldControl
from gz.msgs.boolean_pb2 import Boolean
from gz.msgs.physics_pb2 import Physics
from gz.transport12 import Node
set_verbosity(4)


# In[2]:


runner.node.request('/world/dem_heightmap/set_physics', Physics(max_step_size=3e-3), 1, Boolean())


# In[33]:


runner.reset()
runner.server.run(False, 0, True)


# In[ ]:


from chdrft.display.service import oplt
oplt.plot(dict(obs=ctrl_obs), 'metric')

#oplt.plot(dict(obs=gp.state_src), 'metric')


# In[ ]:


n.list_topics(), n.list_services()


# In[ ]:


u = n.service_info('/world/dem_heightmap/set_physics')
for x in u:
    print(x.req_type_name())
    print(x.rep_type_name())


# In[3]:


input_file = "/usr/share/gz/gz-sim7/worlds/dem_moon.sdf"
input_file = "./dem_moon.sdf"


class GZCallbackMode(enum.Enum):
    PRE = enum.auto()
    POST = enum.auto()
    UPDATE = enum.auto()
    RESET = enum.auto()


class GZRunner(cmisc.PatchedModel):
    fixture: TestFixture
    fname: str
    info: sim7.UpdateInfo = None
    ecm: sim7.EntityComponentManager = None
    server: sim7.Server = None
    node: Node = None
    callbacks: dict[GZCallbackMode, list] = pydantic.Field(default_factory=lambda: cmisc.defaultdict(list))
    
    @classmethod
    def Build(cls, fname: str) -> GZRunner:
        fixture = TestFixture(input_file)
        res = cls(fixture=fixture, fname=fname, node=Node())
        res.build()
        return res
    
    def init_cbs(self):
        self.fixture.on_post_update(lambda info, ecm: self.cb(GZCallbackMode.POST, info, ecm))
        self.fixture.on_pre_update(lambda info, ecm: self.cb(GZCallbackMode.PRE, info, ecm))
        self.fixture.on_update(lambda info, ecm: self.cb(GZCallbackMode.UPDATE, info, ecm))
        self.fixture.on_reset(lambda info, ecm: self.cb(GZCallbackMode.RESET, info, ecm))
    
    def build(self):
        self.init_cbs()
        self.fixture.finalize()
        self.server = self.fixture.server()
        self.ecm = self.server.entity_comp_mgr(0)
        
    @cmisc.logged_f()
    def cb(self, mode: GZCallbackMode, info: sim7.UpdateInfo, ecm: sim7.EntityComponentManager):
        self.info = info
        self.ecm = ecm
        for cb in self.callbacks[mode]:
            cb(self)
            
            
    @property
    def world(self) -> sim7.World:
        wid = world_entity(self.ecm)
        return World(wid)
    
    def model(self, name:str) -> sim7.Model:
        mid = self.world.model_by_name(self.ecm, name)
        return sim7.Model(mid)

    def model_link(self, model: sim7.Model):
        return sim7.Link(model.link(self.ecm))
        #link.world_linear_velocity(mg)
        
    def reset(self):
        wc = WorldControl(reset=dict(all=True), pause=False)
        sl = '/world/dem_heightmap/control'
        self.node.request(sl, wc, 1, Boolean())
        self.server.run(True, 1, False)
        self.init_cbs()
            
class GZDataMapping(cmisc.PatchedModel):
    cl: object
    converter: object
    name: str
        
    def make_request(self, target: str, entity: int) -> StatRequest:
        return StatRequest(mapping=self, target=target, entity=entity)
        
        
def gzquat2local(quat: math7.Quaterniond) -> sim_base.R:
    return sim_base.R.from_quat(quat.xyzw())
    
class GZDatas:
    l_velocity = GZDataMapping(name='vl', cl=sim7.LinearVelocity, converter=lambda x: sim_base.Vec3(np.array(x)))
    a_velocity = GZDataMapping(name='va', cl=sim7.AngularVelocity, converter=lambda x: sim_base.Vec3(np.array(x)))
    tsf = GZDataMapping(name='tsf', cl=sim7.Pose, converter=lambda x: sim_base.Transform.From(pos=np.array(x.pos()), rot=gzquat2local(x.rot())))
    
class StatRequest(cmisc.PatchedModel):
    mapping: GZDataMapping
    target: str
    entity: int
        
    @property
    def col_name(self) -> str:
        return f'{self.target}.{self.mapping.name}'
    
    def fill(self, record:dict, runner: GZRunner):
        a = self.mapping.cl.GetOrCreate(runner.ecm, self.entity)
        record[self.col_name] = self.mapping.converter(a.data())
    
class StatEntry(cmisc.PatchedModel):
    record: dict[GZDatas, object]
        
class StatsGatherer(cmisc.PatchedModel):
    records: list[dict[str, object]] = pydantic.Field(default_factory=list)
    requests: list[StatRequest]
    iter_downsample: int = 1
        
        
    def register(self, runner: GZRunner):
        runner.callbacks[GZCallbackMode.PRE].append(self.record)
        
    def record(self, runner: GZRunner):
        if runner.info.iterations % self.iter_downsample != 0: return
        
        entry = dict()
        entry['sim_time'] = runner.info.sim_time
        entry['real_time'] = runner.info.real_time
        entry['iterations'] = runner.info.iterations
        for req in self.requests:
            req.fill(entry, runner)
        self.records.append(entry)
    


# In[4]:


get_ipython().run_line_magic('pdb', 'off')

n = Node()
runner = GZRunner.Build(input_file)
box = runner.model('box')
link = runner.model_link(box)


# In[5]:


runner.server.run(True, 1, False)
runner.info.iterations

sim7.LinearVelocityCmd.GetOrCreate(runner.ecm, box.entity()).set_data(math7.Vector3d(*(1,1,10)))

sim7.LinearVelocityCmd.Remove(runner.ecm, box.entity())
sim7.LinearVelocityCmd.GetOrCreate(runner.ecm, box.entity()).data()
runner.server.run(True, 1, False)
sim7.LinearVelocityCmd.Remove(runner.ecm, box.entity())


# In[16]:


runner.ecm.component_types


# In[23]:


sim7.LinearVelocityCmd.GetOrCreate(runner.ecm, box.entity()).set_data(math7.Vector3d(*(1,1,10)))
sim7.GravityCmd.GetOrCreate(runner.ecm, runner.world.entity()).set_data(math7.Vector3d(*(0,0,-0)))
runner.reset()
runner.server.run(True, 1000, False)


# In[26]:


sim7.LinearVelocityCmd.GetOrCreate(runner.ecm, box.entity()).set_data(math7.Vector3d(*(1,1,10)))
runner.server.run(True, 1000, False)


# In[14]:


runner.world.gravity(runner.ecm)


# In[9]:


runner.world.


# In[ ]:


from chdrft.sim.rb import rb_player
from chdrft.dsp.utils import linearize_clamp
from chdrft.utils.math import make_norm
import shapely.geometry
import shapely.ops
from chdrft.utils.geo import Circle, to_shapely
max_ang = np.pi/8
max_norm = 40
max_norm = 400
max_ang = np.pi/4
p0 = np.array([-np.sin(max_ang), np.cos(max_ang)]) * 2
allowed_shape = shapely.geometry.Polygon([(0,0), p0, p0 * [-1,1]])
angle_shape = to_shapely(Circle(pos=np.zeros(2), r=1).polyline())

def linearize_clamp_abs(v, low, high, ylow, yhigh):
    
    sgn = cmisc.sign(v)
    if abs(v) < low: return ylow * sgn
    return linearize_clamp(abs(v), low, high, ylow, yhigh) * sgn



def getp(inx: np.ndarray) -> np.ndarray:
    p = shapely.geometry.Point(inx[:2])
    p0 =shapely.ops.nearest_points(angle_shape, p)[0]
    v = np.array([p0.x, p0.y, 1])
    v = make_norm(v) * inx[2]
    print(v)
    return v

    
@cmisc.logged_f()
def controller2ctrl(controller: rb_player.SceneControllerInput) -> np.ndarray:
    cx = controller.scaled_ctrl
    v = np.array([
        linearize_clamp_abs(cx[rb_player.ButtonKind.LEFT_X], 0.5, 1, 0, 1),
        linearize_clamp_abs(cx[rb_player.ButtonKind.LEFT_Y], 0.5, 1, 0, 1),
        linearize_clamp(cx[rb_player.ButtonKind.LTRT], 0, 1, 0, 1),
                 ])
    res =  getp(v) * max_norm
    print(v,res)
    return res


# In[ ]:


parameters = rb_player.InputControllerParameters(controller2ctrl=controller2ctrl)

s0 = rb_player.SceneControllerInput(parameters=parameters)
s0.stop = True
ctrl_obs = rb_player.gamepad2scenectrl_io(s0)

gp = rb_player.OGamepad(1)
#ctrl_obs.last.gp = gp
rb_player.pipe_connect(
  gp.state_src,
  ctrl_obs,
)
gp.start()


# In[ ]:


def conf_torque(*args):
    a = sim7.ExternalWorldWrenchCmd.GetOrCreate(runner.ecm, link.entity())
    ctrl = ctrl_obs.last.get_ctrl()
    w = Wrench(force=Vector3d(x=ctrl[0], y=ctrl[1], z=ctrl[2]))
    a.set_data(w)
    
    b = math7.MassMatrix3d()
    b.set_mass(100)
    b.set_diagonal_moments(math7.Vector3d(1,2,3))
    a = math7.Inertiald(b, math7.Pose3d())
    sim7.InertialCmd.GetOrCreate(runner.ecm, link.entity()).set_data(a)
##sim7.LinearVelocityCmd.GetOrCreate(runner.ecm, box.entity()).set_data(math7.Vector3d(*(1,1,10)))
runner.callbacks[GZCallbackMode.POST].clear()
runner.callbacks[GZCallbackMode.PRE].clear()
runner.callbacks[GZCallbackMode.POST].append(conf_torque)


# In[ ]:


runner.server.


# In[7]:


runner.reset()
runner.server.run(True, 10000, False)


# In[ ]:


stats = StatsGatherer(requests=[GZDatas.l_velocity.make_request('link' , link.entity()), GZDatas.tsf.make_request('box', box.entity())], iter_downsample=10)
stats.register(runner)

runner.reset()
runner.server.run(True, 1000, False)
dfx =pd.DataFrame.from_records(stats.records)
pd.DataFrame.from_records(stats.records)


# In[ ]:


dfx =pd.DataFrame.from_records(stats.records)

import seaborn as sns
tmp = dfx['box.tsf'].apply(lambda x:x.pos[2])
sns.lineplot(data=tmp)


# In[ ]:


runner.server.run(True, 10000, False)

