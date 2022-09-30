#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from dataclasses import dataclass, field
import xarray as xr
from typing import Callable, List
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase

from collections import deque
from pydantic import BaseModel, Field
from chdrft.sim.phys import *

df_consts = pd.read_csv('../data/consts.csv')
thrust_w = pd.read_csv('../data/ds_thrust_and_weight.csv')
data = Z.FileFormatHelper.Read('../data/measures.pickle')


# In[2]:


sctx = SceneContext()
ma = MoveDesc(local_vel=Vec3([0,0,0]), local_rotang=Vec3.X(), rotspeed=-3)
ra = sctx.create_rb(spec=SolidSpec.Sphere(2, 0.5), name='Wheel')
ral = RigidBodyLink(child=ra,
               local2world=Transform.From(pos=[0,0,0], rot=R.from_rotvec(Vec3.X().data * 0)),
               move_desc=ma,
                      link_data=LinkData(static=False, pivot_rotaxis=Vec3.X()),
                    )
mzero = MoveDesc()
rb = sctx.create_rb(spec=SolidSpec.Box(1,1,1,1), name='WheelCase')
rbl = RigidBodyLink(child=rb,
               local2world=Transform.From(pos=[-0, 0, 0], rot=R.from_rotvec(Vec3.X().data * 0)),
               move_desc=mzero)
wheel_system = sctx.compose([ral, rbl], name='WheelSys')

wlink = RigidBodyLink(child=wheel_system,
                move_desc =  MoveDesc(local_vel=Vec3([0,0,0]), local_rotang=Vec3.Z(), rotspeed=3),
                local2world = Transform.From(pos=[1,2,3], rot=R.from_rotvec(Vec3.Y().data * 0)),
                      link_data=LinkData(static=False, pivot_rotaxis=Vec3.Z()),
                     )

r0 = sctx.compose([wlink])
r0l = RigidBodyLink.FixedLink(r0)
px = r0.get_particles(3000)

#px.plot()
r0l.world_angular_momentum(only_static=False), r0l.world_linear_momentum(only_static=False)
wlink.world_angular_momentum(1)
ral.world_angular_momentum(1)
wlink.world_angular_momentum(1)


# In[3]:


tc = TorqueComputer()
tc.update(0.1, r0l)




# In[ ]:


get_ipython().run_line_magic('pdb', '')


# In[6]:


LinkChainEntry(lw=Transform.From(), rot_w_wl=Vec3.Zero())


# In[13]:


Vec3.Zero().exp_rot()


# In[7]:


get_ipython().run_line_magic('pdb', '')


# In[ ]:


get_ipython().run_line_magic('pdb', 'off')


# In[6]:


px = r0.get_particles(30000, use_importance=0)
px.linear_momentum(), px.angular_momentum()


# In[12]:


px.plot(fx=0.1, by_col=1)


# In[39]:


wheel_system.get_particles(3000).plot()


# In[2]:


m0 = MoveDesc(worldvel=Vec3.Zero())
r0 = RigidBody(spec=SolidSpec.Sphere(2, 1),
               local2world=Transform.From(pos=[-2, 0, 0]),
               move_desc=m0)
r1 = RigidBody(spec=SolidSpec.Sphere(2, 1),
               local2world=Transform.From(pos=[2, 0, 0]),
               move_desc=m0)
r2 = RigidBody.Compose([r1, r0])


# In[4]:


r0.world_inertial_momentum.data @ ra.local2world.rot_R.apply((ra.move_desc.local_rotang * ra.move_desc.rotspeed).data)


# In[ ]:





# In[39]:


ra.local2world.rot_R.apply((ra.move_desc.local_rotang * ra.move_desc.rotspeed).data)


# In[6]:


r0.spec.desc.importance_sampling0(100)


# In[2]:


class SimulEvent(Enum):
    FORCE_GENERATOR = 'force_generator'
    POST_FORCE_GENERATOR = 'post_force_generator'


class ForceEntry(cmisc.PatchedModel):
    pos: Vec3
    f: Vec3
    tid: int


class SimulEnv(cmisc.PatchedModel):
    id2force: dict
    hooks: "dict[SimulEvent, list[ObjectLogic]]" = Field(
        default_factory=lambda: defaultdict(list))
    objs: "list[ObjectLogic]" = Field(default_factory=list)
    rb2obj: dict = Field(default_factory=dict)

    def add(self, obj: "ObjectLogic"):
        self.objs.append(obj)
        self.rb2obj[obj.rb] = obj

    def add_to_updates(self, event, logic):
        self.hooks[event].append(logic)


class SimulBlock(cmisc.PatchedModel):
    env: SimulEnv

    def process(self):
        forces = dict()
        for x in self.env.hooks[SimulEvent.FORCE_GENERATOR]:
            f = x.process_event(SimulEvent.FORCE_GENERATOR)
            forces[x.id] = f
        for x in self.env.hooks[SimulEvent.POST_FORCE_GENERATOR]:
            x.process_event(SimulEvent.POST_FORCE_GENERATOR)


class ObjectLogic(cmisc.PatchedModel):
    obj: RigidBody = None
    env: SimulEnv = None
    spec: None = None
    id: None = None

    def register(self, env):
        self.env = env
        self._register()

    def build(self):
        self.id = self.spec.get_new_id()
        self._build()

    def _build(self):
        pass

    def _register(self):
        pass

    def _update(self, status, control):
        pass

    def process_event(self, event: SimulEvent):
        pass

    def update(self):
        status = self.env.get_status(self)
        ctrl = self.env.get_status(self)
        self._update(status, ctrl)

class ReactionWheelControl(cmisc.PatchedModel):
class ReactionWheelStatus(cmisc.PatchedModel):
    speed: float
class ReactionWheelSpec(cmisc.PatchedModel):
    ww: float
    w0: float
    
    

class ReactionWheel(ObjectLogic):

    def __init__(self, w0, ww, i0, dims):
        speed = 0
        spec = SolidSpec(mass=w0 + ww, I_0=i0)
        rb = RigidBody(spec=spec,
                       local2world=Transform.From(),
                       move_desc=MoveDesc(rotspeed=speed,
                                          local_rotang=Vec3.Z()))
        super().__init__(ww=ww, w0=w0, speed=speed, obj=rb, obj=rb)

    def _register(self):
        self.env.add_to_updates(SimulEnv.FORCE_GENERATOR, self)
        self.env.add_to_updates(SimulEnv.POST_FORCE_GENERATOR, self)

    def process_event(self, event: SimulEvent):
        if event is SimulEvent.FORCE_GENERATOR:
            pass
        elif event is SimulEvent.POST_FORCE_GENERATOR:
            pass


class ThrusterStatus(cmisc.PatchedModel):
    rotation: R = Field(default_factory=R.identity)
    thrust: float = 0


class ThrusterControl(cmisc.PatchedModel):
    throttle: float
    pistons: np.ndarray
    override_status: ThrusterStatus = None


class ThrusterDesc(cmisc.PatchedModel):
    gimbal_max_ang: float = None
    thrust_provider: float = None
        
    def compute_status(self, ctrl: ThrusterControl) -> ThrusterStatus:
        if ctrl.override_status: return ctrl.override_status
        assert 0


class Thruster(ObjectLogic):
    force_pos: Vec3 = None
    force_direction: Vec3 = None
    desc: ThrusterDesc
    ctrl: ThrusterControl = None

    status: ThrusterStatus = Field(default_factory=ThrusterStatus)

    def _register(self):
        self.env.add_to_updates(SimulEnv.FORCE_GENERATOR, self)

    def process_event(self, event: SimulEvent):
        if event is SimulEvent.FORCE_GENERATOR:
            self.status = self.desc.compute_status(self.ctrl)
            dir = self.status.rot * self.force_direction
            fe = ForceEntry(pos=self.force_pos, f=dir * status.thrust)
            return fe


# In[5]:


def an2(df1):
    i1, i2 = df1.inertia_tensor
    p1, p2 = df1.tank2com
    m1, m2 = df1.mass

    p1 =np.array(p1)
    p2 =np.array(p2)
    i1 = np.reshape(i1, (3,3,))
    i2 = np.reshape(i2, (3,3,))
    i1 = InertialTensor(data=i1)
    i2 = InertialTensor(data=i2)
    ix = InertialTensor(data = i2.data - i1.shift_inertial_tensor(p2-p1, m1).data)
    iy = ix.shift_inertial_tensor(-p2, -(m2-m1))
    print('result >> ', iy)
    
def an3(p):
    df1 = data.phys_measures[p:p+2]
    an2(df1)
an3(2)
an3(3)
an3(5)



# In[7]:


a = TriangleActorBase()
a.add_meshio(path='/tmp/res.stl')
a.build()
oplt.plot(a.vispy_data.update(conf=A(mode='3D')))


# In[85]:


@dataclass
class TankStatus:
    volume: float
    density: float

@dataclass
class ThrusterStatus:
    thrust: float
    rot: R
        
@dataclass
class ComposedObjectLogic(ObjectLogic):
    id2logic: dict = None
    def build(self):
        self.obj = RigidBody.Compose([x.obj for x in self.id2logic.values()])
    def _register(self):
        for x in self.id2logic.values(): x.register(self.env)
    
            
            
            
@dataclass
class EntitySpec:
    instance_count: int = 0
    @property
    def name(self): return type(self).__name__
    
    
    def get_new_id(self):
        id = self.instance_count
        self.instance_count += 1
        return f'{self.name}_{id}'
        
    def build_logic(self): 
        res = self.get_cl()(**self.get_args())
        res.spec = self
        res.build()
        return res
    def get_cl(self): assert 0
    def get_args(self): return {}
    
@dataclass
class ObjEntitySpec(EntitySpec):
    solid_spec: SolidSpec = None
    def get_cl(self):
        class ObjEntityLogic(ObjectLogic):
            def _build(self_tl):
                self_tl.obj = RigidBody(spec=self.solid_spec)
                super()._build()
        return ObjEntityLogic
        
    
@dataclass
class EntitySpecChild:
    spec : EntitySpec
    tsf: Transform
        
    
    
@dataclass
class ComposeSpec(EntitySpec):
    instances: list[EntitySpecChild] = None
        
    def build_id2logic(self):
        res = {}
        for x in self.instances:
            lx = x.spec.build_logic()
            lx.obj.local2world = x.tsf
            res[lx.id] = lx
        return res
        
    def get_cl(self): return ComposedObjectLogic
    def get_args(self):
        a = dict(id2logic=self.build_id2logic())
        a.update(super().get_args())
        return a
        

@dataclass
class ThrusterSpec(ObjEntitySpec):
    force_pos: Vec3 = None
    force_direction: Vec3 = None
    gimbal_max_ang: float = None
    thrust_provider: float = None
    
        
    def get_cl(self):
        class ThrusterLogic(super().get_cl()):
                
            def _update(self_tl, status: ThrusterStatus, control):
                dir = status.rot * self.force_direction
                self_tl.env.update_force(self_tl, ForceEntry(pos=self.force_pos, f=dir * status.thrust))
            def _register(self_tl):
                self_tl.env.add_to_udpates(self_tl)
                                        
            
        return ThrusterLogic
    
    
thruster_simple = ThrusterSpec(force_pos=Vec3, force_direction=-Vec3.Z(), gimbal_max_ang=np.deg2rad(2), 
                            thrust_provider=lambda throttle: throttle * 1000,
                            solid_spec = SolidSpec.Box(10, 1, 1, 1))


thruster_instances = []
def circle_pos(ang, r):
    return Vec3([np.cos(ang), np.sin(ang), 0]) * r
for i in range(4):
    thruster_instances.append(EntitySpecChild(spec=thruster_simple,  tsf=Transform.From(pos=circle_pos(np.pi*2/4*i, 5))))
thrusters_desc = ComposeSpec(instances=thruster_instances)
bl = thrusters_desc.build_logic()

@dataclass
class RocketSpec:
    thrust_curve= None 
    prop_density: float
    thrust_specs: list
        
        
    def build_logic(self):
        class RocketLogic(ComposedObjectLogic):
            def _build(self_tl):
                self.fill_id2logic(self_tl.id2logic)
                super()._build()
    


# In[94]:


pts, weights = bl.obj.spec.desc.importance_sampling(10000)


# In[90]:


tg = r3
if 0:
    pts = tg.spec.desc.rejection_sampling(1000000)
    weights=None
else:
    pts,weights = tg.spec.desc.importance_sampling(10000000)
ix = InertialTensor.FromPoints(pts, weights=weights, com=False, mass=tg.spec.mass)
ix, tg.world_inertial_tensor


# In[95]:


#pts,weights = tg.spec.desc.importance_sampling(100000)
viridis = K.vispy_utils.get_colormap('viridis')
cols = viridis.map(weights/max(weights))
oplt.plot(A(points=pts, conf=A(mode='3D'), points_color=cols))


# In[80]:


r1 = RigidBody(SolidSpec.Sphere(1, 1), local2world=Transform.From(), worldvel=[0,0,0])
pts = r1.spec.desc.rejection_sampling(10000)
pts,weights = r1.spec.desc.importance_sampling(100000)
ix = InertialTensor.FromPoints(pts, weights=weights)
ix


# In[152]:


pts = [
    [0,0,0],
    [2,0,0],
    [2,0,0],
]
pts = np.array(pts)
ix = InertialTensor.FromPoints(pts, weights=np.array([1,1,1]))
ix.around([0,1,0]), ix


# In[116]:


r1 = RigidBody(SolidSpec.Point(1), local2world=Transform.From(pos=[0,0,0]), worldvel=[0,0,0])
r2 = RigidBody(SolidSpec.Point(2), local2world=Transform.From(pos=[2,0,0]), worldvel=[0,0,0])
rx = RigidBody.Compose([r1,r2])
rx.spec.I_0.around([0,1,0])


# In[129]:


rx.spec


# In[117]:


rx.spec.I_0


# In[93]:


np.einsum('i,jk->jk', [1,2,3], np.identity(3))


# In[14]:


@dataclass
class SimulDataEntry:
    by_t: list = field(default_factory=list)
    tlist: list = field(default_factory=list)
    lim :int = 10
    @property
    def last(self): return self.by_t[-1]
        
    def add(self, tid, v):
        self.tlist.append(tid)
        self.by_t.append(v)
    @property
    def as_xarray(self):
        return xr.DataArray(self.by_t)
        
@dataclass
class SimulEnv:
    ctrl: A = field(default_factory=A)
    consts: A = field(default_factory=A)
    data: A = field(default_factory=lambda: A(other=lambda _: SimulDataEntry()))
    
class Block:
    def process_impl(self, env: SimulEnv):
        assert 0
    
    def process(self, env : SimulEnv):
        self.process_impl(env)
        pass
@dataclass
class BlockFunc:
    func: Callable[[SimulEnv], None]
    def process_impl(self, env):
        self.func(env)
        
@dataclass
class ForwardBlock(Block):
    block: Block
    def cond(self, env):
        assert 0
    def process_impl(self, env):
        if self.cond(env):
            self.block.process(env)
        
@dataclass
class AggBlock(Block):
    blocks: List[Block]
    def process_impl(self, env):
        for x in self.blocks:
            x.process(env)
            
@dataclass
class SkipBlock(ForwardBlock):
    every: int
    current: int = 0
    def cond(self, env):
        self.current += 1
        if self.current == self.every:
            self.current = 0
            return True
        return False
        

@dataclass
class Simulator:
    block: Block
    env: SimulEnv = field(default_factory=SimulEnv)
        
    def tick(self):
        self.block.process(self.env)
        
class RocketPhys(Block):
    def process_impl(self, env:  SimulEnv):
        env.data.phys_data.last
        pass
    
class RocketSensors(Block):
    def process_impl(self, env):
        pass
    
class RocketControls(Block):
    def process_impl(self, env):
        pass

        
consts = A(mass0=df_consts.mass0[0], diameter=df_consts.diameter[0], thrust_weight=thrust_w)
        
phys = RocketPhys()
sensors = RocketSensors()
ctrls = RocketControls()
phys = SkipBlock(every=1, block=phys)
ctrls = SkipBlock(every=10, block=ctrls)
main_block = AggBlock(blocks=[phys, sensors, ctrls])
sx = Simulator(main_block)


s0 = A(pos=np.array([0,0,0]), v=np.array([0,0,0]), rot=R.identity(), inertialm0=1000 )
ctrl0 = A(thrust=1, thrust_vector=np.array([0,0]))
sx.env.data.phys_data.add(0, s0) 
sx.env.data.phys_pos.add(0, [0, 0, 0])

sx.tick()
    
    

