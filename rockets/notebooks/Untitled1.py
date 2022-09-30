#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from dataclasses import dataclass, field
from collections import deque
import xarray as xr
from typing import Callable, List
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase

df_consts = pd.read_csv('../data/consts.csv')
thrust_w = pd.read_csv('../data/ds_thrust_and_weight.csv')


# In[2]:


class Vec3:
    data: np.ndarray
    @staticmethod
    def Natif(v):
        if isinstance(v, Vec3): return v.data
        return v
    
    @staticmethod
    def Make(v):
        if v is None: return v
        if isinstance(v, Vec3): return v
        return Vec3(v)
    
    @staticmethod
    def Zero(): return Vec3(0,0,0)
    @staticmethod
    def Z(): return Vec3(0,0,1)
    @staticmethod
    def X(): return Vec3(1,0,0)
    @staticmethod
    def Y(): return Vec3(0,1,0)
    def __neg__(a): return Vec3(-a.data)
    def __add__(a, b): return Vec3(a.data + Vec3.Natif(b))
    def __mul__(a, b): return Vec3(a.data * Vec3.Natif(b))
    def __sub__(a, b): return Vec3(a.data - Vec3.Natif(b))
    def __div__(a, b): return Vec3(a.data / Vec3.Natif(b))
        
        
    def __init__(self, *args, data=None):
        if data is None:
            if len(args) == 1:
                data = args[0]
                if not cmisc.is_list(data): data = [data, data, data]
            elif len(args) == 0:
                data = [0,0,0]
            else:
                data = args
                
        self.data = np.array(data)
            
    
@dataclass
class Transform:
    data: np.ndarray
    @property
    def inv(self): return np.linalg.inv(self.data)
    @property
    def pos(self): return self.data[:3,3]
    @pos.setter
    def pos(self, v): 
        self.data[:3,3] = v
    @property
    def rot(self): return self.data[:3,:3]
    @rot.setter
    def rot(self, v): self.data[:3,:3] = v
        
    def map(self, v):
        if len(v.shape)==2:
            return Z.MatHelper.mat_apply_nd(self.data, v.T, point=1).T
        return Z.MatHelper.mat_apply_nd(self.data, v, point=1)
    def imap(self, v):
        return Z.MatHelper.mat_apply_nd(self.inv, v, point=1)
    
    def __mul__(self, peer):
        return Transform(data=self.data @ peer.data)
    
    @staticmethod
    def From(pos=None, rot=None, scale=None):
        res = np.identity(4)
        tsf = Transform(data=res)
        pos = Vec3.Make(pos)
        if scale is not None: 
            res[:3,:3] *= scale
            scale = None
        elif pos is not None: 
            res[:3,3] = pos.data
            pos = None
        
        elif rot is not None: 
            if isinstance(rot, R):
                rot = rot.as_matrix()
            res[:3,:3] = rot
            rot = None
        else:
            return tsf
            
        return tsf * Transform.From(pos=pos, rot=rot, scale=scale)
    
class AABB:

  @staticmethod
  def FromPoints(*pts):
    min = np.min(pts, axis=0)
    max = np.min(pts, axis=1)
    return AABB(min, max-min)

  def __init__(self, p, v):
    self.p = np.array(p)
    self.n = len(p)
    self.v = np.array(v)
    
  def corner(self, m):
    v= np.array(self.v)
    for i in range(self.n):
        v[i] *= (m>>i&1)
    return self.p + v

  @property
  def surface_mesh(self):
    coords = [self.corner(i) for i in range(8)]
    res = TriangleActorBase()
    ids = res.add_points(coords)
    res.push_quad(ids[:4], pow2=1)
    res.push_quad(ids[4:8], pow2=1)
    adjs = [0,1,3,2,0]
    for i in range(4):
        a = adjs[i]
        b= adjs[i+1]
        
        res.push_quad(ids[[a,b,a+4,b+4]], pow2=1)
    res.build()
    return res

    

  @property
  def low(self): return self.p
  @property
  def high(self): return self.p + self.v


# In[4]:


from vispy.geometry.meshdata import MeshData
a = AABB([-0.5,-0.5,-0.5], [1,1,1]).surface_mesh
oplt.plot(a.vispy_data.update(conf=A(mode='3D')))


# In[33]:


def rejection_sampling(n, bounds, checker, grp_size=1000):
    res = []
    while len(res)<n:
        cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
        cnds = cnds[checker(cnds)]
        res.extend(cnds)
        
    return np.array(res[:n])


def loop(x):
    ix = iter(x)
    a = next(ix)
    yield a
    yield from ix
    yield a

@dataclass
class SurfaceMeshParams:
    points: int = None
    
    

        
def importance_sampling(n, bounds, weight_func, map=None):
    cnds = np.random.uniform(low=bounds[0], high=bounds[1], size=(n, len(bounds[0])))
    weights = weight_func(cnds)
    weights= weights / sum(weights)
    if map is not None: cnds = map(cnds)
    return cnds, weights

class MeshDesc:
    @property
    def bounds(self) -> AABB: pass
    def rejection_sampling(self, n):
        return self.rejection_sampling0(n)
    def importance_sampling(self, n, wtarget=1):
        pts, w= self.importance_sampling0(n)
        w *= wtarget
        return pts, w
    def rejection_sampling0(self, n):
        pass
    def importance_sampling0(self, n):
        pass
    def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
        assert 0
    
@dataclass
class TransformedMeshDesc(MeshDesc):
    mesh: MeshDesc
    transform: Transform
    def rejection_sampling0(self, n):
        return self.transform.map(self.mesh.rejection_sampling0(n))
    def importance_sampling0(self, n):
        pts, w = self.mesh.importance_sampling0(n)
        return self.transform.map(pts), w
    def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
        assert 0
        
@dataclass
class ComposedMeshDesc(MeshDesc):
    meshes: list
    weights: list
        
    @property
    def entries(self): return [A(w=w, m=m) for m, w in zip(self.meshes, self.weights)]
    def rejection_sampling0(self, n):
        return np.array(list(itertools.chain(*[x.m.rejection_sampling0(int(n*x.w)) for x in self.entries])))
    def importance_sampling0(self, n):
        return [np.array(list(itertools.chain(*x))) for x in zip(*[y.m.importance_sampling(int(n*y.w), wtarget=y.w) for y in self.entries])]
    
class CubeDesc(MeshDesc):
    
    @property
    def bounds(self) -> AABB:  return AABB([-0.5,-0.5,-0.5], [1,1,1])
    
    def rejection_sampling0(self, n):
        return rejection_sampling(n, (self.bounds.low, self.bounds.high), lambda tb: np.full(tb.shape[0], True))
    def importance_sampling0(self, n):
        return importance_sampling(n, (self.bounds.low, self.bounds.high), weight_func=lambda tb: np.full(tb.shape[0], 1))
    
    def surface_mesh(self, params: SurfaceMeshParams) -> TriangleActorBase:
        return self.bounds.surface_mesh
    
class SphereDesc(MeshDesc):
    @property
    def bounds(self) -> AABB:  return AABB([-1,-1,-1], [2,2,2])
    
    @staticmethod
    def spherical2xyz_unpacked(r, alpha, phi):
        x = r*np.cos(alpha) * np.cos(phi)
        y = r*np.sin(alpha) * np.cos(phi)
        z = r*np.sin(phi) * np.ones_like(alpha)
        return np.array([x,y,z]).T
    @staticmethod
    def spherical2xyz(sph):
        r =  sph[:,0]
        alpha =  sph[:,1]
        phi =  sph[:,2]
        return SphereDesc.spherical2xyz_unpacked(r, alpha, phi)
        
    @staticmethod
    def Weight(sph):
        r, alpha, phi = sph.T
        det_g_sqrt = r*r * np.abs(np.sin(alpha))
        det_g_sqrt = r*r * np.abs(np.cos(phi))
        return det_g_sqrt
    
    @property
    def bounds_spherical(self):
        return ((0,0,-np.pi/2), (1,2*np.pi,np.pi/2))
    
    def rejection_sampling0(self, n):
        return rejection_sampling(n,(self.bounds.low, self.bounds.high), lambda pts: np.einsum('ij,ij->i', pts, pts) <= 1) 
    
    def importance_sampling0(self, n):
        return importance_sampling(n,self.bounds_spherical, weight_func=SphereDesc.Weight, map=SphereDesc.spherical2xyz)
    
    def surface_mesh(self, params: SurfaceMeshParams=None) -> TriangleActorBase:
        res = TriangleActorBase()
        alphal = np.array(list(loop(np.linspace(0, np.pi*2, 20))))
        for phi_a,phi_b in itertools.pairwise(np.linspace(-np.pi/2, np.pi/2, 10)):
            la = SphereDesc.spherical2xyz_unpacked(1, alphal, phi_a)
            lb = SphereDesc.spherical2xyz_unpacked(1, alphal, phi_b)
            ids_a = res.add_points(la)
            ids_b = res.add_points(lb)
            for i in range(len(alphal)-1):
                res.push_quad(np.array([ids_a[i], ids_a[i+1], ids_b[i], ids_b[i+1]]), pow2=1)
        res.build()
        return res
    
a = SphereDesc()
b = a.rejection_sampling(100)
c = a.importance_sampling(100)


# In[32]:


a = SphereDesc()
oplt.plot(a.surface_mesh().vispy_data.update(conf=A(mode='3D')))


# In[22]:


a.surface_mesh().trs


# In[ ]:





# In[84]:


@dataclass
class InertialTensor:
    data: np.ndarray = field(default_factory= lambda: np.zeros((3,3)))
        
    def shift_inertial_tensor(self, p, mass):
        res = np.array(self.data)
        px,py,pz=p
        mx = [
            [py*py+pz*pz, -px*py, -px*pz],
            [-px*py, px*px+pz*pz, -py*pz],
            [-px*pz, -py*pz, py*py+px*px],
        ]
              
        res += np.array(mx)*mass
        return InertialTensor(data=res)
    
    def get_world_tensor(self, local2world_rot):
        m = local2world_rot @ self.data @ local2world_rot.T
        return InertialTensor(data=m)
    
    def __add__(self, v): return InertialTensor(data=self.data + v.data)
    def around(self, v): return v @ self.data @ v
    
    @staticmethod
    def FromPoints(pts, weights=None, com=False, mass=1):
        r =pts
        if weights is None: weights = np.full(len(pts),1)
        sweight = sum(weights)
        weights = weights * (mass /sweight)
        if com:
            r = pts - np.average(pts, weights=weights, axis=0)
        r2 = np.sum(np.linalg.norm(r,axis=1)**2 * weights)
        e3r2 = r2 * np.identity(3)
        res = e3r2 - np.einsum('i,ij,ik->jk', weights, r, r)
        return InertialTensor(data=res)
    
   
@dataclass
class SolidSpec:
    mass: float
    I_0: InertialTensor
    desc: MeshDesc = None
    
    @staticmethod
    def Point(mass):
        return SolidSpec(mass=mass, I_0=InertialTensor(data=np.zeros((3,3))))
    
    @staticmethod
    def Sphere(mass, r):
        v = mass*2/5 * r**2
        desc = TransformedMeshDesc(mesh=SphereDesc(), transform=Transform.From(scale=r))
        return SolidSpec(mass=mass, I_0=InertialTensor(data=np.identity(3) * v), desc=desc)
    
    @staticmethod
    def Cylinder(mass, r, h):
        iz = 1/2*mass*r**2
        ix = iz/2 + M * h**2/12
        return SolidSpec(mass=mass, I_0=InertialTensor(data=np.diag([ix,ix,iz])))
        
    @staticmethod
    def Box(mass, x, y, z):
        c = mass / 12
        desc = TransformedMeshDesc(mesh=CubeDesc(), transform=Transform.From(scale=[x,y,z]))
        return SolidSpec(mass=mass, I_0=InertialTensor(data=np.diag([y*y+z*z, x*x+z*z, x*x+y*y])*c), desc=desc)
    
    
@dataclass
class RigidOrientation:
    local2world: np.ndarray = None

        
@dataclass
class MoveDesc:
    worldvel: np.ndarray = None
    local_rotang: np.ndarray = None
    rotspeed: float = None
    
@dataclass
class RigidBody:
    spec: SolidSpec = None
    local2world: Transform = None
    move_desc : MoveDesc = None
    children: list = field(default_factory=list)
    parent = None
        
    @property
    def world_inertial_tensor(self):
        return self.spec.I_0.get_world_tensor(self.local2world.rot).shift_inertial_tensor(-self.local2world.pos, self.spec.mass)
        
    @staticmethod
    def Compose(children):
        mass = sum([x.spec.mass for x in children])
        com = np.sum([x.local2world.pos * x.spec.mass for x in children], axis=0) / mass
        print('COM >> ', com)
        tsf = Transform.From(pos=com)
        
        res=RigidBody(children=children, local2world=Transform.From())
        descs = []
        weights = []
        for x in children:
            x.local2world.pos -= com
            descs.append(TransformedMeshDesc(mesh=x.spec.desc, transform=x.local2world))
            weights.append(x.spec.mass / mass)
            x.parent = res
                         
        its = sum([x.world_inertial_tensor for x in children], InertialTensor())
        res.spec = SolidSpec(mass=mass, I_0 = its, desc=ComposedMeshDesc(meshes=descs, weights=weights))
        
        return res
    
    
r0 = RigidBody(SolidSpec.Box(2, 1, 10, 5), local2world=Transform.From(rot=R.from_euler('xy', [np.pi/3, np.pi/5])), worldvel=[0,0,0])
r1 = RigidBody(SolidSpec.Box(1, 1, 1, 1), local2world=Transform.From(pos=[-10,0,0]), worldvel=[0,0,0])
r2 = RigidBody(SolidSpec.Sphere(1, 1), local2world=Transform.From(pos=[1,0, 0]), worldvel=[0,0,0])
r3 = RigidBody.Compose([r1, r0, r2])
r3.spec, r1.spec, r2.spec
for x in [r1,r2]:
    print(x.world_inertial_tensor)


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
class ForceEntry:
    pos: Vec3
    f : Vec3
    apply_object: ObjectLogic
    tid: int
    
@dataclass
class SimulEnv:
    id2force: dict
    to_update : list = field(default_factory= list)
        
    def update_force(self, logic, fe):
        fe.apply_object=  logic
        id2force[logic] = fe
        
    def add_to_updates(self, logic):
        self.to_update.append(logic)
    
@dataclass
class ObjectLogic:
    obj: RigidBody = None
    env: SimulEnv = None
    spec: None = None
    id: None = None
        
    def register(self, env):
        self.env = env
        self._register()
        
    def build(self): 
        self.id= self.spec.get_new_id()
        self._build()
    def _build(self): pass
    def _register(self): pass 
    def _update(self, status, control): pass
    
    def update(self):
        status = self.env.get_status(self)
        ctrl = self.env.get_status(self)
        self._update(status, ctrl)
    
    
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
    
@dataclass
class ThrusterControl:
    rotation: R
    throttle: float 
    thrust: float
    


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
    
    

