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


# In[ ]:


m0 = MoveDesc(worldvel=np.array([0,0,0]))
r0 = RigidBody(
    spec=SolidSpec.Box(1, 1, 1, 1),
    local2world=Transform.From(),
    move_desc=m0)
r0.local2world.pos += [0,0,3]

r1 = RigidBody(
    spec=SolidSpec.Sphere(1, 1),
    local2world=Transform.From(pos=[0, 0, -3]),
    move_desc=m0)
r2 = RigidBody.Compose([r1,r0])

r00 = RigidBody(
    spec=SolidSpec.Box(1, 1, 1, 1),
    local2world=Transform.From(),
    move_desc=m0)
r00.local2world.pos += [0,0,3]

r11 = RigidBody(
    spec=SolidSpec.Sphere(1, 1),
    local2world=Transform.From(pos=[0, 0, -3]),
    move_desc=m0)
r22 = RigidBody.Compose([r11,r00])
r22.local2world.pos += [100,0, 0]

r3 = RigidBody.Compose([r2, r22])





# In[ ]:


a = TriangleActorBase()
a.add_meshio(path='/tmp/res.stl')
a.build()
oplt.plot(a.vispy_data.update(conf=A(mode='3D')))


# In[ ]:


from vispy.geometry.meshdata import MeshData
a = AABB([-0.5,-0.5,-0.5], [1,1,1]).surface_mesh
oplt.plot(a.vispy_data.update(conf=A(mode='3D')))
    
a = SphereDesc()
b = a.rejection_sampling(100)
c = a.importance_sampling(100)
TransformedMeshDesc(mesh=CubeDesc(), transform=Transform.From())
a = SphereDesc()
oplt.plot(a.surface_mesh().vispy_data.update(conf=A(mode='3D')))
a.surface_mesh().trs


# In[ ]:


m0 = MoveDesc(worldvel=[0,0,0])
r0 = RigidBody(
    SolidSpec.Box(2, 1, 10, 5),
    local2world=Transform.From(rot=R.from_euler('xy', [np.pi / 3, np.pi / 5])),
    move_desc=m0)
r1 = RigidBody(SolidSpec.Box(1, 1, 1, 1),
               local2world=Transform.From(pos=[-10, 0, 0]),
               move_desc=m0)
r2 = RigidBody(SolidSpec.Sphere(1, 1),
               local2world=Transform.From(pos=[1, 0, 0]),
               move_desc=m0)
r3 = RigidBody.Compose([r1, r0, r2])
r3.spec, r1.spec, r2.spec
for x in [r1, r2]:
    print(x.world_inertial_tensor)

