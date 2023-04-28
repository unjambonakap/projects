#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter(run_app=True)
from chdrft.sim.base import *
from chdrft.sim.rb.blender_helper import *
from chdrft.sim.blender import *

clear_scene()
helper = BlenderPhysHelper()


# In[2]:


from chdrft.geo.satsim import gl

p = gl.geocode('Paris')
ll = Z.deg2rad(np.array([p.longitude, p.latitude]))
md = 13
u = create_earth_actors(
  BlenderTriangleActor,
  max_depth=md,
  tile_depth=md,
  m2u=1e-3,
  ll_box=Box(center=ll, size=(np.pi / 1000, np.pi / 1000))
)


a = actors_to_obj('x0', helper.main_col, u.actors)
helper.set_cam_focus(u.points, Vec3.X() * 3, expand=1.2)


# In[ ]:


ll


# In[3]:


helper.get_aabb(a).points


# In[5]:


a.

