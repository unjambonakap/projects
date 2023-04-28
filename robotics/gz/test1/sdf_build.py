#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
from __future__ import annotations
import sys
sys.path.append('/usr/lib/python/')
import sdformat13 as sdf


# In[11]:


s = scenes.box_scene()
rbl = s.sctx.roots[0].self_link
cx = SDFConverter('./traj.sdf')
cx.reset_world(cx.rbl_to_model(rbl))
cx.write('./traj_new.sdf')


# In[ ]:


planeId = p.loadSDF("/usr/share/gz/gz-sim7/worlds/dem_moon.sdf")


# In[ ]:


p.disconnect()

