#!/usr/bin/env python
# coding: utf-8

# In[1]:


#init_jupyter()
import sys
sys.path.append('/usr/lib/python/')
from gz.sim7 import TestFixture, World, world_entity
import gz.sim7 as sim7

import os
os.environ['GZ_DEBUG_COMPONENT_FACTORY'] = 'true'

from gz.common import set_verbosity
from gz.sim7 import TestFixture, World, world_entity
from gz.math7 import Vector3d

u = Vector3d()

assert 0
set_verbosity(4)


input_file = "/usr/share/gz/gz-sim7/worlds/dem_moon.sdf"
input_file = "./dem_moon.sdf"
fixture = TestFixture(input_file)

post_iterations = 0
iterations = 0
pre_iterations = 0
first_iteration = True
ecm= None

def on_pre_udpate_cb(_info, _ecm):
    global pre_iterations
    global first_iteration
    global ecm
    ecm = _ecm
    pre_iterations += 1
    if first_iteration:
        first_iteration = False
        world_e = world_entity(_ecm)
        print('World entity is ', world_e)
        w = World(world_e)
        v = w.gravity(_ecm)
        print('Gravity ', v)
        modelEntity = w.model_by_name(_ecm, 'falling')
        print('Entity for falling model is: ', modelEntity)


def on_udpate_cb(_info, _ecm):
    global iterations
    iterations += 1


def on_post_udpate_cb(_info, _ecm):
    global post_iterations
    post_iterations += 1
    #if _info.sim_time.seconds == 1:
    print('Post update sim time: ', _info.sim_time)


fixture.on_post_update(on_post_udpate_cb)
fixture.on_update(on_udpate_cb)
fixture.on_pre_update(on_pre_udpate_cb)
fixture.finalize()


# In[2]:


server = fixture.server()
#server.run(True, 1, False)

mg = server.entity_comp_mgr(0)


world_e = world_entity(mg)
w = World(world_e)
mid = w.model_by_name(mg, 'box')
model = sim7.Model(mid)
lid = model.link(mg)

mg.component_types(mid)
y = sim7.LinearVelocity.GetOrCreate(mg, mid).data()
print(y)
breakpoint()
print(list(y))
assert 0

# In[5]:


server.run(True, 100, False)


# In[3]:




# In[4]:


print(sim7.Pose.GetOrCreate(mg, mid).data())
print(sim7.Pose.GetOrCreate(mg, mid).data())

server.run(True, 100, False)

print(sim7.Pose.GetOrCreate(mg, mid).data())
