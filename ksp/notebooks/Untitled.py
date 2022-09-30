#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
import krpc


# In[2]:


conn = krpc.connect(
    name='My Example Program',
    address='localhost',
    rpc_port=50000, stream_port=50001)
print(conn.krpc.get_status().version)


# In[5]:


conn.nop(req_phys_loop=True)


# In[3]:



conn.space_center.launch_vessel2('VAB', "./Ships/VAB/SpaceX Falcon 9 Block 5.craft", 'LaunchPad', True)


# In[2]:


import krpc.schema.KRPC_pb2 as KRPC
x = KRPC.Request()
x.lock_update = False
conn.nop()


# In[4]:


cam = conn.space_center.camera


# In[9]:


import time
time.sleep(1)
cam.pitch = 30


# In[23]:


import time
for i in range(100):
    conn.lock_update = False
    conn.wait_req_id = i+1
    conn.req_id = i
    conn.nop()
    time.sleep(0.5)
    conn.wait_req_id = i+1
    conn.req_id = i+1
    conn.nop()
    time.sleep(0.5)
#vessel = conn.space_center.active_vessel
#flight_info = vessel.flight()


# In[39]:


conn.space_center.physics_warp_factor = 1
conn.space_center.time_warp_helper.set_physical_warp_rate(0, 0.3)
conn.space_center.time_warp_helper.set_warp_rate(0, 1)
conn.space_center.physics_warp_factor = 0


# In[3]:





# In[10]:


vessel = conn.space_center.active_vessel
flight_info = vessel.flight()
earth = conn.space_center.bodies['Kerbin']
flight = vessel.flight(earth.reference_frame)


# In[4]:


a=  earth.reference_frame


# In[5]:


a.direction_to_world_space([1,1,0])


# In[69]:


alt=earth.altitude_at_position(flight.center_of_mass, earth.reference_frame)
lat=earth.latitude_at_position(flight.center_of_mass, earth.reference_frame)
lon=earth.longitude_at_position(flight.center_of_mass, earth.reference_frame)
alt,lat,lon


# In[ ]:


flight.world_relative_velocity(earth, flight.center_of_mass, flight.velocity)


# In[8]:


flight.simulate_aerodynamic_force_at(earth, flight.center_of_mass, flight.velocity)


# In[19]:


np.linalg.norm(conn.space_center.aerodynamics.sim_aero_force_by_alt(earth, vessel, [0,100,0], 1000))


# In[58]:


px = earth.surface_position(lat, lon, earth.reference_frame)
earth.pos_to_body_velocity(px, earth.reference_frame)


# In[66]:


flight.velocity


# In[55]:


vessel.bounding_box(earth.reference_frame)


# In[81]:


flight.simulate_aerodynamic_force_at(earth,flight.center_of_mass,flight.velocity)


# In[ ]:


earth.reference_frame.


# In[42]:


vessel.bounding_box(vessel.reference_frame)
vessel.inertia_tensor


# In[27]:


for x in vessel.active_engines:
    print(x.available_torque)


# In[141]:


rscs = set()
for engine in vessel.active_engines:
    for prop in engine.propellants:
        for rsc in vessel.resources.with_resource_by_id(prop.id):
            rscs.add(rsc)
import time
def set_prop(v, target_name=None):
    for rsc in rscs:
        if target_name is None or target_name == rsc.part.name:
            rsc.amount = rsc.max * v
xl = np.linspace(0, 1, num=2)


# In[190]:


tensors = []
masses = []
poslist = []
for lvl in xl:
    set_prop(lvl)
    time.sleep(0.1)
    tensors.append(vessel.inertia_tensor)
    masses.append(vessel.mass)
    poslist.append(vessel.position(earth.reference_frame))
    
    for engine in vessel.active_engines:
        print(engine.thrusters, engine.max_thrust, engine.gimballed)
        for th in engine.thrusters:
            if th.gimballed:
                print(th.thrust_position(vessel.reference_frame))
                print(th.thrust_direction(vessel.reference_frame))
            pass
        break


# In[188]:


poslist


# In[104]:


oplt.plot(K.Dataset(tx[:,0]))


# In[126]:


tx = np.array(tensors)
p = np.poly1d(np.polyfit(xl, tx[:,0], 3))
px = oplt.plot(K.Dataset(tx[:,0]), o=1)


# In[127]:


px.w.add_plot(K.Dataset(p(xl)))
px.w.add_plot(K.Dataset(tx[:,0]))


# In[134]:


e0 = vessel.active_engines[0]
e0.gimbal_range
e0.available_torque


# In[138]:


t0 = e0.thrusters[0]
t0.gimballed


# In[144]:


e0.thrusters[0].gimballed
e0.gimballed


# In[178]:


vessel.control.pitch = -1
vessel.control.yaw =1 


# In[179]:


for engine in vessel.active_engines:
    print(engine.thrusters, engine.max_thrust, engine.gimballed)
    for th in engine.thrusters:
        if th.gimballed:
            print(th.thrust_position(vessel.reference_frame))
            print(th.thrust_direction(vessel.reference_frame))
        pass


# In[120]:


for engine in vessel.active_engines:
    for prop in engine.propellants:
        print()
        print(prop.name, prop.total_resource_available, prop.current_amount)
        for rsc in vessel.resources.with_resource_by_id(prop.id):
            rsc.amount = rsc.max


# In[53]:


for engine in vessel.active_engines:
    for prop in engine.propellants:
        print()
        print(prop.name, prop.total_resource_available, prop.current_amount)
        for rsc in vessel.resources.with_resource_by_id(prop.id):
            print(rsc.amount)


# In[29]:


for engine in vessel.active_engines:
    rsc = vessel.resources.with_resource_by_id(x.propellants[0].id)
    print(rsc)
    


# In[10]:


vessel.inertia_tensor


# In[113]:


vessel = conn.space_center.active_vessel
flight_info = vessel.flight()
earth = conn.space_center.bodies['Kerbin']
flight = vessel.flight(earth.reference_frame)


# In[7]:


cheats = conn.space_center.cheats
cheats.unbreakable_joints=True
cheats.no_crash_damage=True
cheats.ignore_max_temperature=True


# In[11]:


vessel.control.sas = False
vessel.control.rcs = False
vessel.control.throttle = 1.0
vessel.control.activate_next_stage()
vessel.control.activate_next_stage()


# In[33]:


vessel.control.throttle=1


# In[119]:


vpos = vessel.position(earth.reference_frame)
flight.set_position(tuple(np.array(vpos)*1.5))
conn.nop(req_phys_loop=1)
flight.set_position(tuple(np.array(vpos)*1.9))
conn.nop(req_phys_loop=1)
flight.set_position(tuple(np.array(vpos)*3.9))
conn.nop(req_phys_loop=1)


# In[66]:


cam = conn.space_center.camera
cam.mode = conn.space_center.CameraMode.locked
cam.yaw = 90
cam.pitch = 0


# In[32]:


set_prop(1)


# In[156]:


tank = [x for x in vessel.parts.all if 'S1tank' in x.name ][0]
t0 = [x for x in vessel.parts.all if 'Merlin' in x.name ][0]


# In[105]:


set_prop(0)
conn.nop(req_phys_loop=1)
print(t0.center_of_mass(vessel.reference_frame), tank.mass-tankmass)
set_prop(0.5)
conn.nop(req_phys_loop=1)
print(t0.center_of_mass(vessel.reference_frame), tank.mass-tankmass)
set_prop(1)
conn.nop(req_phys_loop=1)
print(t0.center_of_mass(vessel.reference_frame), tank.mass-tankmass, v0_mass)


# In[140]:


vessel.active_engines[0].part.name


# In[144]:


tank.massless


# In[164]:


np.linalg.norm(t0.center_of_mass(tank.reference_frame))


# In[166]:


tank.name


# In[167]:


d0


# In[162]:


def an1(x):
    set_prop(x, target_name=tank.name)
    conn.nop(req_phys_loop=1)
    m1 = tank.mass -tankmass
    print(m1)
    dist = np.linalg.norm(t0.center_of_mass(vessel.reference_frame))-d0
    print(t0.center_of_mass(vessel.reference_frame))
    return dist * (m1 + v0_mass) / m1
[an1(x) for x in np.linspace(0.001, 1, 10)] 


# In[157]:



set_prop(0)
conn.nop(req_phys_loop=1)
print(tank.center_of_mass(vessel.reference_frame))
set_prop(1)
conn.nop(req_phys_loop=1)
print(tank.center_of_mass(vessel.reference_frame))


# In[169]:


set_prop(1)
conn.nop(req_phys_loop=1)
mass_prop = tank.mass-tankmass

set_prop(0, target_name=tank.name)
conn.nop(req_phys_loop=1)
d0= np.linalg.norm(t0.center_of_mass(vessel.reference_frame))
tankmass = tank.mass
v0_mass = vessel.mass
print(np.linalg.norm(tank.center_of_mass(vessel.reference_frame)))


# In[160]:


for i, p in enumerate(vessel.parts.all):
    set_prop(1, target_name=tank.name)
    conn.nop(req_phys_loop=1)
    print(i, p.name, p.mass)
    set_prop(0, target_name=tank.name)
    conn.nop(req_phys_loop=1)
    print(i, p.name, p.mass)


# In[ ]:





# In[58]:


cam.transform.rotation = [0,0,0,1]


# In[55]:


from scipy.spatial.transform import Rotation as R
R.identity().as_quat()


# In[62]:


cam.pitch, cam.yaw


# In[57]:


e0  = vessel.active_engines[0]


# In[40]:


import time
for i in np.linspace(-1, 1, 2):
    for j in np.linspace(-1, 1, 5):
        time.sleep(1)
        e0.gimbal.set_gimbal_rot(e0.gimbal.actuation2_rot([i,j]))


# In[ ]:



e0.gimbal.set_gimbal_rot(e0.gimbal.actuation2_rot([i,j]))


# In[29]:


e0.gimbal.enable_gimbal = False


# In[60]:





# In[61]:


cam.set_target_vessel(vessel)


# In[63]:


cam.distance = 100


# In[49]:


cam.max_pitch


# In[50]:


cam.pitch, cam.yaw


# In[58]:





# In[59]:


cam.set_target_part(e0.part)


# In[33]:


cam.pitch


# In[391]:


flight.set_rotation(tuple(unity_ylookat([1,0,0]).as_quat()))


# In[ ]:


unity_ylookat([1,0,0]).as_quat()


# In[114]:


for i in range(10):
    flight.set_rotation(tuple(unity_ylookat([1,0,0]).as_quat()))
    time.sleep(1)


# In[397]:


flight.aerodynamic_force


# In[450]:


from scipy.spatial.transform import Rotation as R
import numpy as np
import time
def unity_ylookat(dy ,dz=None):
    dy = np.array(dy)
    if dz is None: dz = Z.opa_math.make_norm(np.random.uniform(-1, 1, 3))
    dz = Z.opa_math.make_orth_norm(dz, dy)
    r1 = Z.opa_math.rot_look_at(dz, dy)
    return r1
   
def stereo_XY2xyz(X, Y, r):
    X = X/2/r
    Y = Y/2/r
    
    a = X**2 + Y**2
    return np.array([2*X, 2*Y, -a+1])/(a+1) * r

xl = np.linspace(0, 0, 1)
xl = np.linspace(-0.10, 0.10, 10)
tb = np.stack(np.meshgrid(xl, xl), axis=-1)
X,Y = tb.reshape((-1, 2)).T
vecs = stereo_XY2xyz(X,Y,1).T


# In[452]:


alt = 10000.
vx = [0, 0, 1]
flight.set_rotation(tuple(unity_ylookat(vx, dz=np.ones(3)).as_quat()))
pos = earth.position_at_altitude(90., 0., alt, earth.reference_frame)
print(pos, dir * vnorm)
vnorm = 10
for dir in vecs:
    flight.set_rotation(tuple(unity_ylookat(tuple(dir), dz=None).as_quat()))
    fx = flight.simulate_aerodynamic_force_at(earth, pos, tuple(dir*vnorm))
    print(dir, v, fx)
    v =np.abs(np.dot(fx, dir))


# In[274]:


ef = earth.reference_frame
ef.


# In[162]:


vecs[0], vecs[-1]


# In[36]:


r1= R.from_quat(flight.rotation)
r1.as_matrix()


# In[ ]:


for r in cnds:
    r = earth.equatorial_radius
    flight.simulate_aerodynamic_force_at(earth, (r,0,0), flight.direction)


# In[18]:


flight.direction


# In[19]:


r = earth.equatorial_radius
flight.simulate_aerodynamic_force_at(earth, (r,0,0), flight.direction)


# In[ ]:


earth.altitude_at_position(vpos,earth.reference_frame)

