#!/usr/bin/env python
# coding: utf-8

# In[1]:


init_jupyter()
import krpc


# In[2]:


conn = krpc.connect(name='My Example Program',
                    address='localhost',
                    rpc_port=50000,
                    stream_port=50001)
print(conn.krpc.get_status().version)


# In[14]:


conn.space_center.launch_vessel2('VAB', "./Ships/VAB/SpaceX Falcon 9 Block 5.craft", 'LaunchPad', True)


# In[3]:


vessel = conn.space_center.active_vessel
flight_info = vessel.flight()
earth = conn.space_center.bodies['Kerbin']
flight = vessel.flight(earth.reference_frame)


# In[4]:



vessel.control.activate_next_stage()


# In[ ]:


vessel.control.sas = False
vessel.control.rcs = False
vessel.control.throttle = 1.0
conn.nop(req_phys_loop=1)
vessel.control.activate_next_stage()
conn.nop(req_phys_loop=1)
vessel.control.activate_next_stage()
conn.nop(req_phys_loop=1)
vpos = vessel.position(earth.reference_frame)
flight.set_position(tuple(np.array(vpos)*1.5))
conn.nop(req_phys_loop=1)
flight.set_position(tuple(np.array(vpos)*1.9))
conn.nop(req_phys_loop=1)


# In[90]:


flight.set_position(tuple(np.array(vpos)*3.9))

conn.nop(req_phys_loop=1)
vessel.control.throttle = 0


# In[76]:


e0  = vessel.active_engines[0]
tank = [x for x in vessel.parts.all if 'S1tank' in x.name ][0]
t0 = [x for x in vessel.parts.all if 'Merlin' in x.name ][0]

vx = A(vessel=vessel, tank=tank)
rscs = set()
for engine in vessel.active_engines:
    for prop in engine.propellants:
        for rsc in vessel.resources.with_resource_by_id(prop.id):
            rscs.add(rsc)
def set_prop(v, target_part=None):
    for rsc in rscs:
        if target_part is None or target_part == rsc.part:
            rsc.amount = rsc.max * v
    conn.nop(req_phys_loop=1)
    
vx.set_s1_prop= lambda v: set_prop(v, tank)
vx.set_prop = set_prop
import time


# In[52]:



for x in rscs:
print(x.amount,x.density, x.part)


# In[7]:





# In[8]:


# checking flow is correct
e0.throttle_locked=True
e0.throttle=1
conn.nop(req_phys_loop=1)
start = tank.mass
fx = e0.cur_flow
tx = 10
time.sleep(tx)
print(e0.step_mass_flow, e0.cur_flow * conn.space_center.time_warp_helper.fixed_delta_time)
end = tank.mass
e0.throttle_locked=True
e0.throttle=0
(start-end)/fx, tx


# In[109]:





# In[117]:


def get_vessel_prop():
    vx.set_prop(1)
    vx.set_s1_prop(0)
    data = A()
    data.tank_mass_empty = tank.mass
    data.vessel_mass_empty = vessel.mass

    vx.set_s1_prop(1)
    conn.nop(req_phys_loop=1)
    data.prop_mass = tank.mass - data.tank_mass_empty
    lon = earth.longitude_at_position(tank.position(vx.vessel.reference_frame),
                                      vx.vessel.reference_frame)
    lat = earth.latitude_at_position(tank.position(vx.vessel.reference_frame),
                                     vx.vessel.reference_frame)
    alt = earth.altitude_at_position(tank.position(vx.vessel.reference_frame),
                                     vx.vessel.reference_frame)
    data.tank_start_geo = np.array([lat, lon, alt])
    return data


def get_earth_prop():
    return A(GM=earth.gravitational_parameter,
             mass=earth.mass,
             radius=earth.equatorial_radius,
             rot_per=earth.rotational_period)


def analyse_gimballed_engine(engine, nx=3, ny=3):

    engine.gimbal.enable_gimbal = False
    data = dict()
    for th in engine.thrusters:
        for i in np.linspace(-1, 1, nx):
            for j in np.linspace(-1, 1, ny):
                rot = engine.gimbal.actuation2_rot([i, j])
                engine.gimbal.set_gimbal_rot(rot)
                conn.nop(req_phys_loop=1)
                data[(i, j)] = th.thrust_direction(tank.reference_frame)
    return data


def do_measure(data, f, col_name=None):
    measures = []
    for x in itertools.product(*data.values()):
        d = A({k: v for k, v in zip(data.keys(), x)})
        y = f(d)
        if isinstance(y, dict):
            for k, v in y.items():
                d[k] = v
        else:
            d[col_name] = y
        measures.append(d)

    return pd.DataFrame.from_records(measures)


def measure_thrust_and_flow(x):
    e0 = vessel.active_engines[0]
    return A(thrust=e0.get_thrust_adv(earth, x.thrust, x.altitude, x.speed),
             flow=e0.get_mass_flow(earth, x.thrust, x.altitude, x.speed))


def aero_forces(x):
    orientation = K.R.from_euler('XZY',
                                 [x.yaw, x.pitch, 0]).as_matrix() @ [0, 1, 0]
    return conn.space_center.aerodynamics.sim_aero_force_by_alt(
        earth, vessel, orientation * x.norm_speed, x.altitude)


def measure_earth_prop(x):
    return A(pressure=earth.pressure_at(x.altitude),
             density=earth.density_at(x.altitude))


def measure_phys_prop(x):
    set_s1_prop(x.resource_lvl)
    return dict(inertia_tensor=vessel.inertia_tensor,
                mass=vessel.mass,
                tank2com=np.array(vessel.position(tank.reference_frame)))


def compute_thruster_geo():
    thruster_geo = cmisc.defaultdict(list)
    for i, engine in enumerate(vessel.active_engines):
        assert len(engine.thrusters) == 1
        thruster_geo['pos'].append(engine.thrusters[0].thrust_position(
            tank.reference_frame))
        thruster_geo['dir'].append(analyse_gimballed_engine(engine))
    return thruster_geo


# In[113]:


data = A()
data.earth_prop = get_earth_prop()
data.vessel_prop = get_vessel_prop()


# In[114]:


vx.set_prop(1)
 
data.thruster_geo = compute_thruster_geo()
    
thrust_measures = dict(speed=np.linspace(0, 4000, num=10),
                          thrust=[1],
                          altitude=np.geomspace(1,80000, num=10)
                         )
aero_measures= A(altitude=np.linspace(0, 80000, num=10), norm_speed=np.linspace(0,4000, num=10), yaw=np.linspace(-np.pi, np.pi, num=11), pitch=np.linspace(-np.pi/2, np.pi/2, num=11))
pression_measures= A(altitude=np.linspace(0, 80000, num=10))

phys_measures = A(resource_lvl=np.linspace(0,1,11))
data.engine_thrust_and_flow = do_measure(thrust_measures, measure_thrust_and_flow)
data.friction= do_measure(aero_measures, aero_forces, col_name='friction')
data.phys_measures= do_measure(phys_measures, measure_phys_prop)


# In[118]:


Z.FileFormatHelper.Write('../data/measures.pickle', data)


# In[119]:


u = Z.FileFormatHelper.Read('../data/measures.pickle')


# In[140]:


e0 = vessel.active_engines[0]
e0.get_thrust_adv(
    earth, 1,
    earth.altitude_at_position(e0.part.center_of_mass(flight.reference_frame),
                               flight.reference_frame) * pa2atm,
    np.linalg.norm(flight.velocity))
#e0.get_thrust_adv(earth, 1,flight.mean_altitude, np.linalg.norm(flight.velocity))
#e0.get_thrust(1,earth.pressure_at(flight.mean_altitude)* pa2atm)


# In[108]:


earth.equatorial_radius


# In[11]:



e0.get_mass_flow(earth, 1,flight.mean_altitude, np.linalg.norm(flight.velocity))


# In[11]:


earth.altitude_at_position(flight.center_of_mass,flight.re)


# In[ ]:


R.


# In[10]:


e0.mixture_density


# In[37]:


e0.throttle_locked = True
e0.throttle = 0.0


# In[41]:


for i in np.linspace(0,1,11):
    e0.throttle = i
    conn.nop(req_phys_loop=1)
    print(i, e0.cur_thrust, e0.thrust, e0.cur_flow)


# In[33]:


e0.is_independant_throttle = True
e0.independant_throttle = 1
conn.nop(req_phys_loop=1)
e0.cur_thrust, e0.thrust
for i in range(10):
    print(e0.cur_flow / e0.mixture_density* 0.02)
    for x in e0.propellants:
        print(x.current_amount, x.current_requirement, x.total_resource_available, x.ratio)


# In[ ]:


e0.dens


# In[ ]:





# In[9]:


conn.space_center.time_warp_helper.fixed_delta_time


# In[182]:


set_prop(1)


# In[ ]:


from scipy.spatial.transform import Rotation as R
import numpy as np
import time
def unity_ylookat(dy ,dz=None):
    dy = np.array(dy)
    if dz is None: dz = Z.opa_math.make_norm(np.random.uniform(-1, 1, 3))
    dz = Z.opa_math.make_orth_norm(dz, dy)
    r1 = Z.opa_math.rot_look_at(dz, dy)
    return r1

