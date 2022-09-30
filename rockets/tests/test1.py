#!/usr/bin/env python

from typing import Tuple, Optional
from dataclasses import dataclass
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize as A
import chdrft.utils.misc as cmisc
import glog
import numpy as np
from chdrft.utils.types import *
from pydantic import BaseModel, Field
from typing import Tuple
import xarray as xr
from typing import Callable, List
from scipy.spatial.transform import Rotation as R
from chdrft.display.base import TriangleActorBase
from chdrft.utils.math import MatHelper
import itertools
from enum import Enum
import functools
from chdrft.sim.phys import *
import chdrft.utils.Z as Z

global flags, cache
flags = None
cache = None


def go():

  from chdrft.sim.blender import BlenderPhysHelper, ObjectSync
  from chdrft.display.blender import clear_scene, AnimationSceneHelper, KeyframeObjData
  from chdrft.sim.base import compute_cam_parameters
  clear_scene()

  rs = scene_test()
  tg = rs.target
  sctx = rs.sctx

  helper = BlenderPhysHelper()
  helper.load(tg.rb)
  cam_loc = np.array([5,0,0])
  aabb = tg.aabb()
  params = compute_cam_parameters(cam_loc, aabb.center, Vec3.Z().data, aabb.points, blender=True)
  helper.cam.data.angle_y = params.angle_box.yn
  helper.cam.mat_world = params.toworld
  helper.update()


  #r0l = sctx.compose([wlink])
  #tg = r0l
  #px.plot(by_col=True, fx=0.2)


  dt = 1e-2
  nsz = 300000
  print(f'start >> ', tg.world_angular_momentum())
  #for i in range(10):
  #  print('FUU ', tg.get_particles(10000000).angular_momentum())
  #return

  #osync = ObjectSync(helper.obj2blender[rbl.child], helper.cam)

  animation = AnimationSceneHelper(frame_step=1)
  animation.start()
  dt = 1e-2
  for i in range(510):
    print()
    print()
    print()
    print()
    tc = TorqueComputer()
    tc.setup(tg, dt)
    tc.update()
    for x in sctx.obj2name.keys():
      rl = x.self_link.root_link
      print(f'>>>>>>>> {x.name=} {rl.rotvec_w=} {rl.world_angular_momentum()=}')
    print(f'{i:03d} {i*dt:.2f}', tg.world_angular_momentum())

    updates = []
    for child in tc.updates.keys():
      updates.append(KeyframeObjData(obj=helper.obj2blender[child], wl=child.self_link.wl.data))
    helper.update()

    #osync.sync()
    updates.append(KeyframeObjData(obj=helper.cam, wl=helper.cam.mat_world))
    animation.push(updates)

  animation.finish()
  cam = helper.cam

  return helper


helper = go()
