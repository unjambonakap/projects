#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import pydantic
import pprint
from chdrft.utils.othreading import ThreadGroup
import time
import shutil
from pymavlink.dialects.v20 import ardupilotmega as ml
import io

global flags, cache
flags = None
cache = None

import pymavlink.dialects.v20.ardupilotmega as mavlink_msgs


def mavlink_message_to_dict(msg: mavlink_msgs.MAVLink_message) -> dict:
  data = {k: getattr(msg, k) for k in type(msg).fieldnames}
  header = msg.get_header()

  header_data = dict(
      len=header.mlen,
      seq=header.seq,
      src_sys=header.srcSystem,
      src_comp=header.srcComponent,
      msg_id=header.msgId
  )
  return dict(data=data, header=header_data)


def create_mav_heartbeat_gen():
  mav = ml.MAVLink(file=io.BytesIO, srcSystem=11, srcComponent=12)
  seq_id = 0
  while True:
    mav.seq = seq_id % 256
    seq_id += 1
    mav.file = io.BytesIO()
    mav.heartbeat_send(
        type=ml.MAV_TYPE_ONBOARD_CONTROLLER,
        autopilot=ml.MAV_AUTOPILOT_INVALID,
        base_mode=0,
        custom_mode=0,
        system_status=ml.MAV_STATE_ACTIVE,
    )
    yield mav.file.getvalue()
