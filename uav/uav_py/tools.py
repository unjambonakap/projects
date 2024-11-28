#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
from pymavlink.dialects.v20 import common as mavlink2
import os
import struct
from chdrft.tube.file_like import FileLike
from chdrft.tube.connection import Connection
import threading

import time
import sys
import pydantic
import typing
import pydantic_settings
import pickle

os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as ml
from pymavlink.dialects.v20.ardupilotmega import MAVLink
import io
import pymavlink.mavutil
from lib.tunnel import TunnelHelper, Conf

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--config-file', type=str)
  parser.add_argument('--toggle-dir', action='store_true')
  parser.add_argument('--plain-endpoint', type=str)
  parser.add_argument('--encoded-endpoint', type=str)

  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  #%%

  conf = Conf(**pydantic_settings.YamlConfigSettingsSource(Conf, ctx.config_file)()
             ).maybe_toggle(ctx.toggle_dir)

  with Connection(ctx.plain_endpoint) as f_plain, Connection(ctx.encoded_endpoint) as f_enc:
    th_encode = TunnelHelper(src=conf.src, dst=conf.dst)
    th_decode = TunnelHelper(src=conf.src, dst=conf.dst)
    td = TunnelData(
        th_encode=th_encode,
        th_decode=th_decode,
        f_plain=f_plain,
        f_enc=f_enc,
    )
    td.tg.start()
    td.tg.join()


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
