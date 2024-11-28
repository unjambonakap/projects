#!/usr/bin/env python
from __future__ import annotations
import numpy as np
from chdrft.config.env import init_jupyter

init_jupyter()

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import inspect
import chdrft.projects.gpx as ogpx
import chdrft.utils.K as K
import chdrft.utils.colors as ocolors
import pipe as p
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import OpaPlot
import tqdm
from tqdm.contrib.concurrent import process_map
import functools
import re
from chdrft.config.env import g_env, qt_imports

global flags, cache
flags = None
cache = None


import astropy.units as u


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--infile')

def return_to_ipython():
  locs,_ = cmisc.get_n2_locals_and_globals(n=0)
  get_ipython().user_global_ns.update(locs)
  assert 0


def test(ctx):
  #import serial
  #a = serial.Serial('/dev/ttyUSB0', baudrate=57600, timeout=0)
  from PyQt5 import QtSerialPort

  qp = QtSerialPort
  p = qp.QSerialPort()
  p.setPortName('/dev/ttyUSB0')
  p.setBaudRate(115200)
  p.setDataBits(8)
  p.setParity(qp.QSerialPort.Parity.NoParity)
  p.setStopBits(qp.QSerialPort.StopBits.OneStop)
  p.setFlowControl(qp.QSerialPort.FlowControl.NoFlowControl)
  r = p.open(qt_imports.QtCore.QIODevice.ReadWrite)
  print(r)
  #p.close()

  return_to_ipython()

  #idx = sys.argv.index('--')
  #ctx = app.setup_jup('', parser_funcs=[args], argv=sys.argv[idx:])

  return

def test_siyi(ctx):
  import serial
  a = serial.Serial(ctx.infile, baudrate=115200, timeout=0)


  msg = bytes.fromhex('556601000000000164c4')
  print(msg)
  a.write(msg)
  print(a.read(10))
  a.write(msg)
  print(a.read(10))


#%%

def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
#%%
