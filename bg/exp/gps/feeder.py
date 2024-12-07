#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import time
import chdrft.utils.Z as Z
from chdrft.tube.connection import Server
import subprocess as sp
import chdrft.projects.gpx as  ogpx
import pynmea2
import datetime



global flags, cache
flags = None
cache = None

def deg2nmea(f):
  assert f >= 0
  deg = int(f)
  min = (f-int(f))* 60
  return f'{deg:02d}{min:07.04f}'


class MessageGen(cmisc.PatchedModel):

  @property
  def base_msg(self):
    return pynmea2.GGA.parse('$GPGGA,170400.132,5232.183,N,01325.977,E,1,12,1.0,0.0,M,0.0,M,,*6C')


  def gen(self, lon, lat, alt, time: datetime.datetime):
    msg = self.base_msg
    def set_signed(v, field, neg_pos_char):
      if v < 0:
        v=  -v
        setattr(msg, field+ '_dir', neg_pos_char[0])
      else:
        setattr(msg, field+ '_dir', neg_pos_char[1])
      setattr(msg, field, deg2nmea(v))

    set_signed(lon, 'lon', 'SN')
    set_signed(lat, 'lat', 'WE')
    msg.altitude = alt
    msg.timestamp = time.strftime('%H%M%S.%f')[:-3]
    return msg





def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile', type=str)
  parser.add_argument('--feed-speed', type=float, default=1)
  parser.add_argument('--start-agent', action='store_true')
  parser.add_argument('--refresh-rate-sec', type=float, default=0.1)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



def feed_gpx(ctx, fname):
  gpx = ogpx.GPXData.FromFile(fname , loop=True, tot_duration=np.timedelta64(10, 'm'))
  mgen = MessageGen()
  ix = gpx.idf_t
  glog.debug('%s', ix.index[[0, -1]])
  for i in cmisc.itertools.count():
    ti = ix.index[0] + i * ctx.refresh_rate_sec * ctx.feed_speed
    if ti > ix.index[-1]: break
    l = ix(ti)
    glog.debug(f'Set {l.lon} {l.lat} {l.alt}')
    msg = mgen.gen(l.lon, l.lat, l.alt, datetime.datetime.now(datetime.UTC))
    glog.debug(f'{msg}')
    print(msg)
    time.sleep(ctx.refresh_rate_sec)
  #%%


def feed_nmea(ctx, fname):
  lines = open(fname, 'r').readlines()
  for x in cmisc.itertools.cycle(lines):
    print(x + '\n')
    time.sleep(0.1)


def test(ctx):
  if ctx.start_agent:
    sp.Popen('/usr/lib/geoclue-2.0/demos/agent', shell=1)

  print(ctx.infile)
  if ctx.infile.endswith('.gpx'):
    feed_gpx(ctx, ctx.infile)
  else:
    return feed_nmea(ctx, ctx.infile)




def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()


#ctx = app.setup_jup('--infile='+cmisc.proc_path(r'~/Downloads/gr54-depuis-le-bourg-d-oisans.gpx'), parser_funcs=[args])
#test(ctx)
