#!/usr/bin/env python
from __future__ import annotations
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

global flags, cache
flags = None
cache = None

pl.config.Config.set_tbl_rows(20)
pl.config.Config.set_tbl_cols(20)

import astropy.units as u

def return_to_ipython():
  locs,_ = cmisc.get_n2_locals_and_globals(n=0)
  get_ipython().user_global_ns.update(locs)
  assert 0

def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)
  parser.add_argument('--input-dir')



def proc_files(files, func):
  results = process_map(func, files)
  lst = []
  for fname, res in zip(files, results):
    if res:
      lst.append(fname)
  return lst


def test(ctx):

  #idx = sys.argv.index('--')
  #ctx = app.setup_jup('', parser_funcs=[args], argv=sys.argv[idx:])

  return

  #%%
  ctx = A()
  ctx.input_dir = '/home/benoit/data/t1/data/'
  files = list(cmisc.list_files_rec(ctx.input_dir))
  print('laa', ctx.input_dir)
  fx = functools.partial(procit, stop=True)
  lst = proc_files(files, fx)

  #Z.FileFormatHelper.Write('./stuff.pickle', A(interest1=lst))
  stuff = Z.FileFormatHelper.Read('./stuff.pickle')

  #%%


  #%%
  l2 = proc_files(stuff.interest1, is_plane)

  #Z.FileFormatHelper.Write('./stuff.pickle', stuff.update(l2=l2))

  #%%
  procit(l2[4])
  #%%






def is_plane(fname):
  data = Z.FileFormatHelper.Read(fname)
  if not 'desc' in data: return
  if 'copter' in data.desc.lower(): return
  if not re.search('AIRBUS|BOEING', data.desc, re.IGNORECASE): return
  return True


def procit(fname, stop=False):

  data = Z.FileFormatHelper.Read(fname)

  ts = data.timestamp
  ft2m = u.imperial.ft.to(u.m)
  fpm2mps = (u.imperial.ft / u.min).to(u.m / u.s)
  deg2rad = u.deg.to(u.rad)

  def num_none(x, scale=1):
    if isinstance(x, str):
      return None
    if x is None: return x

    return x * scale

  rows = [
      dict(
          t=ts + x[0],
          lat=x[1],
          lon=x[2],
          ground=x[3] == 'ground',
          alt=num_none(x[3], ft2m),
          track=num_none(x[4], deg2rad),
          vrate=num_none(x[5], fpm2mps)
      ) for x in data.trace
  ]

  df = pl.from_records(
      rows, schema_overrides=dict(alt=pl.Float64, vrate=pl.Float64, track=pl.Float64)
  ).with_row_index()
  speed_window = 10
  if len(df) < speed_window * 10: return
  df = df.with_columns(
      DQ=(pl.col('ground').shift(1).backward_fill() != df['ground']),
      t=pl.from_epoch('t', time_unit='s'),
      ts=pl.col('t'),
      alt=pl.col('alt').backward_fill().forward_fill().fill_null(0),
  )
  gx = ogpx.GPXData(df, xyz_ecef=True, speed_window=speed_window)

  df = gx.df
  speed_thresh = 10
  stops = df.filter(df['speed_v'] < speed_thresh).filter(pl.col('ground') == False)
  inters = ogpx.InterFolder.Proc(list(stops['index'].to_list() | p.map(lambda x: (x, x + 1))))
  stop_thresh_s = 50
  ix = set()
  for a, b in inters:
    dt = gx.data.t[b - 1] - gx.data.t[a]
    if dt > stop_thresh_s:
      ix.update(range(a, b))
  gx.df = gx.df.with_columns(
      ground=pl.when(pl.col('index').is_in(ix)).then(True).otherwise(pl.col('ground'))
  )

  df = gx.df.with_columns(group=df['DQ'].cum_sum())
  flights = df.filter(df['ground'] == False)
  airports = df.filter(df['ground'] == True).group_by(pl.col('group')).first()
  by_group = dict(
      list(airports.partition_by(['group'], as_dict=True).items()) |
      p.map(lambda kv: (kv[0][0], kv[1]['lon', 'lat', 'index'].row(0, named=True)))
  )

  gb = flights.group_by(pl.col('group'))
  cmap = ocolors.ColorPool()
  if len(by_group) == 0: return

  ls = []
  pts = []
  found=False
  for (group,), v in gb:
    v = ogpx.pl_to_numpy(v)
    mdist = np.ones(len(v.index)) * 1e9
    for idx in (group - 1, group + 1):
      if idx not in by_group: continue
      e = by_group[idx]
      apos = gx.data.pos[e['index']].reshape((1, -1))
      diff = np.linalg.norm(gx.data.pos[v.index] - apos, axis=1)
      mdist = np.minimum(mdist, diff)

    airport_dist_thresh_m = 10_000
    time_manouver_s = 300

    ixl = list(v.index[mdist < airport_dist_thresh_m])
    inters = ogpx.InterFolder.Proc(list(ixl | p.map(lambda x: (x, x + 1))))
    for a, b in inters:
      dt = gx.data.t[b - 1] - gx.data.t[a]
      if dt > time_manouver_s:
        print('MANOUVER IN ', dt)
        found = True

  if not found: return
  if stop: return True


  for (group,), v in gb:
    ll = gx.data.lla[v['index'], :2]
    if len(ll) < 2: continue
    ls.append(ogpx.cc.shapely_as_feature(ogpx.shapely.LineString(ll), color='#%06x' % cmap.get()))
    pts.extend(ll)

  pts = np.array(pts)

  return_to_ipython()
  #%%
  x0 = df.row(0, named=True)
  fh = ogpx.cc.FoliumHelper()
  fh.create_folium((x0['lon'], x0['lat']), zoom_start=15, sat=False)

  _ = ogpx.cc.folium.GeoJson(
      dict(type='FeatureCollection', features=ls), style_function=lambda f: f['properties']
  ).add_to(fh.m)

  ogpx.cc.folium.plugins.FastMarkerCluster(pts[:, [1, 0]]).add_to(fh.m)

  fh.m.show_in_browser()
  #%%
  dx = Dataset(x=xx.data.t, y=xx.data.lon, name='z')
  opp = OpaPlot(dx, legend=1)

  K.oplt.create_window()

  gw = K.oplt.find_window().gw
  gw.add(opp, label='data')

  gw.add(fh.generate_widget())

  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
#%%
