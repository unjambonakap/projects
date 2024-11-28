#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import polars as pl
from chdrft.external.http_vlc import HttpVLC
import mpv
import folium
import folium.plugins

import fastapi, fastapi.encoders
import uvicorn
import chdrft.utils.rx_helpers as rx_helpers
import threading
import contextlib
from chdrft.config.env import qt_imports
import io
from fastapi.responses import JSONResponse
import shapely
import time
import reactivex as rx
from chdrft.display.service import oplt
from chdrft.dsp.datafile import Dataset
from chdrft.display.ui import OpaPlot
from sage.all import Polyhedron, RealDoubleField
#<NMEA(GPGGA, time=22:03:28, lat=44.8988163333, NS=N, lon=6.6427143333, EW=E, quality=1, numSV=6, HDOP=2.37, alt=1319.2, altUnit=M, sep=47.3, sepUnit=M, diffAge=, diffStation=)>
from chdrft.sim.rb.base import Transform, Vec3, make_rot
from chdrft.projects.gpx import GPXData, pl_to_numpy

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--infile')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def xyzw_to_wxyz(xyzw):
  return np.array(xyzw)[[3, 0, 1, 2]]


def wxyz_to_xyzw(xyzw):
  return np.array(xyzw)[[1, 2, 3, 0]]


def rot2wxyz(rot: R):
  return xyzw_to_wxyz(rot.as_quat())


def wxyz2rot(wxyz):
  return R.from_quat(wxyz_to_xyzw(wxyz))


def xyzw2rot(xyzw):
  return R.from_quat(xyzw)


class MapperEntry(cmisc.PatchedModel):
  func: object = cmisc.pyd_f(lambda: cmisc.identity)
  postproc: object = cmisc.pyd_f(lambda: (lambda x, stats: x))


class Converter(cmisc.PatchedModel):

  content: A
  payload_map: dict
  quat_ord: list[float] = cmisc.Field(default=[1, 3, 2, 0])
  quat_sgn: list[float] = cmisc.Field(default=[1, 1, -1, 1])
  lla2xyz: object = None
  xyz2lla: object = None

  @classmethod
  def Make(cls, content):
    return cls(content=content, payload_map={x.payload_id: x for x in content.samples.payloads})

  def map_entries(self, code, mapper: MapperEntry):
    items = self.content.samples.code2samples[code]

    stats = self.code_stats(code)

    def fw(x):

      res = mapper.func(np.array([y.value for y in x['values']]))
      if not isinstance(res, dict): return {code: res}
      return {f'{code}_{k}': v for k, v in res.items()}

    tb = [
        A(tsec=self.payload_map[x.payload_id].start_time + x.inner_payload_pos * stats.dt, **fw(x))
        for x in items
    ]
    df = pl.from_pandas(pd.DataFrame(tb))
    df = mapper.postproc(df, stats)
    return df

  def code_stats(self, code):
    items = self.content.samples.code2samples[code]

    pl = self.payload_map
    st = pl[items[0].payload_id].start_time
    et = pl[items[-1].payload_id].start_time + pl[items[-1].payload_id].duration
    dt = (et - st) / len(items)
    return A(st=st, et=et, dt=dt, freq=1 / dt)

  @property
  def rec_time(self):
    entry = cmisc.asq_query(self.content.metadata.streams).where(lambda x: x.codec_tag_string=='tmcd').single()
    ct = cmisc.datetime.fromisoformat(entry.tags.creation_time)
    tc = cmisc.datetime.strptime(entry.tags.timecode, '%H:%M:%S:%f')
    return np.datetime64(ct.replace(hour=tc.hour, minute=tc.minute, second=tc.second, microsecond=tc.microsecond))

  @property
  def start_time(self):
    return self.content.samples.payloads[0].start_time

  @property
  def end_time(self):
    return self.content.samples.payloads[-1].start_time + self.content.samples.payloads[-1].duration

  @property
  def codes(self) -> list[str]:
    return list(self.content.samples.code2samples.keys())

  @property
  def usable_codes(self) -> list[str]:
    return list(self.mappers.keys())

  def extract(self, code):
    return self.map_entries(code, self.mappers[code])

  def compute_downsample(self, code):
    return slice(None, None, int(self.max_freq / self.code_stats(code).freq + 0.5))

  @property
  def mappers(self):

    def norm_gp_quat(x):  # stored as wxzy, need xyzw
      return x[self.quat_ord] * self.quat_sgn
      return x[[1, 2, 3, 0]]

    return {
        'GRAV': MapperEntry(func=lambda x: np.array(x[[0, 2, 1]]) * [1, -1, -1]
                           ),  # given as x,-z,-y; local frame
        'ACCL': MapperEntry(),
        'GYRO': MapperEntry(),
        'CORI': MapperEntry(func=norm_gp_quat),
        'IORI': MapperEntry(func=norm_gp_quat),
        'GPS5': MapperEntry(func=lambda x: dict(lla=x[[1, 0, 2]], v=x[:2])
                           ),  #input data: lat, lon,alt,vground, v3d.
        # output gps5: lon,lat,alt
    }

  @property
  def max_freq(self):
    return max(self.code_stats(x).freq for x in self.usable_codes)

  def process(self, tfreq=None) -> pl.DataFrame:
    mp = A({c: self.extract(c) for c in self.usable_codes})
    res = cmisc.functools.reduce(
        lambda a, b: a.join(b, on='tsec', how='full', coalesce=True), mp.values()
    )

    res = res.sort(pl.col('tsec')).select(pl.all().forward_fill()).unique('tsec', keep='last')
    if tfreq is None: tfreq = self.max_freq
    res = pl.DataFrame({
        "t": self.rec_time + pd.to_timedelta(np.arange(self.start_time, self.end_time, 1 / tfreq), unit='s').values,
        "tsec": np.arange(self.start_time, self.end_time, 1 / tfreq),
    }).join_asof(res, on="tsec", strategy='nearest')
    return res



def cv_to_numpy(df):
  return pl_to_numpy(df, CORI=R.from_quat, IORI=R.from_quat)


def norm_vecs(vecs):
  return vecs / np.linalg.norm(vecs, axis=1).reshape((-1, 1))


class OptRunner:

  def run(self, content):
    with Z.tempfile.TemporaryDirectory() as tempdir:
      infile = f'{tempdir}/infile.json'
      outfile = f'{tempdir}/outfile.json'
      Z.FileFormatHelper.Write(infile, content)
      self.call(action='do_opt1', infile=infile, outfile=outfile, opt_type='cori')
      return A(input=content, output=Z.FileFormatHelper.Read(outfile))

  def call(self, **kwargs):

    def norm_args(d):
      return [f'--{k}={v}' for k, v in d.items()]

    Z.sp.check_call(
        ['./build/projects/uav/projects_uav_sample_tools.cpp'] + norm_args(kwargs),
        cwd='/home/benoit/programmation'
    )


vx, vy, vz = np.identity(3)
runner = OptRunner()


def get_data(cv: Converter, **kwargs):
  res = cv.process(**kwargs)

  dx = cv_to_numpy(res[['IORI', 'CORI', 'GRAV', 't', 'GPS5_lla']])
  dx.gpx_data = GPXData(
      pl.from_dict(A(
          lon=dx.GPS5_lla[:, 0],
          lat=dx.GPS5_lla[:, 1],
          alt=dx.GPS5_lla[:, 2],
          t=dx.t,
      )),
      dt=cv.code_stats('GPS5').dt
  )

  dx.lla = dx.gpx_data.data.lla
  dx.pos = dx.gpx_data.data.pos
  dx.smooth_speed = dx.gpx_data.data.speed
  dx.speed_dir = norm_vecs(dx.smooth_speed)
  dx.speed = np.linalg.norm(dx.smooth_speed, axis=1)
  return dx


def compute_cori0(cv, dx):
  t1 = dx.speed > np.max(dx.speed) / 2
  # guessed transforms:
  # image_space = IORI  * CORI * ORIG_CORI * world_space
  vx, vy, vz = np.identity(3)
  vfront = -vy
  meas_front = (dx.CORI.inv() * dx.IORI.inv()).apply(vfront)
  real_front = dx.speed_dir
  meas_grav = dx.CORI.inv().apply(norm_vecs(dx.GRAV))
  real_grav = np.repeat(-vz[np.newaxis, :], repeats=len(dx.IORI), axis=0)

  downsample_gps = cv.compute_downsample('GPS5')
  downsample_grav = cv.compute_downsample('GRAV')

  data = A(
      meas=np.concatenate((meas_front[t1][downsample_gps], meas_grav[t1][downsample_grav])),
      real=np.concatenate((real_front[t1][downsample_gps], real_grav[t1][downsample_grav])),
      #meas=meas_front[t1] ,
      #real=real_front[t1],
      cori=dx.CORI[t1].as_quat(),
  )

  res = runner.run(data)
  if res.output.converged:
    res.score = res.output.cost_final / len(data.meas)
    res.nmeas = len(data.meas)
  return res


def eval_quat_ord(sample_file, ord=None, sgn=None):
  content = Z.FileFormatHelper.Read(sample_file)
  cv = Converter.Make(content)
  if ord is not None: cv.quat_ord = list(ord)
  if sgn is not None: cv.quat_sgn = list(sgn)

  dx = get_data(cv)
  return compute_cori0(cv, dx)


def test_gopro_tlm(ctx):
  content = Z.FileFormatHelper.Read(ctx.infile)
  cv = Converter(content)
  print(cv.extract_cam_orient())
  print(cv.codes)


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
