#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import chdrft.utils.Z as Z
import numpy as np
from pydantic.v1 import Field
import pandas as pd
from scipy.spatial.transform import Rotation as R
import astropy
from astropy import units as u
import polars as pl

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
  func: object = Field(default_factory=lambda: cmisc.identity)
  postproc: object = Field(default_factory=lambda: (lambda x, stats: x))


def to_pos(x):
  return astropy.coordinates.WGS84GeodeticRepresentation(
      x[1] * u.deg, x[0] * u.deg, x[2] * u.m, 'WGS84'
  ).to_cartesian().xyz.value


def lla_to_local_coord(lla):
  base_lla = lla.mean(axis=0)
  # right hand basis, need lon lat alt

  def lla2xyz(lla):
    return astropy.coordinates.WGS84GeodeticRepresentation(
        lla[0] * u.deg, lla[1] * u.deg, lla[2] * u.m, 'WGS84'
    ).to_cartesian().xyz.to(u.m).value
  p0 = lla2xyz(base_lla)

  cmul = 1e-5
  diff = [
    lla2xyz(base_lla + [cmul, 0, 0]),
    lla2xyz(base_lla + [0, cmul, 0]),
    lla2xyz(base_lla + [0, 0, 1]),
  ]
  scale = np.array([np.linalg.norm(x-p0) for x in diff]) / (cmul, cmul, 1)
  rmp = (lla - base_lla.reshape((1, 3)))
  return rmp * np.reshape(scale, (1, 3))


def compute_speed(df, period):

  wsize = int(1 / period)
  c = df['GPS5_pos']
  u = df.select(
      re=pl.col('GPS5_pos').shift(-wsize).forward_fill().backward_fill(),
      rs=pl.col('GPS5_pos').shift(wsize).forward_fill().backward_fill(),
  )
  cmul = 2 * wsize * period
  pos = u.with_row_index('i').explode(['re', 'rs']
                                     ).group_by('i').agg(diff=(pl.col('re') - pl.col('rs')) / cmul)
  return pos['diff']


def proc_gps(df, stats):
  df = df.with_columns(GPS5_pos=lla_to_local_coord(np.array(df['GPS5_lla'].to_list())))
  speed = compute_speed(df, stats.dt)
  return df.with_columns(GPS5_smooth_speed=speed)


class Converter(cmisc.PatchedModel):

  content: A
  payload_map: dict
  quat_ord: list[float] = Field(default=[1,3,2,0])
  quat_sgn: list[float] = Field(default=[1,1,-1,1])

  @classmethod
  def Make(cls, content):
    return cls(content=content, payload_map={x.payload_id: x for x in content.payloads})

  def map_entries(self, code, mapper: MapperEntry):
    items = self.content.code2samples[code]

    stats = self.code_stats(code)

    def fw(x):

      res = mapper.func(np.array([y.value for y in x['values']]))
      if not isinstance(res, dict): return {code: res}
      return {f'{code}_{k}': v for k, v in res.items()}

    tb = [
        A(t=self.payload_map[x.payload_id].start_time + x.inner_payload_pos * stats.dt, **fw(x))
        for x in items
    ]
    df = pl.from_pandas(pd.DataFrame(tb))
    df = mapper.postproc(df, stats)
    return df

  def code_stats(self, code):
    items = self.content.code2samples[code]

    pl = self.payload_map
    st = pl[items[0].payload_id].start_time
    et = pl[items[-1].payload_id].start_time + pl[items[-1].payload_id].duration
    dt = (et - st) / len(items)
    return A(st=st, et=et, dt=dt, freq=1 / dt)

  @property
  def start_time(self):
    return self.content.payloads[0].start_time

  @property
  def end_time(self):
    return self.content.payloads[-1].start_time + self.content.payloads[-1].duration

  @property
  def codes(self) -> list[str]:
    return list(self.content.code2samples.keys())

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
        'GPS5': MapperEntry(func=lambda x: dict(lla=x[[1, 0, 2]], v=x[:2]),
                            postproc=proc_gps),  #lat, lon,alt,vground,, v3d
    }

  @property
  def max_freq(self):
    return max(self.code_stats(x).freq for x in self.usable_codes)

  def process(self):
    mp = A({c: self.extract(c) for c in self.usable_codes})
    res = cmisc.functools.reduce(
        lambda a, b: a.join(b, on='t', how='full', coalesce=True), mp.values()
    )

    res = res.sort(pl.col('t')).select(pl.all().forward_fill()).unique('t', keep='last')
    res = pl.DataFrame({
        "t": np.arange(self.start_time, self.end_time, 1 / self.max_freq)
    }).join_asof(res, on="t", strategy='nearest')
    return res


def test_gopro_tlm(ctx):
  content = Z.FileFormatHelper.Read(ctx.infile)
  cv = Converter(content)
  print(cv.extract_cam_orient())
  print(cv.codes)


def test_plot(ctx):
  import chdrft.utils.K as K
  K.oplt.plot(np.array(np.arange(1, 10)), typ='graph')
  input()


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
