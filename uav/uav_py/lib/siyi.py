#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field
import pydantic
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import chdrft.utils.Z as Z
from chdrft.dsp.image import ImageData
from chdrft.config.env import g_env
import enum
import struct
import crcmod
import ctypes
import time
import threading
import queue
from chdrft.tube.connection import Connection
import contextlib
import requests
import datetime
import concurrent.futures
from chdrft.display.render import ImageGrid
from chdrft.inputs.controller import OGamepad, configure_joy, debug_controller_state
from chdrft.utils.rx_helpers import ImageIO, rx, pipe_connect
from chdrft.dsp.utils import linearize_clamp
import asyncio
import typing
import base64
import urllib.parse
from chdrft.utils.path import FileFormatHelper

import Pyro5 as pyro
import Pyro5.server
import rich.json
import rich.live
from .proto import comm_p2p
from .proto.comm_p2p import SiyiJoyData, SiyiJoyMode
from . import tunnel

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


crcx = crcmod.mkCrcFun(0x11021, initCrc=0, rev=False)
kSiyiOffsetData = 8


def run_ev_protected_op(fun, timeout, *args, **kwargs):
  ev = threading.Event()
  th = threading.Thread(target=fun, args=(ev,) + args, kwargs=kwargs)

  with concurrent.futures.ThreadPoolExecutor() as executor:
    try:
      future = executor.submit(lambda: fun(ev, *args, **kwargs))
      future.result(timeout)
    except (KeyboardInterrupt, TimeoutError):
      pass
    ev.set()
    return future.result()


class SDFolder(cmisc.PatchedModel):
  count: int = 0


class SDState(cmisc.PatchedModel):
  folders: dict[str, SDFolder] = cmisc.Field(default_factory=dict)
  is_video: bool
  guessing: bool


class DeltaSDState(cmisc.PatchedModel):
  hc: SiyiHTTPClient
  orig: dict = cmisc.Field(default_factory=dict)
  res: dict = None
  diff: dict = None

  @contextlib.contextmanager
  def enter(self) -> DeltaSDState:
    self.orig = self.hc.get_files(False) | self.hc.get_files(True)
    yield self
    self.res = self.hc.get_files(False) | self.hc.get_files(True)
    self.diff = {k: v for k, v in self.res.items() if k not in self.orig}


class SyncResult(cmisc.PatchedModel):
  sd_state: SDState
  added: list[str] = cmisc.pyd_f(list)
  existing: list[str] = cmisc.pyd_f(list)


class SiyiHTTPClient(cmisc.PatchedModel):
  host: str | None
  mock: bool = False
  port: int = 82

  def delta(self) -> DeltaSDState:
    if self.mock: return None
    return DeltaSDState(hc=self)

  @property
  def base_url(self) -> str:
    return f'http://{self.host}:{self.port}/cgi-bin/media.cgi/api/v1'

  def query(self, action, params):
    res = A.RecursiveImport(requests.get(f'{self.base_url}/{action}', params).json())
    assert res.code == 200, res
    return res.data

  def get_directories(self, video: bool) -> dict[str, str]:
    return {
        x.name: x.path
        for x in self.query('getdirectories', dict(media_type=1 if video else 0)).directories
    }

  def get_media_count(self, path: str, video: bool) -> int:
    return self.query('getmediacount', dict(media_type=1 if video else 0, path=path)).count

  def get_media_list(self,
                     path: str,
                     start: int,
                     video: bool,
                     count: int = 0,
                     to: int = None) -> dict[str, str]:
    if to is not None:
      count = to + 1
    res = self.query(
        'getmedialist', dict(media_type=1 if video else 0, path=path, start=start, count=count)
    )

    return {x.name: x.url.replace('192.168.144.25', self.host) for x in res.list}

  def get_sd_state(
      self, video: bool = None, from_state: SDState = None, guessing: bool = None
  ) -> SDState:
    if self.mock:
      return SDState(is_video=video, guessing=guessing, folders=dict(abc=dict(count=3)))
    if from_state is not None and from_state.guessing:

      if guessing is None: guessing = True

      dirs = list(from_state.folders.keys())
      is_video = from_state.is_video
    else:
      if guessing is None: guessing = False
      dirs = self.get_directories(video=video)

    res = SDState(is_video=video, guessing=guessing)
    for name, dir in dirs.items():
      count = self.get_media_count(path=dir, video=video)
      f = res.folders[dir] = SDFolder(count=count)

    return res

  def sync_from(self, sync_folder: str, from_state: SDState) -> SyncResult:
    to_state = self.get_sd_state(video=from_state.is_video)
    res = SyncResult(sd_state=to_state)
    for rel, f in self.get_delta_files(from_state, to_state).items():
      ofile = os.path.join(sync_folder, rel)
      if self.maybe_download(ofile, f.url):
        res.added.append(ofile)
      else:
        res.existing.append(ofile)
    return res

  def get_fname_url(self, dir, is_video, i):
    fname = 'IMG_{i:04d}.jpg'
    type = ['photo', 'mp4'][is_video]
    if is_video:
      fname = f'REC_{i:04d}.mp4'
    return fname, f'http://{self.host}:{self.port}/{type}/{dir}/{fname}'

  def get_delta_files(self, from_state: SDState, to_state: SDState) -> dict[str, str]:
    res = {}
    for dir, to_folder in to_state.folders.items():
      from_folder = from_state.folders.get(dir, SDFolder())
      if from_folder.count == to_folder.count: continue

      if to_state.guessing:
        cur = dict(
            [
                self.get_fname_url(dir, to_state.is_video, i)
                for i in range(from_folder.count, to_folder.count)
            ]
        )
      else:
        cur = self.get_media_list(
            path=dir, video=to_state.is_video, start=from_count, to=to_count - 1
        )
      glog.info(f'Getting delta files {from_count=} {to_count=} {len(cur)=}')

      res |= {f'{dir}/{k}': A(url=v, video=to_state.is_video) for k, v in cur.items()}
    return res

  def get_files(self, video: bool) -> dict[str, str]:
    return self.get_delta_files(from_state=SDState(), to_state=sel.get_sd_state(video))

  def file_content(self, url):
    return requests.get(url, stream=True).raw.read()

  def maybe_download(self, target, url) -> bool:
    if os.path.exists(target): return False
    cmisc.makedirs(os.path.dirname(target))
    FileFormatHelper.Write(target, self.file_content(url))
    return True


DeltaSDState.update_forward_refs()


class SyncHelper(cmisc.PatchedModel):
  host: str
  sync_folder: str
  video: bool = False

  @cmisc.cached_property
  def hc(self) -> SiyiHTTPClient:
    return SiyiHTTPClient(host=self.host)

  @property
  def dump_file(self):
    return os.path.join(self.sync_folder, 'sync.json')

  def save_state(self, state: SDState):
    FileFormatHelper.Write(self.dump_file, state)

  def init(self):
    self.save_state(self.hc.get_sd_state(self.video))

  @property
  def cur_state(self):
    return SDState.parse_obj(FileFormatHelper.Read(self.dump_file))

  def sync(self):
    rs = self.hc.sync_from(self.sync_folder, self.cur_state)
    self.save_state(rs.sd_state)
    print(rs.json())


class SiyiMessage(cmisc.PatchedModel):
  need_ack: bool
  ack_pack: bool
  data_len: int
  seq: int
  cmd_id: int
  data: bytes | None = None
  crc: int = 0
  prefix: int = 0x6655
  obj: object = None

  @classmethod
  def compute_crc(cls, data) -> int:
    return crcx(data)

  def prepare(self) -> bytes:
    ctrl = self.need_ack | self.ack_pack << 1
    data = struct.pack(
        '<HBHHB', self.prefix, ctrl, self.data_len, self.seq, self.cmd_id
    ) + self.data
    return data

  @property
  def sizeof(self) -> int:
    return self.data_len + kSiyiOffsetData + 2

  @classmethod
  def unpack(cls, data, with_data: bool = True) -> SiyiMessage:
    (prefix, ctrl, data_len, seq, cmd_id) = struct.unpack('<HBHHB', data[:kSiyiOffsetData])
    if with_data:
      crc, = struct.unpack('<H', data[-2:])
      assert len(data) == data_len + kSiyiOffsetData + 2
      actual = cls.compute_crc(data[:-2])
      assert crc == actual
      msg_data = data[kSiyiOffsetData:kSiyiOffsetData + data_len]
    else:
      msg_data = None

    return cls(
        need_ack=ctrl & 1,
        ack_pack=ctrl >> 1,
        seq=seq,
        cmd_id=cmd_id,
        data_len=data_len,
        data=msg_data
    )

  def pack(self) -> bytes:
    content = self.prepare()
    crc = SiyiMessage.compute_crc(content)
    return content + struct.pack('<H', crc)


siyi_message_types = {}


class SiyiResolution(enum.Enum):
  # R_120p = (160, 120) not working
  R_192p = (256, 192)
  R_240p = (320, 240)
  R_360p = (480, 360)
  R_480p = (720, 480)
  R_720p = (1280, 720)
  R_1080p = (1920, 1020)
  R_2K = (2560, 1440)
  R_4K = (3840, 2160)


class SiyiCmd_Metaclass(type):

  def __new__(cls, clsname, bases, attrs):

    res = type.__new__(cls, clsname, bases, attrs)

    def define_wrapper(obj, info):
      if obj is None: return None

      fields = []
      norms = {}
      for k, v in obj.__dict__.items():
        if k.startswith('_'): continue
        if v.__module__ != ctypes.__name__:
          fields.append((k, v.ctype))
          norms[k] = v
        else:
          fields.append((k, v))

      class Fx(ctypes.LittleEndianStructure):
        _fields_ = fields
        _pack_ = 1
        _base = res

        def __init__(self, **kwargs):
          self.base = res
          data = {k: 0 for k, _ in fields}
          data.update(
              {
                  k: getattr(norms.get(k, object()), 'pack', cmisc.identity)(v)
                  for k, v in kwargs.items()
              }
          )
          super().__init__(**data)

        def _info(self):
          return info

        def pack(self):
          return bytes(self)

        def to_obj(self):
          return A(
              #raw=raw,
              **{
                  k: getattr(norms.get(k, object()), 'unpack', cmisc.identity)(getattr(self, k))
                  for k, _ in fields
              }
          )

        @classmethod
        def unpack(clx, data):
          assert ctypes.sizeof(clx) == len(data)
          raw = clx.from_buffer_copy(data)
          return raw.to_obj()

      if info == 'Req':

        def default_mock_res(x):
          return x.base.Res()

        Fx._to_mock_res = getattr(obj, '_to_mock_res', default_mock_res)

      for attr in '__doc__', '__name__', '__qualname__', '__module__':
        setattr(Fx, attr, getattr(obj, attr))
      return Fx

    siyi_message_types[res.ID] = res
    for x in ('Req', 'Res'):
      u = getattr(res, x)
      setattr(res, x, define_wrapper(getattr(res, x), info=x))
    return res


class SiyiRecordStatus(enum.IntEnum):
  RECORD_OFF = 0
  RECORD_ON = 1
  NO_SD = 2
  RECORD_DATALOSS = 3


class SiyiGimbalMotionMode(enum.IntEnum):
  LOCK = 0
  FOLLOW = 1
  FPV = 2


class SiyiGimbalMountingDir(enum.IntEnum):
  RESERVED = 0
  NORMAL = 1
  UPSIDE_DOWN = 2


class SiyiActionType(enum.IntEnum):
  PICTURE_TAKE = 0
  HDR_TOGGLE = 1
  RECORDING_TOGGLE = 2
  MOTION_LOCK = 3
  MOTION_FOLLOW = 4
  MOTION_FPV = 5


class SiyiFunctionFeedback(enum.IntEnum):
  SUCCESS = 0
  FAIL_PHOTO = 1
  HDR_ON = 2
  HDR_OFF = 3
  FAIL_VIDEO = 4


class SiyiManualZoom(enum.IntEnum):
  ZOOM_IN = 1
  STOP = 0
  ZOOM_OUT = 255


class FloatAsS16:
  ctype = ctypes.c_int16

  def pack(self, v):
    return int(round(v * 10))

  def unpack(self, v):
    return float(v) / 10


class FloatDecAsU16:
  ctype = ctypes.c_uint16

  def pack(self, v):
    return int(v) + int((v - int(v)) * 10) * 256

  def unpack(self, v):
    return (v & 0xff) + (v >> 8) / 10


class UnitFloatAsToS8Range:
  ctype = ctypes.c_int8

  def __init__(self, bound):
    self.bound = bound

  def pack(self, v):
    return int(linearize_clamp(v, -1, 1, -self.bound, self.bound))

  def unpack(self, v):
    return linearize_clamp(float(v), -self.bound, self.bound, -1, 1)


class EnumAsU8:
  ctype = ctypes.c_uint8

  def __init__(self, en: enum.Enum):
    self.en = en

  def pack(self, v):
    return v.value

  def unpack(self, v):
    return self.en(v)


class SiyiCmd_Firmware(metaclass=SiyiCmd_Metaclass):
  ID = 0x01

  class Req:
    pass

  class Res(ctypes.Structure):
    code_board_ver = ctypes.c_uint32
    gimbal_firmware_ver = ctypes.c_uint32
    #zoom_firmware_ver = ctypes.c_uint32


class SiyiCmd_FormatSD(metaclass=SiyiCmd_Metaclass):
  ID = 0x48

  class Req:
    pass

  class Res:
    sta = ctypes.c_uint8


class SiyiCmd_Heartbeat(metaclass=SiyiCmd_Metaclass):
  ID = 0x00

  class Req:
    pass

  class Res:
    pass


class SiyiCmd_Center(metaclass=SiyiCmd_Metaclass):
  ID = 0x08

  class Req:
    center_pos = ctypes.c_uint8

    def _to_mock_res(self):
      tmp = self.base.Res(sta=1)
      return tmp

  class Res:
    sta = ctypes.c_uint8


class SiyiCmd_GimbalRotSpeed(metaclass=SiyiCmd_Metaclass):
  ID = 0x07

  class Req:
    yaw = UnitFloatAsToS8Range(100)
    pitch = UnitFloatAsToS8Range(100)

  class Res:
    success = ctypes.c_uint8


class SiyiCmd_GimbalAngle(metaclass=SiyiCmd_Metaclass):
  ID = 0x0e

  class Req:
    yaw = FloatAsS16()
    pitch = FloatAsS16()

  class Res:
    yaw = FloatAsS16()
    pitch = FloatAsS16()
    roll = FloatAsS16()


class SiyiCmd_SetAngleSingleAxis(metaclass=SiyiCmd_Metaclass):
  ID = 0x41

  class Req:
    angle = FloatAsS16()
    axis = ctypes.c_uint8

  class Res:
    yaw = FloatAsS16()
    pitch = FloatAsS16()
    roll = FloatAsS16()


class SiyiCmd_RequestGimbalAttitude(metaclass=SiyiCmd_Metaclass):
  ID = 0x0d

  class Req:
    pass

  class Res:
    yaw = FloatAsS16()
    pitch = FloatAsS16()
    roll = FloatAsS16()
    v_yaw = FloatAsS16()
    v_pitch = FloatAsS16()
    v_roll = FloatAsS16()


class SiyiCmd_RequestGimbalInfo(metaclass=SiyiCmd_Metaclass):
  ID = 0x0a

  class Req:
    pass

  class Res:
    reserved = ctypes.c_uint8
    hdr_sta = ctypes.c_uint8
    reserved2 = ctypes.c_uint8
    record_sta = EnumAsU8(SiyiRecordStatus)
    gimbal_motion = EnumAsU8(SiyiGimbalMotionMode)
    gimbal_mounting_dir = EnumAsU8(SiyiGimbalMountingDir)
    video_sta = ctypes.c_uint8


class SiyiCmd_RequestCodecSpecs(metaclass=SiyiCmd_Metaclass):
  ID = 0x20

  class Req:
    stream_type = ctypes.c_uint8

    def _to_mock_res(self):
      tmp = self.base.Res(stream_type=self.stream_type)
      return tmp

  class Res:
    stream_type = ctypes.c_uint8
    video_enc_type = ctypes.c_uint8
    resolution_l = ctypes.c_uint16
    resolution_h = ctypes.c_uint16
    video_bitrate = ctypes.c_uint16
    video_framerate = ctypes.c_uint8


class SiyiCmd_SendCodecSpecs(metaclass=SiyiCmd_Metaclass):
  ID = 0x21

  class Req:
    stream_type = ctypes.c_uint8
    video_enc_type = ctypes.c_uint8
    resolution_l = ctypes.c_uint16
    resolution_h = ctypes.c_uint16
    video_bitrate = ctypes.c_uint16
    reserve = ctypes.c_uint8

  class Res:
    stream_type = ctypes.c_uint8
    sta = ctypes.c_uint8


class SiyiCmd_ManualZoom(metaclass=SiyiCmd_Metaclass):
  ID = 0x05

  class Req:
    action = EnumAsU8(SiyiManualZoom)

  class Res:
    cur_zoom = FloatAsS16()


class SiyiCmd_GetMaxZoom(metaclass=SiyiCmd_Metaclass):
  ID = 0x16

  class Req:
    pass

  class Res:
    val = FloatDecAsU16()


class SiyiCmd_GetZoom(metaclass=SiyiCmd_Metaclass):
  ID = 0x18

  class Req:
    pass

  class Res:
    val = FloatDecAsU16()


class SiyiCmd_SetZoom(metaclass=SiyiCmd_Metaclass):
  ID = 0x0f

  class Req:
    val = FloatDecAsU16()

  class Res:
    sta = ctypes.c_uint8


class SiyiCmd_SendFCAttitude(metaclass=SiyiCmd_Metaclass):
  ID = 0x22

  class Req:
    roll = ctypes.c_float
    pitch = ctypes.c_float
    yaw = ctypes.c_float
    v_roll = ctypes.c_float
    v_pitch = ctypes.c_float
    v_yaw = ctypes.c_float

  class Res:
    pass


class SiyiCmd_SendFCData(metaclass=SiyiCmd_Metaclass):
  ID = 0x3e

  class Req:
    time_since_boot = ctypes.c_uint32
    lat = ctypes.c_int32
    lon = ctypes.c_int32
    alt = ctypes.c_int32
    alt_ell = ctypes.c_int32
    vn = ctypes.c_float
    ve = ctypes.c_float
    vd = ctypes.c_float

  class Res:
    pass


class SiyiCmd_SetUtcTime(metaclass=SiyiCmd_Metaclass):
  ID = 0x30

  class Req:
    timestamp = ctypes.c_uint64

  class Res:
    ack = ctypes.c_int8


class SiyiCmd_SoftReset(metaclass=SiyiCmd_Metaclass):
  ID = 0x80

  class Req:
    cam_reset = ctypes.c_uint8
    gimbal_reset = ctypes.c_uint8

  class Res:
    cam_reset = ctypes.c_uint8
    gimbal_reset = ctypes.c_uint8


class SiyiCmd_PhotoVideoAction(metaclass=SiyiCmd_Metaclass):
  ID = 0x0c

  class Req:
    action = EnumAsU8(SiyiActionType)

  Res = None


class SiyiCmd_FuncFeedback(metaclass=SiyiCmd_Metaclass):
  ID = 0x0b

  Req = None

  class Res:
    info = EnumAsU8(SiyiFunctionFeedback)


class SiyiCmd_GetGimbalMode(metaclass=SiyiCmd_Metaclass):
  ID = 0x19

  class Req:
    pass

  class Res:
    motion = EnumAsU8(SiyiGimbalMotionMode)


class SiyiDataFreq(enum.IntEnum):
  OFF = 0
  HZ_2 = 1
  HZ_4 = 2
  HZ_5 = 3
  HZ_10 = 4
  HZ_20 = 5
  HZ_50 = 6
  HZ_100 = 7


class SiyiCmd_RequestGimbalAttitudeStream(metaclass=SiyiCmd_Metaclass):
  ID = 0x25

  class Req:
    data_type: ctypes.c_uint8
    data_freq = EnumAsU8(SiyiDataFreq)

  class Res:
    data_type: ctypes.c_uint8


import dataclasses


class SiyiHandler(cmisc.PatchedModel):
  conn: Connection | None = None
  mock: bool = False
  seq: int = 0
  q: dict[int, queue.Queue] = cmisc.Field(default_factory=dict)
  handlers: tunnel.HandlersManager = cmisc.pyd_f(tunnel.HandlersManager)

  @contextlib.contextmanager
  def enter(self):
    for x in siyi_message_types.keys():
      self.q[x] = queue.Queue()

    self.handlers.backup_handler = lambda msg: self.q[msg.typ._base.ID].put_nowait(msg.obj)
    if self.mock:
      yield self
      return

    with self.conn:
      ev = threading.Event()
      t = threading.Thread(target=self.recv_thread, args=(ev,))
      t.start()
      yield self
      ev.set()
      t.join()

  def recv_thread(self, ev: threading.Event):
    while not ev.is_set():
      try:
        b = self.conn.recv_fixed_size(kSiyiOffsetData, timeout=1)
        msg = SiyiMessage.unpack(b, with_data=False)
        glog.debug(str((b, msg)))
        rem = self.conn.recv_fixed_size(msg.sizeof - kSiyiOffsetData)
        b += rem
        m = self.unpack(b, is_req=False)
        glog.info(f'SiyiHandler: Real msg >> {m}')

        self.recv_msg(m)
      except cmisc.TimeoutException:
        pass

  def wait_msg(self, cmd_id: int, timeout=None) -> dict | None:
    q = self.q[cmd_id]
    if self.mock:
      if q.empty():
        return siyi_message_types[cmd_id].Res().to_obj()
      return q.get_nowait()
    else:
      try:
        return q.get(timeout=timeout)
      except queue.Empty:
        print('GOT timeout')
        pass

  def clear_cmd(self, cmd_id: int):
    while not self.q[cmd_id].empty():
      self.q[cmd_id].get_nowait()

  def send_msg(self, msg, wait_reply: bool = False, timeout=None) -> dict | None:
    cmd_id = msg.base.ID

    if wait_reply:
      self.clear_cmd(cmd_id)

    if not self.mock:
      glog.warn('SEND')
      d = self.pack(msg, self.seq)
      self.conn.send(d)
      glog.warn('DONE SEND')
    else:
      if msg.base.Res is not None:
        self.recv_msg(self.unpack(self.pack(msg._to_mock_res(), seq=0), is_req=False))

    if wait_reply:
      glog.warn('WATI')
      res = self.wait_msg(cmd_id, timeout=timeout)
      glog.warn('DONE WATI')
      return res

  async def asend_msg(self, msg, func=None, filter=None):
    cmd_id = msg.base.ID

    loop = asyncio.get_event_loop()
    f = loop.create_future()

    def fx(msg):
      if func is not None: func(msg)
      loop.call_soon_threadsafe(f.set_result, msg)

    with self.handlers.add_handler_oneshot(msg.base.Res, fx, absorb=True, filter=filter):
      self.send_msg(msg)
      return await f

  def recv_msg(self, m: A):
    self.handlers.handle(m, key=m.typ)

  def pack(self, msg, seq):
    data = msg.pack()
    m = SiyiMessage(
        need_ack=True,
        ack_pack=False,
        data=data,
        data_len=len(data),
        cmd_id=type(msg)._base.ID,
        seq=seq
    )
    return m.pack()

  def unpack(self, data, is_req=False) -> A:

    m = SiyiMessage.unpack(data, with_data=True)
    typ = siyi_message_types[m.cmd_id]
    cl_typ = typ.Req if is_req else typ.Res

    return A(obj=cl_typ.unpack(m.data), typ=cl_typ, raw=m)


@pyro.server.expose
class CameraManager(cmisc.PatchedModel):
  seq: int = 0
  mock: bool = False
  target_hostname: str = None
  target_port: int = None
  start_state: DeltaSDState = None

  @cmisc.cached_property
  def hc(self) -> SiyiHTTPClient:
    return SiyiHTTPClient(host=self.target_hostname, mock=self.mock)

  @cmisc.cached_property
  def handler(self) -> SiyiHandler:
    if self.mock:
      conn = None
    else:
      conn = Connection(self.target_hostname, self.target_port, udp=True, udp_bind_port=12345)
    hx = SiyiHandler(conn=conn, seq=self.seq, mock=self.mock)
    return hx

  @contextlib.contextmanager
  def enter(self) -> CameraManager:
    with self.handler.enter():
      self.start_state = self.hc.delta()
      yield self

  def get_attitude(self):
    return self.handler.send_msg(SiyiCmd_RequestGimbalAttitude.Req(), wait_reply=True).obj

  async def aget_attitude(self):
    return (await self.handler.asend_msg(SiyiCmd_RequestGimbalAttitude.Req())).obj

  def request_attitude_stream(self, freq: SiyiDataFreq):
    assert 0, 'not working, crashing cam'
    self.handler.send_msg(
        SiyiCmd_RequestGimbalAttitudeStream.Req(data_type=1, data_freq=freq), wait_reply=True
    )

  def get_firmware(self):
    return self.handler.send_msg(SiyiCmd_Firmware.Req(), wait_reply=True)

  def set_speed(self, yaw, pitch, **kwargs):
    return self.handler.send_msg(
        SiyiCmd_GimbalRotSpeed.Req(yaw=yaw, pitch=pitch), wait_reply=True, **kwargs
    )

  def rel_zoom(self, v: float, **kwargs) -> float:
    zoom_map = {
        -1: SiyiManualZoom.ZOOM_OUT,
        0: SiyiManualZoom.STOP,
        1: SiyiManualZoom.ZOOM_IN,
    }
    return self.handler.send_msg(
        SiyiCmd_ManualZoom.Req(action=zoom_map[v]), wait_reply=True, **kwargs
    ).cur_zoom

  def zoom(self, v: float):
    assert self.handler.send_msg(SiyiCmd_SetZoom.Req(val=v), wait_reply=True).sta == 1

  def max_zoom(self) -> float:
    return self.handler.send_msg(SiyiCmd_GetMaxZoom().Req(), wait_reply=True).val

  def get_zoom(self) -> float:
    return self.handler.send_msg(SiyiCmd_GetZoom.Req(), wait_reply=True).val

  def set_mode(self, mode: SiyiGimbalMotionMode):
    motion2action = {
        SiyiGimbalMotionMode.FPV: SiyiActionType.MOTION_FPV,
        SiyiGimbalMotionMode.FOLLOW: SiyiActionType.MOTION_FOLLOW,
        SiyiGimbalMotionMode.LOCK: SiyiActionType.MOTION_LOCK,
    }
    self.handler.send_msg(SiyiCmd_PhotoVideoAction.Req(action=motion2action[mode]))

  def get_mode(self) -> SiyiGimbalMotionMode:
    return self.handler.send_msg(SiyiCmd_GetGimbalMode.Req(), wait_reply=True).motion

  def set_resolution(
      self,
      res: SiyiResolution,
      stream_type,
  ):
    ans = self.handler.send_msg(
        SiyiCmd_SendCodecSpecs.Req(
            stream_type=stream_type,
            video_enc_type=2,
            resolution_l=res.value[0],
            resolution_h=res.value[1],
            video_bitrate=1000,
            reserve=15,
        ),
        wait_reply=True
    )
    assert ans.sta == 1

  def get_res(self, msg_type: typing.Type[SiyiMessage]):
    return self.handler.send_msg(msg_type.Req(), wait_reply=True)

  def get_codec_specs(self, stream_type) -> A:
    return self.handler.send_msg(
        SiyiCmd_RequestCodecSpecs.Req(stream_type=stream_type), wait_reply=True
    )

  def get_gimbal_info(self) -> A:
    return self.get_res(SiyiCmd_RequestGimbalInfo)

  def format(self):
    assert self.handler.send_msg(SiyiCmd_FormatSD.Req(), wait_reply=True).sta == 1

  def reset(self):
    self.handler.send_msg(SiyiCmd_SoftReset.Req(gimbal_reset=1, cam_reset=1), wait_reply=True)

  def center(self):
    assert self.handler.send_msg(SiyiCmd_Center.Req(center_pos=1), wait_reply=True).sta == 1

  def record_video_as_photos(self, stop_sig: threading.Event, cb_fc_state=lambda idx: {}):
    data = A()
    data.start_state = self.hc.get_files(video=False)
    idx = 0
    entries = []
    while not stop_sig.is_set():
      t0 = time.time()
      self.handler.clear_cmd(SiyiCmd_FuncFeedback.ID)
      entry = A(t=t0, idx=idx)

      self.take_picture(
          lambda: entry.update(dict(att=self.get_attitude(), fc_state=cb_fc_state(idx)))
      )
      entry.ell = time.time() - t0

      entries.append(entry)
      idx += 1

    data.entries = entries
    stop_sig.clear()
    return data

  def take_picture(self, wait_cb=cmisc.nop_func):
    self.handler.clear_cmd(SiyiCmd_FuncFeedback.ID)
    self.handler.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.PICTURE_TAKE))
    glog.info('Try take picture')
    wait_cb()
    ff = self.handler.wait_msg(SiyiCmd_FuncFeedback.ID)
    glog.info(f'take picture>> {ff}')
    assert ff.info == SiyiFunctionFeedback.SUCCESS

  def toggle_video(self):
    self.handler.clear_cmd(SiyiCmd_FuncFeedback.ID)
    self.handler.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.RECORDING_TOGGLE))
    #ff = self.handler.wait_msg(SiyiCmd_FuncFeedback.ID)
    #glog.info(f'toggle vide >> {ff}')
    #return ff

    #assert ff.info == SiyiFunctionFeedback.SUCCESS

  def set_utc(self, dt: datetime.datetime = None):
    if dt is None: dt = datetime.datetime.utcnow()
    self.send_msg(SiyiCmd_SetUtcTime.Req(timestamp=int(dt.timestamp()) * 1000), wait_reply=True)


class ButtonKind(enum.Enum):
  MODE = 'mode'
  CTRL = 'ctrl'
  ZOOM = 'zoom'
  RECORD_COUNT = 'record_count'
  PHOTO_COUNT = 'photo_count'
  CENTER = 'center'


def siyi_joypad(gp: OGamepad | None):

  def mixaxis(v1, v2):
    v = (max(v1, v2) + 1) / 2
    if v1 > v2: return -v
    return v

  def buttons2unit(a, b):
    match (a, b):
      case (True, _):
        return 1
      case (False, True):
        return -1
      case _:
        return 0

  mp = {
      ('Y', True): lambda s: s.inc(ButtonKind.PHOTO_COUNT),
      ('RB', True): lambda s: s.inc(ButtonKind.CENTER),
      ('START', True): lambda s: s.inc(ButtonKind.RECORD_COUNT),
      'DPAD-Y': lambda s, v: s.set(ButtonKind.ZOOM, v),
      ('LT', 'RT'): lambda s, v: s.set(ButtonKind.CTRL, mixaxis(*v)),
      'LB': lambda s, v: s.set(ButtonKind.MODE, {
          True: SiyiJoyMode.PITCH,
      }.get(v, SiyiJoyMode.YAW)),
  }
  return configure_joy(gp, mp, {x: 0 for x in ButtonKind})


class StatsStreamParams(cmisc.PatchedModel):
  name: str
  gather_ellapsed: bool = False


class StatsStream(cmisc.PatchedModel):
  params: StatsStreamParams
  data: list = cmisc.pyd_f(list)
  context: A = cmisc.pyd_f(A)

  @contextlib.contextmanager
  def begin_entry(self) -> StatsEntry:
    start_time = time.clock_gettime(time.CLOCK_REALTIME)
    st = time.monotonic()

    e = dict()
    yield e
    if self.params.gather_ellapsed:
      e['gather_ellapsed'] = time.monotonic() - st
    e['start_time'] = start_time
    self.data.append(e)

  async def alog_func(self, get_data_func):
    with self.begin_entry() as e:
      e.update(await get_data_func())

  def log_func(self, get_data_func):
    with self.begin_entry() as e:
      e.update(get_data_func())

  def dump_data(self) -> dict:
    return dict(params=self.params, context=self.context, data=self.data)


class StatsCollectorParams(tunnel.pydantic_settings.BaseSettings):
  dump_file: str = '/dev/null'


class StatsCollector(cmisc.ContextModel):
  params: StatsCollectorParams
  streams: list[StatsStream] = cmisc.pyd_f(list)

  def create(self, params: StatsStreamParams) -> StatsStream:
    stream = StatsStream(params=params)
    self.streams.append(stream)
    return stream

  def dump_data(self) -> dict:
    return dict(params=self.params, streams=[s.dump_data() for s in self.streams])

  def __exit__(self, *args):
    FileFormatHelper.Write(self.params.dump_file, self.dump_data())

    super().__exit__(*args)


class ImageCollectorParams(tunnel.pydantic_settings.BaseSettings):
  image_freq_hz: float = 0.5
  enable: bool = False


class SiyiServerParams(tunnel.pydantic_settings.BaseSettings):
  send_status_freq_hz: float = 1
  attitude_collection_freq_hz: float = 5
  image_collector_params: ImageCollectorParams = cmisc.pyd_f(ImageCollectorParams)
  stats_params: StatsCollectorParams = cmisc.pyd_f(StatsCollectorParams)


@pyro.server.expose
class SiyiServer(cmisc.ContextModel):
  th: tunnel.PyMsgTunnelData
  m: CameraManager

  params: SiyiServerParams = cmisc.pyd_f(SiyiServerParams)
  status: A = cmisc.pyd_f(A)
  last_joy: SiyiJoyData = SiyiJoyData()
  cmd_timeout: float = 0.3

  sc: StatsCollector = None
  stream_attitude: StatsStream = None

  def __post_init__(self):
    super().__post_init__()
    self.sc = StatsCollector(params=self.params.stats_params)
    self.stream_attitude = self.sc.create(StatsStreamParams(name='attitude', gather_ellapsed=True))

    self.th.hm.add_handler(SiyiJoyData, self.recv_joy_data)
    self.th.hm.add_handler(SiyiServerParams, self.set_params)
    self.th.tg.add_timer(lambda: self.params.send_status_freq_hz, self.proc_status)
    self.th.tg.add_timer(
        lambda: self.params.attitude_collection_freq_hz,
      lambda: self.stream_attitude.alog_func(self.m.aget_attitude), force_async=True)
    self.ctx_push(self.m.enter())
    self.ctx_push(self.th)
    self.ctx_push(self.sc)
    self.ctx_push(
        ImageCollector(
            m=self.m, sc=self.sc, params=self.params.image_collector_params, tg=self.th.tg
        )
    )

  @property
  def siyi_data(self):
    return A(joy=self.m.last_joy, status=self.status)

  def recv_joy_data(self, joy: SiyiJoyData):
    if joy.photo_count != self.last_joy.photo_count:
      self.m.take_picture()
    if joy.center != self.last_joy.center:
      self.m.center()
    if (joy.record_count - self.last_joy.record_count) & 1 != 0:
      self.m.toggle_video()

    def get_joyv(joy: SiyiJoyData, mode):
      return 0 if joy.mode != mode else joy.ctrl

    self.m.set_speed(
        get_joyv(joy, SiyiJoyMode.YAW), get_joyv(joy, SiyiJoyMode.PITCH), timeout=self.cmd_timeout
    )
    if joy.zoom != self.last_joy.zoom:
      self.m.rel_zoom(joy.zoom, timeout=self.cmd_timeout)
    self.last_joy = joy


  def set_params(self, params):
    self.params = params

  def siyimessage2pyd(self, x: A) -> comm_p2p.SiyiRawReq:
    msg = x.ans.typ(**x.ans.obj)
    return comm_p2p.SiyiRawReq(
        desc=x.desc, msg_id=x.ans.typ._base.ID, res=base64.b64encode(msg.pack()).decode()
    )

  async def siyireq2pyd(self, req, desc='', **kwargs) -> comm_p2p.SiyiRawReq:
    ans = await self.m.handler.asend_msg(req, **kwargs)
    return A(ans=ans, desc=desc, name=f'{ans.typ._base.__name__}/{desc}')

  async def proc_status(self) -> list:
    async with asyncio.timeout(1), asyncio.TaskGroup() as tg:
      mlist = [
          self.siyireq2pyd(
              SiyiCmd_RequestCodecSpecs.Req(stream_type=0),
              'stream0',
              filter=lambda x: x.obj.stream_type == 0
          ),
          self.siyireq2pyd(
              SiyiCmd_RequestCodecSpecs.Req(stream_type=1),
              'stream1',
              filter=lambda x: x.obj.stream_type == 1
          ),
          self.siyireq2pyd(SiyiCmd_RequestGimbalInfo.Req()),
      ]
      tasks = [tg.create_task(x) for x in mlist]

    for x in tasks:
      xr = x.result()
      self.status[xr.name] = xr.ans.obj
      self.th.push_msg(self.siyimessage2pyd(xr))


class ImageCollector(cmisc.ContextModel):
  m: CameraManager
  sc: StatsCollector
  params: ImageCollectorParams
  tg: tunnel.ThreadGroup

  cur_state: SDState = None
  active_dir: str = None
  stream: StatsStream = None

  @cmisc.cached_property
  def active_folder(self) -> SDFolder:
    return self.cur_state.folders[self.active_dir]

  def __post_init__(self):
    super().__post_init__()
    self.tg.add_timer(self.get_freq, self.proc)
    self.stream = self.sc.create(StatsStreamParams(name='images', gather_ellapsed=True))

  @property
  def last_image(self):
    return (self.active_dir, self.active_folder.count)

  def get_freq(self) -> float:
    return self.params.image_freq_hz

  def __enter__(self):
    self.cur_state = self.m.hc.get_sd_state(video=False, guessing=True)
    self.active_dir = cmisc.single_value(self.cur_state.folders.keys())
    self.stream.context.update(
        params=self.params.copy(), start_state=self.cur_state.copy(), active_dir=self.active_dir
    )
    return super().__enter__()

  def proc(self):
    if not self.params.enable: return

    with self.stream.begin_entry() as e:
      self.m.take_picture()

      pic_id = self.active_folder.count
      self.active_folder.count += 1
      e.update(pic_id=pic_id)


class SiyiControllerParams(cmisc.PatchedModel):
  input_rate: float = 4


class SiyiController(cmisc.ContextModel):
  gp: OGamepad | None
  th: tunnel.PyMsgTunnelData
  params: SiyiControllerParams = cmisc.pyd_f(SiyiControllerParams)
  msg_handler: tunnel.HandlersManager = cmisc.pyd_f(tunnel.HandlersManager)

  siyi_data: A = cmisc.pyd_f(A)
  joy_state: object = None

  def handle_recv_req(self, msg: comm_p2p.SiyiRawReq):
    msg_typ: SiyiCmd_Metaclass = siyi_message_types[msg.msg_id]
    res = msg_typ.Res.unpack(base64.b64decode(msg.res))
    self.siyi_data[(msg_typ.__name__, msg.desc)] = res
    self.msg_handler.handle(res, key=(type(res), msg.desc))

  def __post_init__(self):
    super().__post_init__()

    self.th.hm.add_handler(comm_p2p.SiyiRawReq, self.handle_recv_req)
    self.siyi_data.joy = SiyiJoyData()

    self.msg_handler.verbose = False
    self.msg_handler.add_handler((SiyiCmd_RequestGimbalInfo.Res, ''), self.handle_gimbal_info)

    def timer_joy():
      jd = SiyiJoyData(**{k.value: v for k, v in self.joy_state.last.inputs.items()})
      self.siyi_data.joy = jd
      self.th.push_msg(jd)

    joyd = siyi_joypad(self.gp)
    self.joy_state = joyd.obj
    self.th.tg.add_timer(lambda: self.params.input_rate, timer_joy)
    self.ctx_push(joyd)
    self.ctx_push(self.th)

  def handle_gimbal_info(self, msg: SiyiCmd_RequestGimbalInfo.Res):
    mp = {
        SiyiRecordStatus.RECORD_ON: 10,
        SiyiRecordStatus.RECORD_OFF: 0,
        SiyiRecordStatus.NO_SD: 5,
        SiyiRecordStatus.RECORD_DATALOSS: 4,
    }

    self.gp.set_led(mp[msg.record_sta])


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
