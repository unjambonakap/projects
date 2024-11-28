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

if not g_env.slim:
  from chdrft.display.service import g_plot_service as oplt

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


class DeltaSDState(cmisc.PatchedModel):
  hc: SiyiHTTPClient
  orig: list[dict] = cmisc.Field(default_factory=dict)
  res: list[dict] = None
  diff: list[dict] = None

  @contextlib.contextmanager
  def enter(self) -> DeltaSDState:
    self.orig = self.hc.get_files(False) | self.hc.get_files(True)
    yield self
    self.res = self.hc.get_files(False) | self.hc.get_files(True)
    self.diff = {k: v for k, v in self.res.items() if k not in self.orig}


class SiyiHTTPClient(cmisc.PatchedModel):
  host: str

  def delta(self) -> DeltaSDState:
    return DeltaSDState(hc=self)

  @property
  def base_url(self) -> str:
    return f'http://{self.host}:82/cgi-bin/media.cgi/api/v1'

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

  def get_media_list(self, path: str, start: int, count: int, video: bool) -> dict[str, str]:
    res = self.query(
        'getmedialist', dict(media_type=1 if video else 0, path=path, start=start, count=count)
    )
    return {x.name: x.url.replace('192.168.144.25', self.host) for x in res.list}

  def get_files(self, video: bool) -> dict[str, str]:
    dirs = self.get_directories(video=video)
    res = {}
    for name, dir in dirs.items():
      count = self.get_media_count(path=dir, video=video)
      if count <= 0: continue
      cur = self.get_media_list(path=dir, video=video, start=0, count=count)
      res |= {f'{dir}/{k}': A(url=v, video=video) for k, v in cur.items()}
    return res

  def file_content(self, url):
    return requests.get(url, stream=True).raw.read()

  def maybe_download(self, target, url):
    if os.path.exists(target): return
    cmisc.makedirs(os.path.dirname(target))
    Z.FileFormatHelper.Write(target, self.file_content(url))


DeltaSDState.update_forward_refs()


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
  R_360p  = (480, 360)
  R_480p  = (720, 480)
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
        if k.startswith('__'): continue
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
    ax: ctypes.c_uint64

  class Res:
    data_type: ctypes.c_uint8


import dataclasses


class SiyiHandler(cmisc.PatchedModel):
  conn: Connection | None = None
  mock: bool = False
  seq: int = 0
  q: dict[int, queue.Queue] = cmisc.Field(default_factory=dict)

  @contextlib.contextmanager
  def enter(self):
    for x in siyi_message_types.keys():
      self.q[x] = queue.Queue()

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
        glog.debug(f'SiyiHandler: Real msg >> {m}')

        self.q[msg.cmd_id].put_nowait(m.obj)
      except cmisc.TimeoutException:
        pass

  def wait_msg(self, cmd_id: int) -> dict:
    q = self.q[cmd_id]
    if self.mock:
      if q.empty():
        return siyi_message_types[cmd_id].Res().to_obj()
      return q.get_nowait()
    else:
      return q.get()

  def clear_cmd(self, cmd_id: int):
    while not self.q[cmd_id].empty():
      self.q[cmd_id].get_nowait()

  def send_msg(self, msg, wait_reply: bool = False) -> dict | None:
    cmd_id = msg.base.ID

    if wait_reply:
      self.clear_cmd(cmd_id)

    d = self.pack(msg, self.seq)

    if not self.mock:
      self.conn.send(d)
    else:
      if msg.base.Res is not None:
        res = self.unpack(self.pack(msg.base.Res(), 0))
        self.q[cmd_id].put_nowait(res.obj)
    if wait_reply:
      res = self.wait_msg(cmd_id)
      return res

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

  def unpack(self, data, is_req=False) -> SiyiMessage:
    m = SiyiMessage.unpack(data, with_data=True)
    typ = siyi_message_types[m.cmd_id]
    cl_typ = typ.Req if is_req else typ.Res

    m.obj = cl_typ.unpack(m.data)
    return m



class CameraManager(cmisc.PatchedModel):
  seq: int = 0
  mock: bool = False
  target_hostname: str = None
  target_port: int = None

  @cmisc.cached_property
  def hc(self) -> SiyiHTTPClient:
    assert not self.mock
    return SiyiHTTPClient(host=self.target_hostname)

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
      yield self

  def get_attitude(self):
    return self.handler.send_msg(SiyiCmd_RequestGimbalAttitude.Req(), wait_reply=True)

  def set_speed(self, yaw, pitch):
    return self.handler.send_msg(SiyiCmd_GimbalRotSpeed.Req(yaw=yaw, pitch=pitch), wait_reply=True)

  def rel_zoom(self, v: float) -> float:
    zoom_map = {
        -1: SiyiManualZoom.ZOOM_OUT,
        0: SiyiManualZoom.STOP,
        1: SiyiManualZoom.ZOOM_IN,
    }
    return self.handler.send_msg(
        SiyiCmd_ManualZoom.Req(action=zoom_map[v]), wait_reply=True
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

  def set_resolution(self, res: SiyiResolution, stream_type):
    ans = self.handler.send_msg(
        SiyiCmd_SendCodecSpecs.Req(
            stream_type=stream_type,
            video_enc_type=2,
            resolution_l=res.value[0],
            resolution_h=res.value[1],
            video_bitrate=20000
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
    assert self.handler.send_msg(SiyiCmd_Center.Req(center_pos=0), wait_reply=True).sta == 1

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
    wait_cb()
    ff = self.handler.wait_msg(SiyiCmd_FuncFeedback.ID)
    glog.info(f'take picture>> {ff}')
    assert ff.info == SiyiFunctionFeedback.SUCCESS

  def toggle_video(self):
    self.handler.clear_cmd(SiyiCmd_FuncFeedback.ID)
    self.handler.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.RECORDING_TOGGLE))
    ff = self.handler.wait_msg(SiyiCmd_FuncFeedback.ID)
    glog.info(f'toggle vide >> {ff}')
    return ff

    #assert ff.info == SiyiFunctionFeedback.SUCCESS


class ButtonKind(enum.Enum):
  MODE = 'mode'
  CTRL = 'ctrl'
  ZOOM = 'zoom'
  RECORD_COUNT = 'photo_count'
  PHOTO_COUNT = 'record_count'


def siyi_joypad(gp: OGamepad):

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
      ('Y', True):
          lambda s: s.inc(
              ButtonKind.PHOTO_COUNT
              if (s.cur_buttons['LB'] and s.cur_buttons['RB']) else ButtonKind.RECORD_COUNT
          ),
      'DPAD-Y': lambda s, v: s.set(ButtonKind.ZOOM, v),
      ('LT', 'RT'):
          lambda s, v: s.set(ButtonKind.CTRL, mixaxis(*v)),
      'LB':
          lambda s, v: s.set(
              ButtonKind.MODE, {
                  True: SiyiJoyMode.PITCH,
              }.get(v, SiyiJoyMode.YAW)
          ),
  }
  return configure_joy(gp, mp, {x: 0 for x in ButtonKind})


class SiyiServerParams(cmisc.PatchedModel):
  send_status_freq_hz: float = 1


class SiyiServer(cmisc.PatchedModel):
  th: tunnel.PyMsgTunnelData
  m: CameraManager
  params: SiyiServerParams = cmisc.Field(default_factory=SiyiServerParams)

  last_joy: SiyiJoyData = SiyiJoyData()

  def set_params(self, params):
    self.params = params

  def siyimessage2pyd(self, msg, desc='') -> comm_p2p.SiyiRawReq:
    return comm_p2p.SiyiRawReq(desc=desc, msg_id=type(msg)._base.ID, res=base64.b64encode(msg.pack()).decode())

  def siyireq2pyd(self, req, desc='') -> comm_p2p.SiyiRawReq:
    ans = self.m.handler.send_msg(req, wait_reply=True)
    return self.siyimessage2pyd(req.base.Res(**ans), desc)

  def get_status(self) -> list:
    return [
        self.siyireq2pyd(SiyiCmd_RequestCodecSpecs.Req(stream_type=0), '0'),
        self.siyireq2pyd(SiyiCmd_RequestCodecSpecs.Req(stream_type=1), '1'),
        self.siyireq2pyd(SiyiCmd_RequestGimbalInfo.Req()),
    ]

  @contextlib.contextmanager
  def enter(self) -> SiyiServer:
    self.th.add_handler(SiyiJoyData, self.recv_joy_data)
    self.th.add_handler(SiyiServerParams, self.set_params)
    self.th.tg.add_timer(
        lambda: self.params.send_status_freq_hz, lambda: self.th.push_msgs(self.get_status())
    )

    with self.m.enter(), self.th.enter():
      yield self

  @cmisc.logged_failsafe
  def recv_joy_data(self, joy: SiyiJoyData):
    if joy.photo_count != self.last_joy.photo_count:
      self.m.take_picture()
    if (joy.record_count - self.last_joy.record_count) & 1 != 0:
      self.m.toggle_video()

    def get_joyv(joy: SiyiJoyData, mode):
      return 0 if joy.mode != mode else joy.ctrl

    self.m.set_speed(get_joyv(joy, SiyiJoyMode.YAW), get_joyv(joy, SiyiJoyMode.PITCH))
    if joy.zoom != self.last_joy.zoom:
      self.m.rel_zoom(joy.zoom)
    self.last_joy = joy


class SiyiControllerParams(cmisc.PatchedModel):
  input_rate: float = 4


class SiyiController(cmisc.PatchedModel):
  gp: OGamepad
  th: tunnel.PyMsgTunnelData
  params: SiyiControllerParams = cmisc.pyd_f(SiyiControllerParams)

  siyi_data: A = cmisc.pyd_f(A)

  def handle_recv_req(self, msg: comm_p2p.SiyiRawReq):
    msg_typ: SiyiCmd_Metaclass = siyi_message_types[msg.msg_id]
    res = msg_typ.Res.unpack(base64.b64decode(msg.res))
    self.siyi_data[(msg_typ.__name__, msg.desc)] = res

  @contextlib.contextmanager
  def enter(self) -> SiyiServer:
    self.th.add_handler(comm_p2p.SiyiRawReq, self.handle_recv_req)


    def timer_joy():
      jd = SiyiJoyData(**{k.value: v for k, v in state.last.inputs.items()})

      self.siyi_data.joy = jd
      self.th.push_msg(jd)

    with siyi_joypad(self.gp) as state:
      self.th.tg.add_timer(lambda: self.params.input_rate, timer_joy)

      with self.th.enter():
        yield self


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
