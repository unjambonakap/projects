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
import os
import struct
from chdrft.tube.file_like import FileLike
from chdrft.tube.connection import Connection
import threading
import contextlib

import time
import sys
import pydantic
import typing
import pydantic_settings
import pickle
from google.protobuf import json_format

os.environ['MAVLINK20'] = '1'
#os.environ['MAVLINK11'] = '1'

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as ml
from pymavlink.dialects.v20.ardupilotmega import MAVLink
from chdrft.utils.othreading import ThreadGroup
import io
import pymavlink.mavutil
import asyncio

global flags, cache
flags = None
cache = None


class Proto2Pydantic(cmisc.PatchedModel):
  proto2pyd: dict = cmisc.pyd_f(dict)
  pyd2proto: dict = cmisc.pyd_f(dict)
  proto2id: dict = cmisc.pyd_f(dict)
  id2proto: dict = cmisc.pyd_f(dict)

  pairs: list = cmisc.pyd_f(list)
  is_built: bool = False

  def build(self):
    assert not self.is_built
    self.is_built = True

    for i, (vproto, vpyd) in enumerate(self.pairs):
      self.proto2pyd[vproto] = vpyd
      self.pyd2proto[vpyd] = vproto

      self.id2proto[i] = vproto
      self.proto2id[vproto] = i

  def add(self, module_proto, module_pyd):
    assert not self.is_built
    for k in sorted(module_proto.__dict__.keys()):
      if k.startswith('__'): continue
      vproto = module_proto.__dict__[k]
      if not (vpyd := module_pyd.__dict__.get(k, None)): continue
      self.pairs.append((vproto, vpyd))

  def pyd2id(self, pyd):
    return self.proto2id[self.pyd2proto[pyd]]

  def to_proto(self, pyd: cmisc.BaseModel):

    tmp = json_format.ParseDict(pyd.dict(), self.pyd2proto[type(pyd)]())
    return tmp

  def to_pyd(self, proto) -> cmisc.BaseModel:
    cl = self.proto2pyd[type(proto)]
    ser_dict = json_format.MessageToDict(
        proto, use_integers_for_enums=True, preserving_proto_field_name=True
    )

    return cl(**ser_dict)


g_p2p = Proto2Pydantic()


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class Config:
  arbitrary_types_allowed = True


dc = pydantic.dataclasses.dataclass(config=Config)


class HandlerDesc(cmisc.PatchedModel):
  func: object = None
  absorb: bool = True
  oneshot: bool = False
  filter: object = None


class HandlersManager(cmisc.PatchedModel):
  handlers: dict[typing.Type, list[HandlerDesc]] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  verbose: bool = True
  backup_handler: object = None

  def remove_handler(self, typ, u):
    try:
      self.handlers[typ].remove(u)
    except ValueError:
      pass

  @contextlib.contextmanager
  def add_handler_oneshot(self, typ, cb, **kwargs):
    u = self.add_handler(typ, cb, **kwargs)
    try:
      yield
    finally:
      self.remove_handler(typ, u)

  def add_handler(self, typ, cb, **kwargs):
    u = HandlerDesc(func=cb, **kwargs)
    self.handlers[typ].append(u)
    return u

  def try_handle(self, handler, msg, key):
    if handler.filter is None or handler.filter(msg):
      try:
        handler.func(msg)
      except Exception as e:
        glog.error(f'FAILED {e} {msg}')
        raise
      if handler.oneshot:
        self.handlers[key].remove(handler)
      if handler.absorb: return True
    return False

  def handle(self, msg, key=None):
    if key is None:
      key = type(msg)

    for handler in self.handlers[key]:
      if self.try_handle(handler, msg, key): return
    for handler in self.handlers[None]:
      if self.try_handle(handler, msg, None): return
    glog.info(f'Message of type {key} has no handlers - nop ')
    if self.backup_handler:
      self.backup_handler(msg)


class MavAddress(cmisc.PatchedModel):
  sys: int
  comp: int


kTunnelPayloadType = 10
kTunnelPayloadMaxSize = 128


class Conf(pydantic_settings.BaseSettings):
  src: MavAddress
  dst: MavAddress

  def maybe_toggle(self, toggle: bool) -> Conf:
    if toggle:
      return Conf(src=self.dst, dst=self.src)
    return self


class MavlinkStreamDecoder(cmisc.PatchedModel):

  mav: MAVLink = None

  def __post_init__(self):
    self.mav = MAVLink(file=io.BytesIO)
    self.mav.robust_parsing = True

  def push(self, data: bytes) -> list[ardupilotmega.MAVLink_message]:
    return self.mav.parse_buffer(data)


class MavlinkEndpoint(cmisc.ContextModel):
  conn: Connection
  recv_handler: HandlersManager = cmisc.pyd_f(HandlersManager)
  tg: ThreadGroup = cmisc.pyd_f(ThreadGroup)
  m: MavlinkStreamDecoder = cmisc.pyd_f(MavlinkStreamDecoder)

  def __post_init__(self):
    super().__post_init__()
    self.ctx_push(self.tg.enter())

    self.tg.add(func=self.decode_thread)

  def recv(self, conn: Connection, ev: threading.Event) -> bytearray:

    while not ev.is_set():
      need = self.m.mav.bytes_needed()
      if need > 0:
        buf = conn.recv_fixed_size(need, timeout=cmisc.Timeout.from_event(0.3, ev))
      else:
        buf = b''
      glog.debug(f'Recv raw {buf}')

      msgs = self.m.push(buf)
      if msgs is None: continue

      for msg in msgs:
        if isinstance(msg, ml.MAVLink_bad_data):
          glog.info(f'Bad mavlink >> {type(msg)} {msg.reason}')
          continue
        self.recv_handler.handle(msg)

  @cmisc.logged_failsafe
  def decode_thread(self):
    while not self.tg.should_stop():
      self.recv(self.conn, self.tg.ev)

  def send(self, msg: bytes):
    self.conn.send(msg)


class TunnelHelper(cmisc.PatchedModel):
  src: MavAddress
  dst: MavAddress
  seq_id: int = 0

  @classmethod
  def FromConf(cls, conf: Conf):
    return cls(src=conf.src, dst=conf.dst)

  def get_mav(self, send=False):
    msg = MAVLink(file=io.BytesIO, srcSystem=self.src.sys, srcComponent=self.src.comp)
    if send:
      msg.seq = self.seq_id % 256
      self.seq_id += 1
    return msg

  def get_heartbeat(self):
    mav = self.get_mav(send=True)
    mav.file = io.BytesIO()
    mav.heartbeat_send(
        type=ml.MAV_TYPE_ONBOARD_CONTROLLER,
        autopilot=ml.MAV_AUTOPILOT_INVALID,
        base_mode=0,
        custom_mode=0,
        system_status=ml.MAV_STATE_ACTIVE,
    )
    return mav.file.getvalue()

  def buf2mav(self, data) -> bytes:
    mav = self.get_mav(send=True)
    mav.file = io.BytesIO()
    mav.tunnel_send(
        target_system=self.dst.sys,
        target_component=self.dst.comp,
        payload_type=kTunnelPayloadType,
        payload_length=len(data),
        payload=bytes(data) + bytes([0] * (kTunnelPayloadMaxSize - len(data)))
    )
    return mav.file.getvalue()

  def is_good_msg(self, msg: ardupilotmega.MAVLink_message) -> bool:
    if (msg.get_srcSystem(), msg.get_srcComponent()) != (self.dst.sys, self.dst.comp):
      return False
    if msg.get_msgId() != ml.MAVLINK_MSG_ID_TUNNEL or msg.payload_type != kTunnelPayloadType:
      return False
    if (msg.target_system, msg.target_component) != (self.src.sys, self.src.comp):
      return False
    return True


class TunnelDataBase(cmisc.ContextModel):
  conf: Conf
  th: TunnelHelper = None
  ep: MavlinkEndpoint
  tg: ThreadGroup = cmisc.pyd_f(ThreadGroup)

  def __post_init__(self):
    super().__post_init__()
    self.th = TunnelHelper.FromConf(self.conf)
    self.tg.add(threading.Thread(target=self.heartbeat_thread))
    self.ep.recv_handler.add_handler(
        ml.MAVLink_tunnel_message, self.process_msg, filter=self.th.is_good_msg
    )

    self.ctx_push(self.tg.enter())
    self.ctx_push(self.ep)

  def process_msg(self, msg):
    self.handle_decoded_msg(bytes(msg.payload[:msg.payload_length]))

  def heartbeat_thread(self):
    while not self.tg.should_stop():
      glog.info('Sending heartbeat')
      self.ep.send(self.th.get_heartbeat())
      time.sleep(1)

  def handle_plain_buf(self, buf):
    assert len(buf) <= kTunnelPayloadMaxSize, len(buf)
    self.ep.send(self.th.buf2mav(buf))

  def handle_decoded_msg(self, content):
    assert 0


class PyMsgTunnelData(TunnelDataBase):
  hm: HandlersManager = cmisc.pyd_f(HandlersManager)
  p2p: Proto2Pydantic = g_p2p

  def __post_init__(self):
    super().__post_init__()
    if not self.p2p.is_built:
      raise ValueError('p2p should be built')

  def decode_msg(self, buf):
    idx, = struct.unpack('<H', buf[:2])
    proto = self.p2p.id2proto[idx]
    msg = self.p2p.to_pyd(proto.FromString(buf[2:]))
    return msg

  def handle_decoded_msg(self, content):
    msg = self.decode_msg(content)
    self.hm.handle(msg)

  def push_msgs(self, msgs: list[pydantic.BaseModel]):
    [self.push_msg(x) for x in msgs]

  def encode_msg(self, msg):
    a = self.p2p.to_proto(msg)
    if self.p2p.to_pyd(a) != msg:
      pass

    buf = struct.pack('<H', self.p2p.pyd2id(type(msg))) + a.SerializeToString()
    return buf

  def push_msg(self, msg: pydantic.BaseModel):

    buf = self.encode_msg(msg)
    res = self.decode_msg(buf)
    self.handle_plain_buf(buf)


class RawTunnelData(TunnelDataBase):
  f_plain: Connection

  def handle_decoded_msg(self, content):
    self.f_plain.send(content)

  def encode_thread(self):
    print('Run encode thread')
    while not self.tg.should_stop():
      try:
        buf = self.f_plain.recv_fixed_size(kTunnelPayloadMaxSize, timeout=0.01)
        if buf is None: break
      except cmisc.TimeoutException:
        buf = self.f_plain.take_current()
      if not buf:
        continue
      glog.info(f'Sending buf {buf}')
      self.handle_plain_buf(buf)

  def __post_init__(self):
    super().__post_init__()
    self.tg.add(threading.Thread(target=self.encode_thread))


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
