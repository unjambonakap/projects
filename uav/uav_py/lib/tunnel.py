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
import contextlib

import time
import sys
import pydantic
import typing
import pydantic_settings
import pickle
from google.protobuf import json_format

os.environ['MAVLINK20'] = '1'

from pymavlink import mavutil
from pymavlink.dialects.v20 import ardupilotmega as ml
from pymavlink.dialects.v20.ardupilotmega import MAVLink
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


class ThreadGroup(cmisc.PatchedModel):
  group: list[threading.Thread] = pydantic.Field(default_factory=list)
  ev: threading.Event = pydantic.Field(default_factory=threading.Event)
  disable_async: bool = False
  asyncs: list = pydantic.Field(default_factory=list)
  context: cmisc.ExitStackWithPush = pydantic.Field(default_factory=cmisc.ExitStackWithPush)

  def run_async(self):
    el = asyncio.new_event_loop()
    el.run_until_complete(self.async_runner())

  def __post_init__(self):
    if not self.disable_async:
      self.add(threading.Thread(target=self.run_async))

  @contextlib.contextmanager
  def enter(self):
    with self.context:
      self.start()
      try:
        yield self
      finally:
        self.stop()
        self.join()

  def add_timer(self, get_freq, action):

    action = cmisc.logged_failsafe(action)

    async def timer_func():
      while not self.should_stop():
        freq = get_freq()
        await asyncio.sleep(1 / freq)
        action()

    self.asyncs.append(timer_func)

  def add_rich_monitor(self, get_freq, cb):
    import rich.live, rich.json
    live = rich.live.Live(refresh_per_second=10)
    self.context.pushs.append(live)
    self.add_timer(get_freq, lambda: live.update(rich.json.JSON(cb())))

  async def async_runner(self):
    glog.info('Async runner go')
    waitables = [cb() for cb in self.asyncs]
    await asyncio.gather(*waitables)
    glog.info('Async runner go')

  def add(self, th: threading.Thread):
    self.group.append(th)

  def should_stop(self) -> bool:
    return self.ev.is_set()

  def stop(self):
    self.ev.set()

  def start(self):
    [x.start() for x in self.group]

  def join(self):
    [x.join() for x in self.group]


class TunnelHelper(cmisc.PatchedModel):
  src: MavAddress
  dst: MavAddress
  seq_id: int = 0

  def get_mav(self, send=False):
    msg =  MAVLink(file=io.BytesIO, srcSystem=self.src.sys, srcComponent=self.src.comp)
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

  def recv(self, conn: Connection, ev: threading.Event) -> bytearray:
    mav = self.get_mav()
    mav.robust_parsing = True
    while not ev.is_set():
      need = mav.bytes_needed()
      if need > 0:
        buf = conn.recv_fixed_size(need)
      else:
        buf = b''
      glog.debug(f'Recv raw {buf}')
      #('BEFORE PARSE >> ', mav.buf, mav.buf_index, buf, need, len(buf))
      msgs: list[ardupilotmega.MAVLink_message] = mav.parse_buffer(buf)
      #print('AFTER parse', mav.buf, mav.buf_index, msgs)
      if msgs is None: continue

      content = bytearray()
      for msg in msgs:
        glog.info(f'Recv mavlink >> {type(msg)}')
        if isinstance(msg ,ml.MAVLink_bad_data):
          continue
        if self.is_good_msg(msg):
          content.extend(msg.payload[:msg.payload_length])
      if content: return content


class TunnelDataBase(cmisc.PatchedModel):
  conf: Conf
  f_enc: Connection
  th: TunnelHelper = None
  tg: ThreadGroup = cmisc.pyd_f(ThreadGroup)

  @contextlib.contextmanager
  def enter(self):
    with self.tg.enter():
      yield self

  def __post_init__(self):
    super().__post_init__()
    self.th = TunnelHelper(src=self.conf.src, dst=self.conf.dst)

    self.tg.add(threading.Thread(target=self.decode_thread))
    self.tg.add(threading.Thread(target=self.heartbeat_thread))

  def decode_thread(self):
    while not self.tg.should_stop():
      content = self.th.recv(self.f_enc, self.tg.ev)
      if content is None:
        break
      glog.info(f'outputting buf {content}')
      self.handle_decoded_msg(content)

  def heartbeat_thread(self):
    while not self.tg.should_stop():
      glog.info('Sending heartbeat')
      self.f_enc.send(self.th.get_heartbeat())
      time.sleep(1)

  def handle_plain_buf(self, buf):
    assert len(buf) <= kTunnelPayloadMaxSize, len(buf)
    self.f_enc.send(self.th.buf2mav(buf))

  def handle_decoded_msg(self, content):
    assert 0


class PyMsgTunnelData(TunnelDataBase):
  handlers: dict[typing.Type, list] = cmisc.pyd_f(lambda: cmisc.defaultdict(list))
  p2p: Proto2Pydantic = g_p2p

  def __post_init__(self):
    super().__post_init__()
    if not self.p2p.is_built:
      raise ValueError('p2p should be built')

  def add_handler(self, typ, cb):
    self.handlers[typ].append(cb)


  def decode_msg(self, buf):
    idx, = struct.unpack('<H', buf[:2])
    proto = self.p2p.id2proto[idx]
    msg = self.p2p.to_pyd(proto.FromString(buf[2:]))
    return msg

  def handle_decoded_msg(self, content):
    msg = self.decode_msg(content)
    handlers = self.handlers[type(msg)]
    if not handlers:
      glog.info(f'Message of type {type(msg)} has no handlers - nop')

    for handler in handlers:
      handler(msg)

  def push_msgs(self, msgs: list[pydantic.BaseModel]):
    [self.push_msg(x) for x in msgs]


  def encode_msg(self, msg):
    a = self.p2p.to_proto(msg)
    if self.p2p.to_pyd(a) != msg:
      pass

    buf = struct.pack('<H', self.p2p.pyd2id(type(msg))) + a.SerializeToString()
    return buf

  def push_msg(self, msg: pydantic.BaseModel):

    buf =self.encode_msg(msg)
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
