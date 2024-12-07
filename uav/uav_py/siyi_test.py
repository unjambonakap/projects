#!/usr/bin/env python

from __future__ import annotations

__package__ = "chdrft.projects.uav_py"
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
from .lib.siyi import *
from .lib.proto import comm_p2p, comm_pb2
import subprocess as sp
from pydantic_core import from_json

if not g_env.slim:
  from chdrft.display.service import g_plot_service as oplt

import Pyro5 as pyro
import Pyro5.server, Pyro5.nameserver, Pyro5.client
try:
  import chdrft.ros.base as ros_base
except:
  glog.error('failed to import chdrft.ros.base')
  pass

import rich.json
import rich.live

global flags, cache
flags = None
cache = None


class ServiceType(enum.Enum):
  SERVER = 'server'
  CLIENT = 'client'


def args(parser):
  clist = CmdsList()
  parser.add_argument('--target', default='192.168.1.25')
  parser.add_argument('--target-port', type=int, default=37260)
  parser.add_argument('--sync-folder', default='./data/')
  parser.add_argument('--ros-pub-id', default='main')
  parser.add_argument('--format', action='store_true')
  parser.add_argument('--reset', action='store_true')
  parser.add_argument('--export-ros', action='store_true')
  parser.add_argument('--center', action='store_true')
  parser.add_argument('--resolution', type=lambda x: SiyiResolution[x])
  parser.add_argument('--set-mode', type=lambda x: SiyiGimbalMotionMode[x])
  parser.add_argument('--target-speed', type=str)
  parser.add_argument('--stream-type', type=int)
  parser.add_argument('--seq', type=int, default=0)
  parser.add_argument('--no-cam', action='store_true')
  parser.add_argument('--no-gamepad', action='store_true')
  parser.add_argument('--no-live', action='store_true')
  parser.add_argument('--video', type=int, default=0)
  parser.add_argument('--video-duration', type=int)
  parser.add_argument('--service-type', type=lambda x: ServiceType(x))
  parser.add_argument('--endpoint', type=str)

  parser.add_argument('--start-pyro', action='store_true')
  parser.add_argument('--pyro-target', type=str)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


crcx = crcmod.mkCrcFun(0x11021, initCrc=0, rev=False)
kSiyiOffsetData = 8
kPyroNSPort = 11111
pyro.config.NS_HOST = 'localhost'
pyro.config.NS_PORT = kPyroNSPort

if 0:
  a = bytes.fromhex('55 66 01 00 00 00 00 19')
  print(hex(crcx(a)))
  print(a)
  assert 0
  import io
  a = io.StringIO()
  print(crcmod.Crc(0x11021, rev=False).generateCode('aa', a))
  print(a.getvalue())


def t1(ctx):
  conn = Connection(ctx.target, ctx.target_port, udp=True, udp_bind_port=12345)
  hx = SiyiHandler(conn=conn, seq=ctx.seq)
  a = hx.pack(SiyiCmd_Heartbeat.Req(), 0)
  print(a.hex(' '))
  a = hx.pack(SiyiCmd_Firmware.Req(), 0)
  print(a.hex(' '))
  b = hx.pack(SiyiCmd_GimbalRotSpeed.Req(yaw=1, pitch=1), 1)
  print(b.hex(' '))
  b = hx.pack(SiyiCmd_GimbalAngle.Req(yaw=-90, pitch=0), 1)
  b = hx.pack(SiyiCmd_GimbalAngle.Req(yaw=-90, pitch=0), 1)
  print(b.hex(' '))
  b = hx.pack(SiyiCmd_SetZoom.Req(val=4.5), 0x10)
  print(b.hex(' '))
  print(hx.pack(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.RECORDING_TOGGLE), 0).hex(' '))
  print(ctypes.sizeof(SiyiCmd_RequestCodecSpecs.Res))
  #b = bytes.fromhex('55 66 01 04 00 00 00 0e 00 00 ff a6 3b 11')
  #print(hx.unpack(b, is_req=True))
  #print(hx.unpack(b, is_req=True).obj.raw.yaw)

  # cannot set utc time twice??
  #
  app.stack.enter_context(hx.enter())
  #print(hx.send_msg(SiyiCmd_SetUtcTime.Req(timestamp=int(datetime.datetime.utcnow().timestamp())* 1000), wait_reply=True))
  print(hx.send_msg(SiyiCmd_Firmware.Req(), wait_reply=True))
  print(hx.send_msg(SiyiCmd_RequestGimbalInfo.Req(), wait_reply=True))
  print('strem1', hx.send_msg(SiyiCmd_RequestCodecSpecs.Req(stream_type=1), wait_reply=True))
  print('stream0', hx.send_msg(SiyiCmd_RequestCodecSpecs.Req(stream_type=0), wait_reply=True))
  return
  print(hx.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.MOTION_FOLLOW)))
  print(hx.send_msg(SiyiCmd_GimbalAngle.Req(yaw=90, pitch=00), wait_reply=True))
  time.sleep(1)
  print(hx.send_msg(SiyiCmd_RequestGimbalAttitude.Req(), wait_reply=True))
  if 0:
    for i in range(100):
      print(
          hx.send_msg(
              SiyiCmd_SendFCAttitude.Req(yaw=0, roll=0, pitch=0, v_yaw=0, v_roll=0, v_pitch=0)
          )
      )
    time.sleep(0.05)
  return
  #print(hx.send_msg(SiyiCmd_SetZoom().Req(val=3), wait_reply=True))

  for i in reversed(np.linspace(1, 6, 51)):
    print(hx.send_msg(SiyiCmd_SetZoom().Req(val=i), wait_reply=True))
    time.sleep(0.3)
    print(hx.send_msg(SiyiCmd_GetZoom().Req(), wait_reply=True))
  print('max zoom', hx.send_msg(SiyiCmd_GetMaxZoom().Req(), wait_reply=True))
  return
  print(
      hx.send_msg(
          SiyiCmd_RequestGimbalAttitudeStream.Req(data_type=1, data_freq=SiyiDataFreq.OFF),
          wait_reply=False
      )
  )
  print(hx.send_msg(SiyiCmd_Firmware.Req(), wait_reply=True))
  time.sleep(1)
  #print(hx.send_msg(SiyiCmd_FormatSD.Req(), wait_reply=True))
  return
  print(hx.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.MOTION_FOLLOW)))
  print(
      hx.send_msg(
          SiyiCmd_SendCodecSpecs.Req(
              stream_type=0,
              video_enc_type=2,
              resolution_l=1980,
              resolution_h=1020,
              video_bitrate=20000
          ),
          wait_reply=True
      )
  )
  print(hx.send_msg(SiyiCmd_RequestCodecSpecs.Req(), wait_reply=True))
  print(hx.send_msg(SiyiCmd_Center.Req(center_pos=1), wait_reply=True))
  time.sleep(0.1)

  #print(hx.send_msg(SiyiCmd_SoftReset.Req(gimbal_reset=1, cam_reset=0), wait_reply=True))
  print(hx.send_msg(SiyiCmd_RequestCodecSpecs.Req(), wait_reply=True))
  print(
      hx.send_msg(
          SiyiCmd_SendFCData.Req(
              time_since_boot=1000, lat=10000, lon=200000, alt=300, alt_ell=4000, vn=1, ve=2, vd=3
          ),
          wait_reply=False
      )
  )
  print(hx.send_msg(SiyiCmd_GimbalRotSpeed.Req(yaw=0.1, pitch=0.05), wait_reply=True))
  if 1:
    for i in range(100):
      t0 = time.time()
      hx.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.PICTURE_TAKE))
      ff = hx.wait_msg(SiyiCmd_FuncFeedback.ID)
      ell = time.time() - t0
      print(ff.info, ell)
    time.sleep(2)
    return

  print(hx.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.RECORDING_TOGGLE)))
  for i in range(10):
    print(
        hx.send_msg(
            SiyiCmd_SendFCData.Req(
                time_since_boot=1000,
                lat=10000 * i,
                lon=200000 * i,
                alt=300 * i,
                alt_ell=4000,
                vn=1,
                ve=2,
                vd=3
            ),
            wait_reply=False
        )
    )
    #print(hx.send_msg(SiyiCmd_GetGimbalMode.Req(), wait_reply=True))
    #print(hx.send_msg(SiyiCmd_GetGimbalMode.Req(), wait_reply=True))
    #print(hx.send_msg(SiyiCmd_GimbalAngle.Req(yaw=0, pitch=0), wait_reply=True))
    time.sleep(0.1)
  #print(hx.send_msg(SiyiCmd_SetAngleSingleAxis.Req(angle=-30, axis=1), wait_reply=True))
  #print(hx.send_msg(SiyiCmd_SetUtcTime.Req(timestamp=0), wait_reply=True))
  print(hx.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.RECORDING_TOGGLE)))

  print('DOOOONE')

  time.sleep(1)


def test_http(ctx):
  hc = SiyiHTTPClient(host=ctx.target)
  is_video = ctx.video
  print(hc.get_files(is_video))


def delta_sync_cam_files_init(ctx):
  sh = SyncHelper(host=ctx.target, sync_folder=ctx.sync_folder, video=False)
  sh.init()


def delta_sync_cam_files_sync(ctx):
  sh = SyncHelper(host=ctx.target, sync_folder=ctx.sync_folder, video=False)
  sh.sync()


def test_sync_cam(ctx):
  sh = SyncHelper(host=ctx.target, sync_folder=ctx.sync_folder, video=False)
  print(sh.cur_state.json())
  sh.init()
  m: CameraManager = ctx.m
  with m.enter():
    m.take_picture()
    time.sleep(1)

  sh.sync()


def cam_list_files(ctx):
  assert ctx.sync_folder
  hc = SiyiHTTPClient(host=ctx.target)
  is_video = ctx.video
  for rel, f in hc.get_files(is_video).items():
    print(rel, f)


def whole_sync_cam_files(ctx):
  assert ctx.sync_folder
  hc = SiyiHTTPClient(host=ctx.target)
  is_video = ctx.video
  for rel, f in hc.get_files(is_video).items():
    hc.maybe_download(os.path.join(ctx.sync_folder, rel), f.url)


def action(ctx):
  m: CameraManager = ctx.m
  with m.enter():
    print(m.handler.send_msg(SiyiCmd_Firmware.Req(), wait_reply=True))
    if ctx.format:
      input('Sure format??')
      m.format()
    if ctx.reset:
      m.reset()
      return
    if ctx.center:
      m.center()
    print(m.handler.send_msg(SiyiCmd_GimbalAngle.Req(yaw=0, pitch=00), wait_reply=True))

    print('spec0', m.get_codec_specs(0))
    print('spec1', m.get_codec_specs(1))
    if ctx.resolution:
      m.set_resolution(ctx.resolution, stream_type=ctx.stream_type)
      print(m.get_codec_specs(ctx.stream_type))

    if ctx.set_mode is not None:
      m.set_mode(ctx.set_mode)

    if ctx.target_speed:
      m.set_speed(*eval(ctx.target_speed))


"""

yaw: main axis --> camera 2 body
pitch/roll: defined with accelerometer G -> camera to world

follow: try to fix pitch/roll to given value,  same for yaw
lock: try to fix pitch/roll to given value, try to maintain yaw in cam2world
fpv: maintain everything in cam2body

"""


def dump_att(ctx):
  m = ctx.m
  with m.enter():
    print(m.handler.send_msg(SiyiCmd_PhotoVideoAction.Req(action=SiyiActionType.MOTION_FPV)))
    print(m.handler.send_msg(SiyiCmd_GimbalAngle.Req(yaw=45, pitch=0), wait_reply=True))
    while True:
      print(m.handler.send_msg(SiyiCmd_GimbalAngle.Req(yaw=45, pitch=0), wait_reply=True))
      print(m.get_attitude())
      time.sleep(0.1)


def test_zoom(ctx):
  m: CameraManager = ctx.m
  with m.enter():
    m.rel_zoom(1)
    m.rel_zoom(1)
    m.rel_zoom(1)
    m.rel_zoom(1)
    m.rel_zoom(1)
    m.rel_zoom(1)
    time.sleep(0.1)
    m.rel_zoom(0)
    m.rel_zoom(0)
    m.rel_zoom(0)
    m.rel_zoom(0)
    m.rel_zoom(0)
    return
    with m.hc.delta().enter() as delta:
      for j in range(2):
        for i in np.linspace(1, 6, 10):
          print(m.get_zoom(), i)
          m.zoom(i)
          time.sleep(0.1)
          m.take_picture()

    imgs = [ImageData.Make(m.hc.file_content(x.url), inv=1) for x in delta.diff.values()]
    oplt.plot(A(images=ImageGrid(images=imgs).get_images()))

    print(delta.diff)
    input()


def test_img1(ctx):
  hc = SiyiHTTPClient(host=ctx.target)
  content = hc.file_content('http://192.168.1.25:82/photo/101SIYI_IMG/IMG_0033.jpg')
  img = ImageData.Make(content, inv=1)
  oplt.plot(A(images=[img, img]))
  oplt.plot(A(images=[img, img]))
  input()


def test_img2(ctx):
  m: CameraManager = ctx.m
  with m.enter():
    with m.hc.delta().enter() as delta:
      m.set_resolution(ctx.resolution, stream_type=0)
      m.take_picture()
    for k, v in delta.diff.items():
      m.hc.maybe_download(os.path.join(ctx.sync_folder, k), v.url)


def test_vid1(ctx):
  m: CameraManager = ctx.m
  with m.enter():
    with m.hc.delta().enter() as delta:
      m.set_resolution(ctx.resolution, stream_type=0)
      res = run_ev_protected_op(m.record_video, ctx.video_duration, lambda idx: idx * 10)
      print(res)
    for k, v in delta.diff.items():
      m.hc.maybe_download(os.path.join(ctx.sync_folder, k), v.url)


def test_controller(ctx):
  gp = OGamepad()

  m: CameraManager = ctx.m
  rctx = None
  if ctx.export_ros:
    rctx = ros_base.RospyContext(subid=ctx.ros_pub_id)
    app.stack.enter_context(rctx.enter())
    rctx.create_publisher(f'joy', SiyiJoyData)

  if m is None:
    with siyi_joypad(gp) as state:
      debug_controller_state(state, lambda x: SiyiJoyData.from_inputs(x.inputs).json())
      return

  with rich.live.Live(refresh_per_second=10) as live, m.enter() as m, siyi_joypad(gp) as state:
    for x in list(ButtonKind):
      state.last.set(x, 0)

    m.set_mode(SiyiGimbalMotionMode.FOLLOW)
    if ctx.center:
      m.center()
      time.sleep(1)

    prev = dict(state.last.inputs)
    while True:
      if not ctx.no_live:
        live.update(rich.json.JSON(cmisc.json_dumps(state.last.inputs)))

      jd = SiyiJoyData(**{k.value: v for k, v in state.last.inputs.items()})
      if rctx is not None:
        rctx.cbs.joy(jd)

      glog.warn('Start')
      m.recv_joy_data(jd)
      glog.warn('end')
      time.sleep(0.1)


def test(ctx):
  sys.path.append('/home/benoit/repos/drone/siyi_sdk')
  from siyi_sdk import SIYISDK
  print("test")
  cam = SIYISDK(server_ip="192.168.1.25", port=37260, debug=False)
  if not cam.connect():
    print("No connection ")
    exit(1)

  print(cam.requestGimbalInfo())
  print("Recording state: ", cam.getRecordingState())
  print("Motion mode: ", cam.getMotionMode())
  print("Mounting direction: ", cam.getMountingDirection())
  print(cam.getRecordingState())
  cam.requestSetAngles(30, -30)
  return
  cam.requestRecording()
  time.sleep(3)
  print(cam.getRecordingState())
  cam.requestSetAngles(100, 30)
  time.sleep(3)
  cam.requestRecording()
  time.sleep(0.1)
  print(cam.getRecordingState())
  #cam.requestPhoto()
  #cam.setGimbalRotation(30, 200)

  print("Attitude (yaw,pitch,roll) eg:", cam.getAttitude())

  cam.disconnect()


def test_tunnel_mock(ctx):
  f1 = '/tmp/test1.sock'
  f2 = '/tmp/test2.sock'

  proc = sp.Popen(f'socat UNIX-LISTEN:{f1} UNIX-LISTEN:{f2}', shell=True)
  time.sleep(1.3)

  def cleanup():
    proc.kill()
    cmisc.failsafe(lambda: os.remove(f1))
    cmisc.failsafe(lambda: os.remove(f2))

  c1 = app.stack.enter_context(tunnel.Connection(f1))
  time.sleep(0.3)
  c2 = app.stack.enter_context(tunnel.Connection(f2))

  conf = tunnel.Conf(
      **cmisc.yaml_load_custom('{src: {sys: 10, comp: 11}, dst: {sys: 20, comp: 21}}')
  )

  tunnel.g_p2p.add(comm_pb2, comm_p2p)
  tunnel.g_p2p.build()
  th1 = tunnel.PyMsgTunnelData(conf=conf, f_enc=c1)
  th2 = tunnel.PyMsgTunnelData(conf=conf.maybe_toggle(True), f_enc=c2)
  drone_server = SiyiServer(th=th2, m=CameraManager(mock=True))
  #drone_server.th.tg.add_rich_monitor(lambda: 10, lambda: drone_server.last_joy.json())

  gp = OGamepad()
  sc = SiyiController(gp=gp, th=th1)
  sc.th.tg.add_rich_monitor(lambda: 10, lambda: sc.siyi_data.to_json())

  with sc, drone_server:
    input('DONE ?')


kConfBase = tunnel.Conf(
    **cmisc.yaml_load_custom('{src: {sys: 10, comp: 11}, dst: {sys: 20, comp: 21}}')
)


class GlobalParams(tunnel.pydantic_settings.BaseSettings):
  server_params: SiyiServerParams

def make_service(th: tunnel.PyMsgTunnelData, ctx):
  if ctx.service_type is ServiceType.SERVER:
    x = tunnel.pydantic_settings.YamlConfigSettingsSource(GlobalParams, './config.yaml')()
    print(x)
    conf = GlobalParams(**x)
    ss = SiyiServer(th=th, m=ctx.m, params=conf.server_params)
    if not ctx.no_live:
      ss.th.tg.add_rich_monitor(lambda: 10, lambda: ss.siyi_data.to_json())

    if ctx.start_pyro:
      ss.th.tg.add(func=lambda: pyro.nameserver.start_ns_loop(port=kPyroNSPort, host='0.0.0.0'))
      ss.th.tg.add(
          func=lambda: pyro.server.Daemon.
          serveSimple({
              ss: 'pyro.siyiserver',
              ctx.m: 'pyro.cameramanager'
          })
      )
    return ss
  else:

    if ctx.no_gamepad: gp = None
    else: gp = OGamepad()
    sc = SiyiController(gp=gp, th=th)

    th.ep.recv_handler.add_handler(None, glog.info)

    if ctx.export_ros:
      rc = ctx.rctx = ros_base.RospyContext(subid=ctx.ros_pub_id)
      rc.create_publisher('joy', SiyiJoyData)
      sc.th.tg.add_timer(lambda: 10, lambda: rc.cbs.joy(sc.siyi_data.joy))
      th.ep.recv_handler.add_handler(
          ros_base.mavlink_msgs.MAVLink_radio_status_message,
          rc.create_publisher('radio', ros_base.mavlink_msgs.MAVLink_radio_status_message)
      )
      sc.ctx_push(rc.enter(), prepend=True)

    if not ctx.no_live:
      sc.th.tg.add_rich_monitor(lambda: 10, lambda: sc.siyi_data.to_json())
    return sc


def run_endpoint(ctx):
  assert ctx.endpoint and ctx.service_type, ctx
  f1 = f'/tmp/test_py_{ctx.service_type.value}.sock'

  # TODO: remove this - use get-smart_tube
  proc = sp.Popen(f'socat UNIX-LISTEN:{f1} {ctx.endpoint}', shell=True)
  time.sleep(1.3)

  def cleanup():
    proc.kill()
    cmisc.failsafe(lambda: os.remove(f1))

  c1 = app.stack.enter_context(tunnel.Connection(f1))
  time.sleep(0.1)

  ep = tunnel.MavlinkEndpoint(conn=c1)
  tunnel.g_p2p.add(comm_pb2, comm_p2p)
  tunnel.g_p2p.build()
  th1 = tunnel.PyMsgTunnelData(
      conf=kConfBase.maybe_toggle(ctx.service_type is ServiceType.SERVER),
      ep=ep,
  )
  sx = make_service(th1, ctx)

  ep.recv_handler.add_handler(None, glog.info)

  app.stack.enter_context(sx)
  if ctx.pyro_target:
    m = pyro.client.Proxy('PYRONAME:pyro.cameramanager')
  app.wait_or_interactive()

  print('Done test')


def interactive(ctx):
  if ctx.pyro_target:
    m = pyro.client.Proxy('PYRONAME:pyro.cameramanager')
  else:
    m: CameraManager = ctx.m

    app.global_context.enter_context(m.enter())
    print(m.get_firmware())

  app.wait_or_interactive()
  #%%

  #%%


def test_proto(ctx):

  p2p = tunnel.Proto2Pydantic()
  p2p.add(comm_pb2, comm_p2p)
  p2p.build()
  a = comm_p2p.SiyiJoyData(mode=comm_p2p.SiyiJoyMode.PITCH, ctrl=0, photo_count=3, record_count=12)
  b = p2p.to_proto(a)
  print(b)
  print(json_format.MessageToDict(b))
  print(json_format.MessageToJson(b))

  c = p2p.to_pyd(b)


def main():
  ctx = A()
  glog.debug(cmisc.json_dumps(flags))

  if not flags.no_cam:
    ctx.m = CameraManager(
        target_hostname=flags.target, target_port=flags.target_port, seq=flags.seq
    )
  else:
    ctx.m = CameraManager(mock=True)
  ActionHandler.Run(ctx)


app()
