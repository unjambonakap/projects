#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.config.env import g_env
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import A
import glog
import numpy as np
from pydantic import Field, BaseModel
from chdrft.utils.path import FileFormatHelper
import subprocess as sp
import time
import chdrft.projects.uav_py.lib.tunnel as tunnel
import chdrft.projects.uav_py.mavlink_utils as mavlink_utils
import chdrft.tube.tube_all as tube
import pydantic_settings
import threading
import chdrft_cel.chdrft_cel
import chdrft.utils.othreading as othreading
import chdrft.utils.rx_helpers as rx_helpers
import asyncio
import collections
import typing
import zmq
import struct
from chdrft.dsp.datafile import DynamicDataset, Dataset
import chdrft.display.plot_req as oplot_req
import numpy.random

import contextlib

try:
  import chdrft.ros.base as ros_base
except Exception as e:
  glog.error('failed to import chdrft.ros.base')
  cmisc.tb.print_exc(e)

global flags, cache
flags = None
cache = None


def get_time() -> float:
  return time.clock_gettime(time.CLOCK_REALTIME)


class SysClock(cmisc.PatchedModel):
  speedup: float = 1
  start_time: float = cmisc.pyd_f(get_time)
  start_boot: float = cmisc.pyd_f(time.monotonic)

  def t(self):
    return self.start_time + (time.monotonic() - self.start_boot) * self.speedup


class EvaluatorWrapper(cmisc.PatchedModel):
  query: str

  @cmisc.cached_property
  def evaluator(self) -> chdrft_cel.chdrft_cel.CelEvaluator:
    return chdrft_cel.chdrft_cel.CreateEvaluator(self.query)

  def get_bool(self, data):
    return self.evaluator.EvaluateToBool(cmisc.json_dumps(data))

  def get_pii(self, data):
    return self.evaluator.EvaluateToPII(cmisc.json_dumps(data))


class ResourceItemDesc(cmisc.PatchedModel):
  type: str
  name: str
  desc: str
  path: str = ''


class ResourceItem(cmisc.PatchedModel):
  obj: object
  desc: ResourceItemDesc


def transformed_oset(
    input: rx_helpers.ObservableSet, to_output, to_input
) -> rx_helpers.ObservableSet:
  res = rx_helpers.ObservableSet()
  i2o = dict()

  def add(x):
    y = to_output(x)
    assert x not in i2o
    i2o[x] = y
    res.add(y)

  def remove(x):
    y = i2o[x]
    del i2o[x]
    res.remove(y)

  input.subscribe(add, remove)
  return res


class ResourceRegister(cmisc.PatchedModel):
  oset: rx_helpers.ObservableSet

  @classmethod
  def FromItems(cls, data: rx_helpers.ObservableSet):
    return ResourceRegister(oset=data)

  def search(self, query: str) -> cmisc.Queryable:
    ev = EvaluatorWrapper(query=query)
    return cmisc.asq_query([x for x in self.oset.data if ev.get_bool(x.desc)])


g_sysc = SysClock()


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


class MonitorParams(pydantic_settings.BaseSettings):
  t1_uri: str
  t2_uri: str


class TunnelView(cmisc.PatchedModel):
  self: tube.Tube
  peer: tube.Tube

  def toggle(self) -> TunnelView:
    return TunnelView(self=self.peer, peer=self.self)


class TunnelMonitor(cmisc.ContextModel):
  t1: tube.Tube
  t2: tube.Tube
  tg: othreading.ThreadGroup = cmisc.pyd_f(othreading.ThreadGroup)
  pusher: object = None
  subject: rx_helprs.WrapRX = cmisc.pyd_f(lambda: rx_helpers.WrapRX(rx_helpers.rx.subject.Subject))

  def get_view(self, id) -> TunnelView:
    v = TunnelView(self=self.t1, peer=self.t2)
    if id: v = v.toggle()
    return v

  def __post_init__(self):
    super().__post_init__()
    self.tg.add(func=cmisc.functools.partial(self.recv_thread, 0))
    self.tg.add(func=cmisc.functools.partial(self.recv_thread, 1))

    self.ctx_push(self.tg.enter())
    self.ctx_push(self.t1)
    self.ctx_push(self.t2)

  def recv_thread(self, id: int, ev: threading.Event):
    v = self.get_view(id)
    while not ev.is_set():
      rx = v.self.recv(1024, cmisc.Timeout.from_event(1e-2, ev))
      if rx and self.pusher is not None:
        self.pusher(id, rx)

      v.peer.send(rx)


def smerge_dicts(a, b):
  a.update(b)
  return a


def mavlink_sniff():
  d1 = tunnel.MavlinkStreamDecoder()
  d2 = tunnel.MavlinkStreamDecoder()

  def tsf(data):
    id, buf = data
    dx: tunnel.MavlinkStreamDecoder = [d1, d2][id]
    yield from [
        smerge_dicts(mavlink_utils.mavlink_message_to_dict(msg), dict(direction=id))
        for msg in dx.push(buf)
    ]

  return tsf


class MavlinkObsSchema(cmisc.PatchedModel):
  t: float
  mlen: int
  seq: int
  msg_id: int
  src_sys: int
  src_comp: int
  dst_sys: int
  dst_comp: int

  @classmethod
  def from_sniff(cls, msg: dict) -> MavlinkObsSchema:
    return cls(
        t=g_sysc.t(),
        mlen=msg['header']['len'],
        msg_id=msg['header']['msg_id'],
        seq=msg['header']['seq'],
        src_sys=msg['header']['src_sys'],
        src_comp=msg['header']['src_comp'],
        dst_sys=msg['data'].get('target_system', -1),
        dst_comp=msg['data'].get('target_component', -1)
    )


class SyncPusherConf(cmisc.PatchedModel):
  rate_hz: float
  name: str
  desc: str
  data_extractor: typing.Callable[[object], float]


class SyncPusher(cmisc.PatchedModel):
  params: SyncPusherConf
  cur_step: int = None
  cur_bucket: float = 0
  pub: rx_helpers.WrapRX = cmisc.pyd_f(rx_helpers.WrapRX.Subject)

  def as_resource_item(self) -> ResourceItem:
    return ResourceItem(
        obj=self,
        desc=ResourceItemDesc(name=self.params.name, desc=self.params.desc, type='SyncPusher')
    )

  def process_until_time(self, t_sec: float):
    step = int(t_sec * self.params.rate_hz)

    if self.cur_step is None: self.cur_step = step
    assert self.cur_step <= step
    while self.cur_step < step:
      self.pub.on_next((self.cur_step / self.params.rate_hz, self.cur_bucket))
      self.cur_step += 1
      self.cur_bucket = 0
    self.cur_step = step

  async def timer_func(self, ev: threading.Event):
    while not ev.is_set():
      self.process_until_time(g_sysc.t())
      await asyncio.sleep(1 / self.params.rate_hz)

  def proc(self, m: object):
    self.process_until_time(m.t)
    self.cur_bucket += self.params.data_extractor(m)


class FilterDispatcher(cmisc.PatchedModel):
  target_syncs: typing.Callable[[object], SyncPusher] = None
  pushers: rx_helpers.ObservableSet = cmisc.pyd_f(rx_helpers.ObservableSet)

  def add_pusher(self, pusher: SyncPusher):
    self.pushers.add(pusher)

  def __call__(self, msg):
    pushers = self.target_syncs(msg)
    self.pushers |= set(pushers)
    for sync in pushers:
      sync.proc(msg)


class FilterParams(cmisc.PatchedModel):
  rate_hz: float = 10
  name: str
  name_register: cmisc.NameRegister = cmisc.pyd_f(cmisc.NameRegister)

  def mode_str(self, mode: int) -> str:
    return ['count', 'bitrate'][mode]

  def mode_extractor(self, mode: int):
    if mode: return lambda x: x.mlen / self.rate_hz
    else: return lambda x: 1

  def make_pusher(self, subname, mode, desc):
    return SyncPusher(
        params=SyncPusherConf(
            name=f'{self.name}/{subname}/{self.mode_str(mode)}',
            desc=desc,
            rate_hz=self.rate_hz,
            data_extractor=self.mode_extractor(mode)
        )
    )

  def create_mav_filter_cond(self, cond, with_len=False, with_bitrate=False, name=None):
    ev = None
    if cond is not None:
      ev = EvaluatorWrapper(query=cond)

    name = self.name_register.force_or_get(name, 'filter_cond')
    pushers = []
    mask = 1 * with_len + 2 * with_bitrate
    for mode in [0, 1]:
      if not (mask >> mode & 1): continue
      pushers.append(self.make_pusher(name, mode, desc='filter_cond >> {cond}'))

    def get_targets(m):
      if ev is None or ev.get_bool(m):
        return pushers
      return []

    fd = FilterDispatcher(target_syncs=get_targets)
    fd.pushers.update(pushers)
    return fd

  def create_mav_filter_pii(self, cond, name=None):
    ev = EvaluatorWrapper(query=cond)

    out2pusher = {}
    fd = FilterDispatcher(target_syncs=get_targets)
    name = self.name_register.force_or_get(name, 'filter_pii')

    def get_targets(m):
      res = ev.get_pii(m)
      for mode in [0, 1]:
        if not (res.B >> mode & 1): continue
        key = (res.A, mode)
        if key not in out2pusher:
          px = out2pusher[key] = self.make_pusher(
              f'{name}/{res.A}', mode, desc=f'filter_pii >> {res.A} {mode}'
          )
          fd.add_pusher(px)
        yield out2pusher[key]

    return fd


class CollectorParams(pydantic_settings.BaseSettings):
  cutoff_time_s: float


class MessageCollector(cmisc.PatchedModel):
  params: CollectorParams
  data: collections.deque = cmisc.pyd_f(collections.deque)
  filters: list[FilterDispatcher] = cmisc.pyd_f(list)
  tg: othreading.ThreadGroup
  pushers: rx_helpers.ObservableSet = cmisc.pyd_f(rx_helpers.ObservableSet)

  def get_resource_register(self) -> ResourceRegister:
    return ResourceRegister.FromItems(
        transformed_oset(
            self.pushers, to_input=lambda x: x.obj, to_output=SyncPusher.as_resource_item
        )
    )

  def __post_init__(self):
    self.pushers.ev.subscribe_safe(
        lambda action_and_pusher: self.tg.add(func=action_and_pusher[1].timer_func)
    )

  def normalize(self):
    curt = get_time()
    cutoff = curt - self.params.cutoff_time_s
    while len(self.data) and self.data[0].t < cutoff:
      self.data.popleft()

  def add_filter(self, filter: FilterDispatcher) -> FilterDispatcher:
    self.filters.append(filter)
    self.pushers.merge(filter.pushers)

    for x in self.data:
      filter(x)

    return filter

  def push(self, msg):
    self.normalize()
    self.data.append(msg)
    for f in self.filters:
      f(msg)


class StreamDatasetParams(cmisc.PatchedModel):
  update_freq_hz: float = 10
  tg: othreading.ThreadGroup = None
  pub: rx_helpers.WrapRX = None
  xy_mode: bool
  x_start: float = 0
  xmode_dt: float = 1


class StreamDataset(cmisc.PatchedModel):
  dds: DynamicDataset = cmisc.pyd_f(
      lambda: DynamicDataset(x=np.zeros(0), y=np.zeros(0), manual_notify_change=True)
  )
  params: StreamDatasetParams

  def __post_init__(self):

    if self.params.pub is None: return

    state = A(pending=[])
    l = threading.Lock()

    def update():
      if not state.pending: return
      with l:
        take, state.pending = state.pending, []

      self.push_data(take)

    def pushit(v):
      with l:
        state.pending.append(v)

    self.params.tg.add_timer(get_freq=lambda: self.params.update_freq_hz, action=update)
    self.params.pub.subscribe_safe(pushit)

  def push_data(self, dx):
    dx = np.array(dx)
    if self.params.xy_mode:
      self.dds.update_data(dx[:, 0], dx[:, 1])
    else:
      xlast = self.dds.xlast
      if xlast is None:
        xlast = self.params.x_start

      dt = self.params.xmode_dt
      x = np.arange(1, len(dx) + 1) * dt + xlast
      self.dds.update_data(x, dx)


class DDSPlotConnectorParams(cmisc.PatchedModel):
  ui_refresh_rate: float = 5
  sync_right_dist: float = None
  plot_req :  oplot_req.PlotRequest = cmisc.pyd_f(oplot_req.PlotRequest)
  tg: othreading.ThreadGroup
  dds: DynamicDataset


class DDSPlotConnector(cmisc.ContextModel):
  params: DDSPlotConnectorParams
  pe: oplot_req.PlotResult = None

  @property
  def dds(self) -> DynamicDataset:
    return self.params.dds

  @contextlib.contextmanager
  def _enter(self):
    assert self.dds.manual_notify_change
    self.pe = self.params.plot_req.plot(self.dds)

    def update():
      self.dds.notify_change()
      if self.params.sync_right_dist is not None and (xlast := self.dds.xlast) is not None:
        self.pe.ge.w.setXRange(xlast - self.params.sync_right_dist, xlast)

    self.params.tg.add_timer(get_freq=lambda: self.params.ui_refresh_rate, action=update)
    yield self


class ObsHelper(cmisc.PatchedModel):
  tg: othreading.ThreadGroup
  zmq_context: zmq.Context = None

  def pair2publisher(self, name: str, pair_obs: rx_helpers.WrapRX, rctx: ros_base.RospyContext):
    pf = rctx.create_publisher(name, ros_base.DataPoint, raw=True)
    pair_obs.map(lambda data: ros_base.DataPoint(stamp=float(data[0]), value=float(data[1]))
                ).subscribe_safe(pf)

  def syncpusher2publisher(self, sp: SyncPusher, rctx: ros_base.RospyContext):
    self.pair2publisher(sp.params.name, sp.pub, rctx)

  def syncpusher2sdds(self, sp: SyncPusher):
    dds = StreamDataset(
        params=StreamDatasetParams(
            pub=sp.pub, tg=self.tg, update_freq_hz=sp.params.rate_hz, xy_mode=True
        )
    )
    return dds

  def data_feeder(self, data, rate_hz: float, xy=True, loop=False):
    pub = rx_helpers.WrapRX.Subject()

    async def feeder(ev_stop):
      dt = 1 / rate_hz
      t = 0
      while True:
        for v in data:
          t += dt
          await asyncio.sleep(dt)
          pub.on_next((t, v) if xy else v)
        if not loop: break

    self.tg.add(func=feeder)
    return pub

  def rx2zmq_pub(self, pub: rx_helpers.WrapRX, pack, prefix=b'', socket=None, uri=None):
    if uri is not None:
      socket = self.zmq_context.socket(zmq.PUB)
      socket.bind(uri)

    def sendfunc(x):
      socket.send(prefix + pack(x))

    pub.subscribe_safe(sendfunc)

  def zmq_sub2rx(self, unpack, sz=None, prefix=b'', socket=None, uri=None, pub=None):
    if pub is None: pub = rx_helpers.WrapRX.Subject()
    if uri is not None:
      socket = self.zmq_context.socket(zmq.SUB)
      socket.connect(uri)

    socket.setsockopt(zmq.SUBSCRIBE, prefix)
    assert not prefix  # prefix nto working with gnuradio

    def loop(ev):
      while True:
        a = socket.recv()
        if a.startswith(prefix):
          if sz is not None:
            rem = a[len(prefix):]
          else:
            rem = a

          for x in unpack(rem):
            pub.on_next(x)
        else:
          assert 0

    self.tg.add(func=loop)
    return pub


def test_cel(ctx):

  a = chdrft_cel.chdrft_cel.CreateEvaluator('(int(obj.test1) % 5) == 3')
  print(a.EvaluateToBool('{"test1": 22}'))
  print(a.EvaluateToBool('{"test1": 123}'))
  print(a.EvaluateToBool('{"test1": "222"}'))
  b = chdrft_cel.chdrft_cel.CreateEvaluator('cel_server.base.PII{ a: 123, b:22 }')
  c = b.EvaluateToPII('{}')
  app.shell()

  pass


class GnuradioSISOBlock(cmisc.PatchedModel):
  iblock: object = None
  oblock: object = None

  @classmethod
  def SingleBlock(cls, blk):
    return GnuradioSISOBlock(iblock=blk, oblock=blk)

  @classmethod
  def NopBlock(cls):
    return cls.SingleBlock(None)


class RxGRMapParams(cmisc.PatchedModel):
  block: GnuradioSISOBlock
  obsh: ObsHelper
  input: rx_helpers.WrapRX


class RxGRMap(cmisc.ContextModel):
  params: RxGRMapParams

  output: rx_helpers.WrapRX = cmisc.pyd_f(rx_helpers.WrapRX.Subject)
  zmq_left: zmq.Socket = None
  zmq_right: zmq.Socket = None

  @contextlib.contextmanager
  def _enter(self):
    self.zmq_left = self.params.obsh.zmq_context.socket(zmq.PUB)
    self.zmq_right = self.params.obsh.zmq_context.socket(zmq.SUB)
    port_left = self.zmq_left.bind_to_random_port('tcp://*', min_port=11111, max_port=11112)
    port_right = self.zmq_right.bind_to_random_port('tcp://*', min_port=11112, max_port=11113)

    from gnuradio import zeromq as gr_zmq
    from gnuradio import gr

    self.params.obsh.rx2zmq_pub(
        pub=self.params.input,
        socket=self.zmq_left,
        pack=lambda x: struct.pack('<f', x[1]),
    )

    def unpackf(x):
      return np.frombuffer(x, dtype=np.float32, count=-1)

    self.params.obsh.zmq_sub2rx(
        socket=self.zmq_right,
        pub=self.output,
        unpack=unpackf,
        prefix=b'',
    )

    def runit(ev):
      gr_zmq_left = gr_zmq.sub_source(
          gr.sizeof_float, 1, f'tcp://localhost:{port_left}', 100, False, (-1), '', False
      )
      gr_zmq_right = gr_zmq.pub_sink(
          gr.sizeof_float, 1, f'tcp://localhost:{port_right}', 100, False, (-1), '', True, False
      )

      tb = gr.top_block()
      if self.params.block.iblock is None:
        tb.connect((gr_zmq_left, 0), (gr_zmq_right, 0))
      else:
        tb.connect((gr_zmq_left, 0), (self.params.block.iblock, 0))
        tb.connect((self.params.block.oblock, 0), (gr_zmq_right, 0))

      if gr.enable_realtime_scheduling() != gr.RT_OK:
        glog.warn("Error: failed to enable real-time scheduling.")
        # needs to be ran here, otherwise cleanup of block are fucking things up
      tb.start()
      tb.wait()

    self.params.obsh.tg.add(func=runit)

    yield self

class SignalGen(cmisc.PatchedModel):
  seed: int = 0

  @cmisc.cached_property
  def state(self):
    return np.random.RandomState(self.seed)

  def cos(self, n, freq, dt=1):
    return np.cos(2 * np.pi * np.arange(n) * dt * freq)
  def gauss(self, n, u=0, std=1):
    return np.cumsum(self.state.normal(size=10000, loc=u, scale=std))



def test_filter(ctx):
  val = SignalGen().gauss(10000)
  ds = Dataset(y=val)
  #res = ds.apply('LowPass(y)')
  #oservice.oplt.plot(res)
  r2 = ds.apply('ManualFilter(y, np.ones(75) /75, [1])')
  oplot_req.PlotRequest().plot(ds)
  oplot_req.PlotRequest().plot(r2)
  #%%
  app.shell()


def test_gr(ctx):
  from gnuradio import filter
  running_mean = filter.iir_filter_ffd(np.ones(5), [1], True)
  zmq_context = zmq.Context()

  tg = othreading.ThreadGroup()
  with tg.enter():
    obsh = ObsHelper(tg=tg, zmq_context=zmq_context)
    feed_rate = 100
    data = SignalGen().cos(100000, 3, dt=1/feed_rate)
    input = obsh.data_feeder(data, rate_hz=feed_rate, xy=True)
    params = RxGRMapParams(
        obsh=obsh, input=input, 
      block=GnuradioSISOBlock.SingleBlock(running_mean),
      #block=GnuradioSISOBlock.NopBlock(),
    )
    e = RxGRMap(params=params)

    sdds = StreamDataset(
        params=StreamDatasetParams(pub=e.output, tg=tg, update_freq_hz=30, xy_mode=False)
    )
    conn = DDSPlotConnector(
        params=DDSPlotConnectorParams(tg=tg, dds=sdds.dds, sync_right_dist=100, ui_refresh_rate=20)
    )
    with e, conn:
      app.shell()


def test_pw(ctx):

  tg = othreading.ThreadGroup()
  rctx = ros_base.RospyContext(subid='test_exports')
  with tg.enter(), rctx.enter():

    #%%
    #sdds = StreamDataset( params=StreamDatasetParams(xy_mode=False, xmode_dt=0.1),)

    obsh = ObsHelper(tg=tg)

    subject = rx_helpers.WrapRX.Subject()
    m = MessageCollector(tg=tg, params=CollectorParams(cutoff_time_s=10))
    m.pushers.ev.subscribe_safe(lambda e: obsh.syncpusher2publisher(e[1], rctx))

    pa = FilterParams(name='main')
    f0 = m.add_filter(pa.create_mav_filter_cond('(int(obj.seq) % 10) !=  0', with_len=1))
    f3 = m.add_filter(pa.create_mav_filter_cond('(int(obj.seq) % 100) ==  0', with_bitrate=1))
    #f3 = m.add_filter(create_mav_filter_cel('obj.mlen > 3', with_bitrate=1))

    subject.flat_map(mavlink_sniff()).map(MavlinkObsSchema.from_sniff).subscribe_safe(m.push)

    reg = m.get_resource_register()
    res = reg.search('obj.name.contains("/count")').single()
    sdds = obsh.syncpusher2sdds(res.obj)

    #%%
    #%%
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:%s" % 12322)
    #f0.pushers[0].pub.subscribe_safe(lambda v: socket.send(b'0' + struct.pack('<f', v[1])))
    #f1.pushers[0].pub.subscribe_safe(lambda v: socket.send(b'A' + struct.pack('<f', v)))
    #f1.pushers[1].pub.subscribe_safe(lambda v: socket.send(b'B' + struct.pack('<f', v)))
    #f2.pushers[0].pub.subscribe_safe(lambda v: socket.send(b'C' + struct.pack('<f', v[1])))

    g_sysc.speedup = 1

    async def feeder(ev_stop):
      gen = mavlink_utils.create_mav_heartbeat_gen()
      for i in range(1000):
        await asyncio.sleep(0.01)
        subject.on_next((False, next(gen)))

    tg.add(func=feeder)

    with DDSPlotConnector(
        params=DDSPlotConnectorParams(tg=tg, dds=sdds.dds, sync_right_dist=5)
    ) as connector:
      app.shell()


def test_stream(ctx):

  context = zmq.Context()
  socket = context.socket(zmq.PUB)
  socket.bind("tcp://*:%s" % 12322)
  rate = 1e2
  for i in range(int(rate) * 100):
    socket.send(b'F' + struct.pack('<f', np.cos(i / rate * 10)))
    #socket.send(b'I'+struct.pack('<I', i))
    time.sleep(1 / rate)


def test_stream_recv(ctx):

  # Socket to talk to server
  context = zmq.Context()

  rate = 1e2

  with othreading.ThreadGroup().enter() as tg:

    float_func = lambda x: struct.unpack('<f', x)[0]
    host = "tcp://localhost:12322"
    add_dynamic_plot(tg, res.w, zmq_sub2rx(host, b'C', float_func))
    add_dynamic_plot(tg, res.w, zmq_sub2rx(host, b'0', float_func))

    g_env.app.exec()


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
