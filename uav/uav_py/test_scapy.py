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
import time
import chdrft.utils.Z as Z
import scapy.all as scapy


global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--target', type=str)
  parser.add_argument('--iface', type=str)
  ActionHandler.Prepare(parser, clist.lst, global_action=1)



def check_caps(ctx):
  import subprocess as sp
  print(sp.check_output(f'getpcaps {os.getpid()}', shell=True))

@cmisc.logged_failsafe
def disp_cb(packet: scapy.Packet):
  if not scapy.UDP in packet: return None
  summary =packet[scapy.IP].summary()
  plen = len(packet)
  plen_ether = len(packet[scapy.Ether].payload)
  plen_ip = len(packet[scapy.IP].payload)
  plen_udp = len(packet[scapy.UDP].payload)
  return f'{plen=} {plen_ether=} {plen_ip=} {plen_udp=} {summary}'

def test_rpi(ctx):
  print('start snif')
  print(scapy.sniff(iface=[ctx.iface], filter=f'udp && host {ctx.target}', prn=disp_cb))

def test(ctx):
  #%%
  print('start snif')
  print(scapy.sniff(iface=['wlp1s0'], filter='udp && host 192.168.193.223', prn=disp_cb))
  #%%
  pass


def main():
  ctx = A()
  ActionHandler.Run(ctx)


app()
