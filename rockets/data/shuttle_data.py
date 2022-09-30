#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
import glog
import chdrft.utils.Z as Z
import numpy as np

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

def get_data():
  data = '''1 6 12 18 24 30 36 42 48 54 60 66 72 78 84 9- 96 102 108 112 112 114 116 118 120 122 124
1108355 1051639 9812759-09343 840987 781382 726201 673831 624044 575407 526297 476384 424727 370760 314736 258294 200243 143571 89025 71584 54717 39054 24366 13310 5975 2383 2246'''
  data = Z.pd.read_csv(Z.io.StringIO(data), index_col=0, header=None).T

  tl  = s_time.split()
  wl  = s_weight_lb.split()
  for t, w in zip(tl, wl)

def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()

