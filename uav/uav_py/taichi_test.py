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
from pydantic import Field
import pandas as pd
import polars as pl
import inspect

global flags, cache
flags = None
cache = None

import taichi as ti


ti.init(arch=ti.cpu)
import ast
import textwrap

#%%

class AstModifier(cmisc.PatchedModel):
  src: str

  @classmethod
  def Make(cls, obj):
    print(obj, obj.__dict__)
    src = textwrap.dedent(inspect.getsource(obj))
    return cls(src=src)

  def get_str(self, astobj: ast.Node) -> str:
    return ast.get_source_segment(self.src, astobj)

  def process(self):

    class TsfX(ast.NodeTransformer):

      def visit_Call(this, obj):
        desc = self.get_str(obj.func)
        if desc.startswith('np.') or desc.startswith('ti.'):
          print('al')

        return obj

    tsf = TsfX()
    print('gogo ', self.src)
    t = ast.parse(self.src)
    print(t)
    nt = tsf.visit(t)
    print(ast.dump(nt, indent=2))
    return ast.unparse(nt)

def abc():

  def f(x, y):
    return x + y

  b = 33
  u = lambda x, y: a.f(x, y, b)
  print(type(u))
  print(u.__closure__[0])
  print(u.__code__)
  a = AstModifier.Make(u).process()
  print(a)

abc()

#%%


def proctest(f):
  a = ti.field(dtype=ti.f32, shape=())
  b = ti.field(dtype=ti.f32, shape=())
  c = ti.field(dtype=ti.f32, shape=())


  def comp():
    c[None] = ti.func(f)(a[None], b[None])

  u = ti.kernel(comp)

  def call(x, y):
    a[None] = x
    b[None] = y
    u()
    return c[None]

  return A(u=u, call=call)


def lx(x,y): return x+y


ti.init(debug=True, print_ir=True, offline_cache=False)
a = proctest(lx)
a.call(10, 20)
m.add_kernel(a.u)

#%%

def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
