#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
from chdrft.utils.misc import Attributize
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
import pandas as pd
import polars as pl
import re
import os.path
import os
import sys
import time
import chdrft.utils.Z as Z
import httpx


global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  parser.add_argument('--target', default='http://localhost:8080')
  ActionHandler.Prepare(parser, clist.lst, global_action=1)


def test(ctx):
  with httpx.Client(base_url=ctx.target, cookies=httpx.Cookies()) as client:
    print(client._cookies)
    print(client.get('/'))
    print(client._cookies)
    print(client.get('/test').json())
    print(client._cookies)
    print(client.get('/test').json())
    print(client._cookies)
    print(client.get(f'/api/v1/user/111/setActiveGame').json())
    g =client.get('/v1/game/create').json()
    client.post(f'/v1/game/op/{g["id"]}/push_pos', json=dict(lonlatalt=[1,2,3])).json()
    map_state =client.get(f'/v1/game/op/{g["id"]}/map_state').json()
    print(map_state)
    map =client.get(f'/v1/game/op/{g["id"]}/map_display')
    print(map.content)



def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
