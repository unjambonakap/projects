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

import fastapi

class RouteHelper(BaseModel):
  func2route: dict = Field(default_factory=dict)

  def setup(self, app: fastapi.FastAPI):
    for route in app.routes:
      ep = getattr(route, 'endpoint', None)
      if ep is None: continue
      assert ep not in self.func2route
      self.func2route[ep] = route


  def get_url(self, func, **kwargs) -> str:
    route =  self.func2route[func]
    return route.url_path_for(route.name, **kwargs)

rh = RouteHelper()
