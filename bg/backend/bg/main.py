#!/usr/bin/env python

#from __future__ import annotations
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
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI
import fastapi
import uvicorn
from bg.config import settings
import bg.model.base as model
from typing import Annotated


from fastapi import Depends

from collections.abc import AsyncGenerator

from sqlalchemy import create_engine 
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pydantic
import uuid
import bg.api.base as api
from bg.utils.base import rh







#def get_session( response: fastapi.Response, db: Depends(get_db_session), cookie_id: str | None = fastapi.Cookie(None),) -> SessionData:
#  return session_cookie


class SPAStaticFiles(StaticFiles):
  async def get_response(self, path: str, scope):
    response = await super().get_response(path, scope)
    #if response.status_code == 404:
    #    response = await super().get_response('.', scope)
    return response

fapp = FastAPI()

origins = [
    "http://localhost",
    "https://localhost",
  "http://localhost:5173",
    "http://localhost:8080",
]

from fastapi.middleware.cors import CORSMiddleware
fapp.add_middleware(
    CORSMiddleware,
  allow_origin_regex="http://localhost:*",
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


fapp.mount('/ts', SPAStaticFiles(directory=cmisc.path_here('../../ts_app/dist'), html=True), name='TS app')
fapp.mount('/assets', SPAStaticFiles(directory=cmisc.path_here('../../ts_app/dist/assets'), html=True), name='TS app')
fapp.mount('/html_tests', StaticFiles(directory=cmisc.path_here('../../html_tests'), html=True), name='HTML test app')
fapp.include_router(api.router)


@fapp.get("/")
def root(s: api.BGSession = Depends(api.BGSession)):
    return fastapi.responses.RedirectResponse('/ts')

@fapp.get("/test")
def test(s: api.BGSession = Depends(api.BGSession), c: api.Context =  Depends(api.Context)):
  return 'hello'


rh.setup(fapp)

global flags, cache
flags = None
cache = None


def args(parser):
  clist = CmdsList()
  ActionHandler.Prepare(parser, clist.lst, global_action=1)

def run(ctx):
  uvicorn.run(fapp, host="0.0.0.0", port=8000)

def test(ctx):
  pass


def main():
  ctx = Attributize()
  ActionHandler.Run(ctx)


app()
