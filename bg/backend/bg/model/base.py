#!/usr/bin/env python

from __future__ import annotations
from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
from chdrft.utils.path import FileFormatHelper

global flags, cache
flags = None
cache = None

import uuid

from sqlalchemy import Column, ForeignKey, Table, orm
from sqlalchemy.dialects.sqlite import CHAR

from sqlalchemy.ext.declarative import declared_attr


class Base(object):

  @declared_attr
  def __tablename__(cls):
    return cls.__name__.lower()


from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base(cls=Base)


class BaseUUID(object):

  id: orm.Mapped[str] = orm.mapped_column(
      CHAR(32),
      primary_key=True,
      default=lambda: str(uuid.uuid4()),
  )


class User(BaseUUID, Base):
  __tablename__ = 'user'

  name: orm.Mapped[str]
