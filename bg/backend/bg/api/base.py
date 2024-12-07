#!/usr/bin/env python

from chdrft.cmds import CmdsList
from chdrft.main import app
from chdrft.utils.cmdify import ActionHandler
import chdrft.utils.misc as cmisc
from chdrft.utils.misc import Attributize as A
import glog
import numpy as np
from pydantic import Field
from chdrft.utils.path import FileFormatHelper
import uuid
from sqlalchemy import exc
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from fastapi import FastAPI
import fastapi
import uvicorn
from bg.config import settings
import bg.model.base as model
from typing import Annotated

import datetime

from collections.abc import AsyncGenerator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pydantic
import uuid
import chdrft.display.control_center as cc
import shapely
import io

from fastapi import APIRouter, Depends, HTTPException, status

router = APIRouter(prefix="/api/v1", tags=["v1"])

engine = create_engine(settings.orm.url)

kParisLonLat = (2.333333, 48.866667)

def get_db_session() -> Session:
  with Session(engine, expire_on_commit=True) as session:
    try:
      yield session
      session.commit()
    except:
      session.rollback()
      raise
    finally:
      session.close()


async def get_asyc_db_session() -> AsyncGenerator[AsyncSession, None]:
  engine = create_async_engine(settings.orm.url)
  factory = async_sessionmaker(engine)
  async with factory() as session:
    try:
      yield session
      await session.commit()
    except exc.SQLAlchemyError as error:
      await session.rollback()
      raise


class UserContext(BaseModel):
  user_id: str
  active_game_id: str = None

class SessionData(BaseModel):
  ctx: UserContext
  user_id: str


class SessionHandler:

  def __init__(self):

    self.sessions = {}  # in memory backend -> single worker required!!

  def get(self, id: str) -> SessionData:
    return self.sessions.get(id)

  def create(self, db: Session) -> tuple[model.User, str]:
    user = model.User(id=str(uuid.uuid4()), name='test')
    db.add(user)
    sd = SessionData(user_id=user.id, ctx=UserContext(user_id=user.id))
    sid = str(uuid.uuid4())
    self.sessions[sid] = sd
    return user, sid


sh = SessionHandler()


class Config:
  arbitrary_types_allowed = True


dc = pydantic.dataclasses.dataclass(config=Config)


@dc
class Context:
  req: fastapi.Request
  res: fastapi.Response
  db: Annotated[Session, fastapi.Depends(get_db_session)]

  def __post_init__(self) -> None:
    glog.debug('Post iit cntext')


@dc
class BGSession:
  ctx: Annotated[Context, fastapi.Depends(Context)]
  cookie_id: Annotated[str | None, fastapi.Cookie()] = None

  def __post_init__(self) -> None:
    glog.debug('Post iit sessio')
    print('GEET BGSESSION ', self.cookie_id, sh.sessions.keys())
    session = sh.get(self.cookie_id)
    if session is None:
      print('CREATING USER - session')
      user, id = sh.create(self.ctx.db)
      self.ctx.res.set_cookie(key='cookie_id', value=id)
      self.cookie_id = id
    else:
      user = self.ctx.db.get(model.User, session.user_id)
    self.user: model.User = user



@dc
class PlayerState:
  pos: np.ndarray
  t: datetime.datetime

@dc
class PlayerPushPos:
  lonlatalt: list[float]
  def to_state(self) -> PlayerState:
    return PlayerState(pos=np.array(self.lonlatalt), t=datetime.datetime.now(datetime.UTC))


@dc
class Player:
  user_id: str
  id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
  states: list[PlayerState] = pydantic.Field(default_factory=list)
  def push_pos(self, pos: PlayerPushPos):
    self.states.append(pos.to_state())


  @property
  def output(self):
    return PlayerOutput(id=self.id)

@dc
class PlayerOutput:
  id: str


@dc
class Game:
  id: str = pydantic.Field(default_factory=lambda: str(uuid.uuid4()))
  players: list[Player] = pydantic.Field(default_factory=list)
  creation_pos: PlayerState = None

  def get_player(self, pid: str) -> Player:
    return cmisc.asq_query(self.players).single(lambda x: x.id == pid)
  def get_player_by_uid(self, uid: str) -> Player | None:
    return cmisc.asq_query(self.players).single_or_default(None, lambda x: x.user_id == uid)


  @property
  def output(self):
    return GameOutput(id=self.id, players=[x.output for x in self.players])

@dc
class GameOutput:
  id: str
  players: list[PlayerOutput]

fh_base = cc.FoliumHelper()


@dc
class GameManager:
  games: list[Game] = pydantic.Field(default_factory=list)

  def get_game(self, gid: str) -> Game | None:
    print(self.games)
    return cmisc.asq_query(self.games).single(lambda x: x.id == gid)

  def create_game_and_player(self, user: model.User):
    g = Game()
    self.games.append(g)
    g.players.append(Player(user_id=user.id))
    return g

  def get_game_map(self, game: Game, user: model.User) -> cc.FoliumHelper:
    from bg.utils.base import rh
    ep = rh.get_url(get_game_map_state, game_id=game.id)
    fh = cc.FoliumHelper(obs_endpoint=ep)
    start_pos = game.creation_pos.pos[:2] if game.creation_pos is not None else kParisLonLat
    fh.create_folium(start_pos, sat=False)
    fh.setup()
    return fh

  def get_player(self, pid: str) -> Player:
    return cmisc.asq_query(self.games).select_many(lambda x: x.players).single(lambda x: x.id == pid)

  def get_game_map_state(self, game: Game, user: model.User) -> list:
    p = game.get_player_by_uid(user.id)
    features = [shapely.Point(kParisLonLat)]
    if p.states:
      plast = p.states[-1]
      features = [shapely.Point(plast.pos)]
    return [fh_base.proc1(x, f'x_{i:03d}') for i, x in enumerate(features)]


mx = GameManager()

@router.get('/user/context')
def user_get_context(
    s: BGSession = fastapi.Depends(BGSession)
) -> fastapi.responses.JSONResponse:
  return s.ctx

@router.get('/user/setActiveGame/{game_id}')
def user_set_active_game(
    game_id: str, s: BGSession = fastapi.Depends(BGSession)
) -> fastapi.responses.JSONResponse:
  print('set actie game', game_id)
  return game_id


@router.post('/game/op/{game_id}/push_pos')
def get_game_map_state(
    game_id: str, pos: PlayerPushPos, s: BGSession = fastapi.Depends(BGSession)
) -> fastapi.responses.JSONResponse:
  mx.get_game(game_id).get_player(s.user.id).push_pos(pos)


@router.get('/game/op/{game_id}/map_state')
def get_game_map_state(
    game_id: str, s: BGSession = fastapi.Depends(BGSession)
) -> fastapi.responses.JSONResponse:
  g = mx.get_game(game_id)
  return mx.get_game_map_state(g, s.user)


@router.get('/game/create')
def game_create(pos: PlayerPushPos = None, s: BGSession = fastapi.Depends(BGSession),) -> GameOutput:
  g = mx.create_game_and_player(s.user)
  if pos is not None:
    g.creation_pos = PlayerPushPos.to_state()
  return g.output

@router.get('/game/op/{game_id}/map_display')
def get_game_map(game_id:str, s: BGSession = fastapi.Depends(BGSession),) -> fastapi.responses.HTMLResponse:
  g = mx.get_game(game_id)
  p = g.get_player_by_uid(s.user.id)
  assert p is not None

  fh =  mx.get_game_map(g, s.user)
  return fastapi.responses.HTMLResponse(content=fh.get_html())

