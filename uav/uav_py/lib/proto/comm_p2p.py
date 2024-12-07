# This is an automatically generated file, please do not change
# gen by protobuf_to_pydantic[v0.3.0.2](https://github.com/so1n/protobuf_to_pydantic)
# Protobuf Version: 5.28.2
# Pydantic Version: 2.9.2
from enum import IntEnum

from google.protobuf.message import Message  # type: ignore
from pydantic import BaseModel, Field


class SiyiJoyMode(IntEnum):
    YAW = 0
    PITCH = 1


class SiyiJoyData(BaseModel):
    mode: SiyiJoyMode = Field(default=0)
    ctrl: float = Field(default=0.0)
    zoom: float = Field(default=0.0)
    photo_count: int = Field(default=0)
    record_count: int = Field(default=0)
    center: int = Field(default=0)


class SiyiRawReq(BaseModel):
    msg_id: int = Field(default=0)
    desc: str = Field(default="")
    res: bytes = Field(default=b"")
