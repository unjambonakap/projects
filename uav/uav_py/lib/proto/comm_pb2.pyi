from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SiyiJoyMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    YAW: _ClassVar[SiyiJoyMode]
    PITCH: _ClassVar[SiyiJoyMode]
YAW: SiyiJoyMode
PITCH: SiyiJoyMode

class SiyiJoyData(_message.Message):
    __slots__ = ("mode", "ctrl", "zoom", "photo_count", "record_count", "center")
    MODE_FIELD_NUMBER: _ClassVar[int]
    CTRL_FIELD_NUMBER: _ClassVar[int]
    ZOOM_FIELD_NUMBER: _ClassVar[int]
    PHOTO_COUNT_FIELD_NUMBER: _ClassVar[int]
    RECORD_COUNT_FIELD_NUMBER: _ClassVar[int]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    mode: SiyiJoyMode
    ctrl: float
    zoom: float
    photo_count: int
    record_count: int
    center: int
    def __init__(self, mode: _Optional[_Union[SiyiJoyMode, str]] = ..., ctrl: _Optional[float] = ..., zoom: _Optional[float] = ..., photo_count: _Optional[int] = ..., record_count: _Optional[int] = ..., center: _Optional[int] = ...) -> None: ...

class SiyiRawReq(_message.Message):
    __slots__ = ("msg_id", "desc", "res")
    MSG_ID_FIELD_NUMBER: _ClassVar[int]
    DESC_FIELD_NUMBER: _ClassVar[int]
    RES_FIELD_NUMBER: _ClassVar[int]
    msg_id: int
    desc: str
    res: bytes
    def __init__(self, msg_id: _Optional[int] = ..., desc: _Optional[str] = ..., res: _Optional[bytes] = ...) -> None: ...
