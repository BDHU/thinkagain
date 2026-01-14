from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ExecuteRequest(_message.Message):
    __slots__ = ("args",)
    ARGS_FIELD_NUMBER: _ClassVar[int]
    args: bytes
    def __init__(self, args: _Optional[bytes] = ...) -> None: ...

class ExecuteResponse(_message.Message):
    __slots__ = ("result", "error")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    result: bytes
    error: str
    def __init__(
        self, result: _Optional[bytes] = ..., error: _Optional[str] = ...
    ) -> None: ...
