from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CallRequest(_message.Message):
    __slots__ = ("replica_name", "method", "args", "kwargs")
    REPLICA_NAME_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    replica_name: str
    method: str
    args: bytes
    kwargs: bytes
    def __init__(self, replica_name: _Optional[str] = ..., method: _Optional[str] = ..., args: _Optional[bytes] = ..., kwargs: _Optional[bytes] = ...) -> None: ...

class CallResponse(_message.Message):
    __slots__ = ("result", "error")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    result: bytes
    error: str
    def __init__(self, result: _Optional[bytes] = ..., error: _Optional[str] = ...) -> None: ...

class DeployRequest(_message.Message):
    __slots__ = ("replica_name", "instances", "cpus", "gpus", "args", "kwargs")
    REPLICA_NAME_FIELD_NUMBER: _ClassVar[int]
    INSTANCES_FIELD_NUMBER: _ClassVar[int]
    CPUS_FIELD_NUMBER: _ClassVar[int]
    GPUS_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    KWARGS_FIELD_NUMBER: _ClassVar[int]
    replica_name: str
    instances: int
    cpus: int
    gpus: int
    args: bytes
    kwargs: bytes
    def __init__(self, replica_name: _Optional[str] = ..., instances: _Optional[int] = ..., cpus: _Optional[int] = ..., gpus: _Optional[int] = ..., args: _Optional[bytes] = ..., kwargs: _Optional[bytes] = ...) -> None: ...

class DeployResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...

class ShutdownRequest(_message.Message):
    __slots__ = ("replica_name",)
    REPLICA_NAME_FIELD_NUMBER: _ClassVar[int]
    replica_name: str
    def __init__(self, replica_name: _Optional[str] = ...) -> None: ...

class ShutdownResponse(_message.Message):
    __slots__ = ("success", "error")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    success: bool
    error: str
    def __init__(self, success: bool = ..., error: _Optional[str] = ...) -> None: ...
