"""Process spawner for local and remote execution."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thinkagain.distributed.nodes import NodeConfig


class _ProcessHandle(ABC):
    @abstractmethod
    def poll(self) -> int | None: ...

    @abstractmethod
    def readline_stdout(self) -> str: ...

    @abstractmethod
    def read_stderr(self) -> str: ...

    @abstractmethod
    def terminate(self) -> None: ...

    @abstractmethod
    def kill(self) -> None: ...

    @abstractmethod
    def wait(self, timeout: float | None = None) -> None: ...


@dataclass
class SpawnedProcess:
    host: str
    port: int
    handle: _ProcessHandle

    def poll(self) -> int | None:
        return self.handle.poll()

    def readline_stdout(self) -> str:
        return self.handle.readline_stdout()

    def read_stderr(self) -> str:
        return self.handle.read_stderr()

    def terminate(self) -> None:
        self.handle.terminate()

    def kill(self) -> None:
        self.handle.kill()

    def wait(self, timeout: float | None = None) -> None:
        self.handle.wait(timeout=timeout)


class ProcessSpawner(ABC):
    @abstractmethod
    async def spawn(
        self,
        node: "NodeConfig",
        command: list[str],
        payload_path: str,
    ) -> SpawnedProcess: ...

    @abstractmethod
    async def cleanup(self) -> None: ...


class _LocalProcess(_ProcessHandle):
    def __init__(self, process: subprocess.Popen):
        self._process = process

    def poll(self) -> int | None:
        return self._process.poll()

    def readline_stdout(self) -> str:
        return self._process.stdout.readline()

    def read_stderr(self) -> str:
        return self._process.stderr.read()

    def terminate(self) -> None:
        self._process.terminate()

    def kill(self) -> None:
        self._process.kill()

    def wait(self, timeout: float | None = None) -> None:
        self._process.wait(timeout=timeout)


class LocalSpawner(ProcessSpawner):
    async def spawn(
        self,
        node: "NodeConfig",
        command: list[str],
        payload_path: str,
    ) -> SpawnedProcess:
        if not node.is_local:
            raise ValueError(f"LocalSpawner cannot spawn on remote host {node.host}")

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        port, _pid = await self._wait_for_ready(process)
        return SpawnedProcess(
            host=node.host,
            port=port,
            handle=_LocalProcess(process),
        )

    async def _wait_for_ready(self, process: subprocess.Popen) -> tuple[int, int | None]:
        import time

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line.startswith("READY:"):
                parts = line.strip().split(":")
                port = int(parts[1])
                pid = int(parts[2]) if len(parts) > 2 else None
                return port, pid
            if process.poll() is not None:
                stderr = process.stderr.read()
                raise RuntimeError(f"Server died during startup: {stderr}")
            await asyncio.sleep(0.05)

        process.terminate()
        raise RuntimeError("Server failed to start within timeout")

    async def cleanup(self) -> None:
        pass


class RemoteProcess(_ProcessHandle):
    def __init__(self, ssh_client, channel, host: str, pid: int | None = None):
        self._ssh_client = ssh_client
        self._channel = channel
        self._host = host
        self._pid = pid
        self._stdout_buffer: list[str] = []
        self._stderr_buffer: list[str] = []
        self._exit_status: int | None = None

        self._stdout_task = asyncio.create_task(
            self._buffer_stream(channel.makefile("r"), self._stdout_buffer)
        )
        self._stderr_task = asyncio.create_task(
            self._buffer_stream(channel.makefile_stderr("r"), self._stderr_buffer)
        )

    async def _buffer_stream(self, stream, buffer: list[str]) -> None:
        try:
            loop = asyncio.get_event_loop()
            while True:
                line = await loop.run_in_executor(None, stream.readline)
                if not line:
                    break
                buffer.append(line)
        except Exception:
            pass

    def poll(self) -> int | None:
        if self._exit_status is not None:
            return self._exit_status
        if self._channel.exit_status_ready():
            self._exit_status = self._channel.recv_exit_status()
        return self._exit_status

    def readline_stdout(self) -> str:
        return self._stdout_buffer.popleft() if self._stdout_buffer else ""

    def read_stderr(self) -> str:
        result = "".join(self._stderr_buffer)
        self._stderr_buffer.clear()
        return result

    def terminate(self) -> None:
        if self._pid:
            try:
                _, stdout, _ = self._ssh_client.exec_command(f"kill {self._pid}")
                stdout.channel.recv_exit_status()
            except Exception:
                pass
        self._channel.close()

    def kill(self) -> None:
        if self._pid:
            try:
                _, stdout, _ = self._ssh_client.exec_command(f"kill -9 {self._pid}")
                stdout.channel.recv_exit_status()
            except Exception:
                pass
        self._channel.close()

    def wait(self, timeout: float | None = None) -> None:
        if timeout:
            self._channel.settimeout(timeout)
        self._exit_status = self._channel.recv_exit_status()


class SSHSpawner(ProcessSpawner):
    def __init__(self):
        self._ssh_clients: dict[str, any] = {}
        self._sftp_clients: dict[str, any] = {}

    async def spawn(
        self,
        node: "NodeConfig",
        command: list[str],
        payload_path: str,
    ) -> SpawnedProcess:
        if node.is_local:
            raise ValueError("SSHSpawner should not be used for localhost")

        ssh_client = await self._get_ssh_client(node)
        remote_payload_path = await self._transfer_payload(node, payload_path)

        remote_command = [
            c if c != payload_path else remote_payload_path for c in command
        ]
        if remote_command[0] in (sys.executable, "python", "python3"):
            remote_command[0] = node.python_executable

        if node.env_setup:
            cmd_str = f"{node.env_setup} && {' '.join(remote_command)}"
        else:
            cmd_str = " ".join(remote_command)

        transport = ssh_client.get_transport()
        channel = transport.open_session()
        channel.exec_command(cmd_str)

        port, pid = await self._wait_for_ready(channel)
        remote_proc = RemoteProcess(
            ssh_client=ssh_client,
            channel=channel,
            host=node.host,
            pid=pid,
        )
        return SpawnedProcess(host=node.host, port=port, handle=remote_proc)

    async def _get_ssh_client(self, node: "NodeConfig"):
        import paramiko

        if node.host in self._ssh_clients:
            return self._ssh_clients[node.host]

        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: ssh_client.connect(
                hostname=node.host,
                port=node.ssh_port,
                username=node.ssh_user,
                key_filename=node.ssh_key_path,
                timeout=10.0,
            ),
        )

        self._ssh_clients[node.host] = ssh_client
        self._sftp_clients[node.host] = ssh_client.open_sftp()
        return ssh_client

    async def _transfer_payload(self, node: "NodeConfig", local_path: str) -> str:
        sftp = self._sftp_clients[node.host]
        remote_path = f"/tmp/{Path(local_path).name}"
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sftp.put(local_path, remote_path))
        return remote_path

    async def _wait_for_ready(self, channel) -> tuple[int, int | None]:
        import time

        deadline = time.monotonic() + 10.0
        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")
        loop = asyncio.get_event_loop()

        while time.monotonic() < deadline:
            line = await loop.run_in_executor(None, stdout.readline)
            if line.startswith("READY:"):
                parts = line.strip().split(":")
                port = int(parts[1])
                pid = int(parts[2]) if len(parts) > 2 else None
                return port, pid
            if channel.exit_status_ready():
                stderr_output = stderr.read()
                raise RuntimeError(
                    f"Remote server died during startup: {stderr_output}"
                )
            await asyncio.sleep(0.05)

        channel.close()
        raise RuntimeError("Remote server failed to start within timeout")

    async def cleanup(self) -> None:
        for sftp in self._sftp_clients.values():
            sftp.close()
        for ssh_client in self._ssh_clients.values():
            ssh_client.close()
        self._ssh_clients.clear()
        self._sftp_clients.clear()


def create_spawner(nodes: dict[str, "NodeConfig"]) -> ProcessSpawner:
    has_remote = any(not nc.is_local for nc in nodes.values())
    return SSHSpawner() if has_remote else LocalSpawner()
