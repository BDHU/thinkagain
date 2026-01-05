"""Process spawner for local and remote execution."""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from thinkagain.distributed.nodes import NodeConfig

logger = logging.getLogger(__name__)


@dataclass
class SpawnedProcess:
    """Represents a spawned server process with its connection details."""

    host: str
    port: int
    process: subprocess.Popen | None = None  # For local processes
    ssh_client: any = None  # For remote processes
    channel: any = None  # SSH channel for remote processes
    pid: int | None = None

    def poll(self) -> int | None:
        """Check if process has exited. Returns exit code or None if still running."""
        if self.process:
            return self.process.poll()
        if self.channel:
            if self.channel.exit_status_ready():
                return self.channel.recv_exit_status()
        return None

    def terminate(self) -> None:
        """Gracefully terminate the process."""
        if self.process:
            self.process.terminate()
        elif self.channel and self.ssh_client and self.pid:
            try:
                _, stdout, _ = self.ssh_client.exec_command(f"kill {self.pid}")
                stdout.channel.recv_exit_status()
            except Exception as exc:
                logger.warning(
                    "Failed to terminate remote process %s on %s",
                    self.pid,
                    self.host,
                    exc_info=exc,
                )
            self.channel.close()

    def kill(self) -> None:
        """Forcefully kill the process."""
        if self.process:
            self.process.kill()
        elif self.channel and self.ssh_client and self.pid:
            try:
                _, stdout, _ = self.ssh_client.exec_command(f"kill -9 {self.pid}")
                stdout.channel.recv_exit_status()
            except Exception as exc:
                logger.warning(
                    "Failed to kill remote process %s on %s",
                    self.pid,
                    self.host,
                    exc_info=exc,
                )
            self.channel.close()

    def wait(self, timeout: float | None = None) -> None:
        """Wait for process to complete."""
        if self.process:
            self.process.wait(timeout=timeout)
        elif self.channel:
            if timeout:
                self.channel.settimeout(timeout)
            self.channel.recv_exit_status()


def _parse_ready_line(line: str) -> tuple[int, int | None]:
    """Parse READY: message to extract port and pid.

    Expected format: READY:port[:pid]
    """
    parts = line.strip().split(":")
    port = int(parts[1])
    pid = int(parts[2]) if len(parts) > 2 else None
    return port, pid


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
        port, pid = await self._wait_for_ready(process)
        return SpawnedProcess(
            host=node.host,
            port=port,
            process=process,
            pid=pid,
        )

    async def _wait_for_ready(
        self, process: subprocess.Popen
    ) -> tuple[int, int | None]:
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            line = process.stdout.readline()
            if line.startswith("READY:"):
                return _parse_ready_line(line)
            if process.poll() is not None:
                stderr = process.stderr.read()
                raise RuntimeError(f"Server died during startup: {stderr}")
            await asyncio.sleep(0.05)

        process.terminate()
        raise RuntimeError("Server failed to start within timeout")

    async def cleanup(self) -> None:
        pass


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
        return SpawnedProcess(
            host=node.host,
            port=port,
            ssh_client=ssh_client,
            channel=channel,
            pid=pid,
        )

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
        deadline = time.monotonic() + 10.0
        stdout = channel.makefile("r")
        stderr = channel.makefile_stderr("r")
        loop = asyncio.get_event_loop()

        while time.monotonic() < deadline:
            line = await loop.run_in_executor(None, stdout.readline)
            if line.startswith("READY:"):
                return _parse_ready_line(line)
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
