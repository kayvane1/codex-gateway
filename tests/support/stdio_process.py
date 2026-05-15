from __future__ import annotations


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[bytes] = []

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    async def drain(self) -> None:
        return None


class FakeStream:
    def __init__(self, lines: list[bytes]) -> None:
        self.lines = list(lines)

    async def readline(self) -> bytes:
        if self.lines:
            return self.lines.pop(0)
        return b""


class FakeProcess:
    def __init__(
        self,
        *,
        stdin: FakeStdin | None = None,
        stdout: FakeStream | None = None,
        stderr: FakeStream | None = None,
        returncode: int | None = None,
    ) -> None:
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode
