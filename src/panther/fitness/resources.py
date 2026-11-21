"""The GPU memory and RAM that everything measured at once shares."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import torch

GIB = 1024**3


class Resources:
    """What is free when the run starts, shared out to the jobs running at once (llama.cpp processes, candidate files
    in RAM). A job waits until its share is free; a job larger than everything runs alone."""

    def __init__(self) -> None:
        self.gpu = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else float("inf")
        self.ram = self.free_ram() - 4 * GIB  # 4 GiB kept for this process
        self.used_gpu = self.used_ram = 0
        self.changed = threading.Condition()

    @staticmethod
    def free_ram() -> int:
        """Free RAM: the machine's, or, inside a container with a memory limit, what is left of that limit."""
        free = psutil.virtual_memory().available
        limit = Path("/sys/fs/cgroup/memory.max")
        if limit.exists() and limit.read_text().strip() != "max":
            used = int(Path("/sys/fs/cgroup/memory.current").read_text())
            free = min(free, int(limit.read_text()) - used)
        return free

    def reserve(self, ram: int) -> None:
        """Takes RAM out of what the jobs share, for something held the whole time (candidate files in RAM)."""
        self.ram -= ram

    @contextmanager
    def hold(self, gpu: int, ram: int) -> Iterator[None]:
        with self.changed:
            self.changed.wait_for(lambda: (self.used_gpu == self.used_ram == 0)
                                  or (self.used_gpu + gpu <= self.gpu and self.used_ram + ram <= self.ram))
            self.used_gpu += gpu
            self.used_ram += ram
        try:
            yield
        finally:
            with self.changed:
                self.used_gpu -= gpu
                self.used_ram -= ram
                self.changed.notify_all()
