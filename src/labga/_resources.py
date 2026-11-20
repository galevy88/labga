"""Process-aware CPU budget detection, including common Linux cgroup layouts."""

import math
import os
from pathlib import Path


def _quota(directory: Path) -> int | None:
    try:
        if (directory / "cpu.max").is_file():
            quota_text, period_text = (directory / "cpu.max").read_text().split()
        else:
            quota_text = (directory / "cpu.cfs_quota_us").read_text()
            period_text = (directory / "cpu.cfs_period_us").read_text()
        quota, period = int(quota_text), int(period_text)
        if quota > 0 and period > 0:
            return max(1, math.ceil(quota / period))
    except (OSError, ValueError):
        pass
    return None


def _cgroup_limit(root: Path = Path("/sys/fs/cgroup")) -> int | None:
    directories = {root, root / "cpu", root / "cpu,cpuacct", root / "cpuacct,cpu"}
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            _, controllers, path = line.split(":", 2)
            if controllers and "cpu" not in controllers.split(","):
                continue
            base = root / controllers if controllers else root
            relative = Path(path.lstrip("/"))
            if ".." in relative.parts:
                continue
            directories.update(base / part for part in (relative, *relative.parents))
    except (OSError, ValueError):
        pass
    quotas = [value for directory in directories if (value := _quota(directory)) is not None]
    return min(quotas, default=None)


def available_cpus() -> tuple[int, dict[str, int | None]]:
    """Return the effective budget and the individual observations behind it."""
    process_count = getattr(os, "process_cpu_count", None)
    process = process_count() if process_count else None
    affinity_fn = getattr(os, "sched_getaffinity", None)
    try:
        affinity = len(affinity_fn(0)) if affinity_fn else None
    except OSError:
        affinity = None
    system = os.cpu_count()
    quota = _cgroup_limit()
    available = process or affinity or system or 1
    if affinity is not None:
        available = min(available, affinity)
    if quota is not None:
        available = min(available, quota)
    return max(1, available), {
        "process": process,
        "affinity": affinity,
        "system": system,
        "quota": quota,
    }
