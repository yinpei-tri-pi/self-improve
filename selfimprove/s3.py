"""Small S3 helpers: JSON objects and markers via boto3, folders via ``aws s3 sync``.

The profile comes from ``$SELFIMPROVE_AWS_PROFILE`` (the workstation: ``sagemaker``) or none (cluster
nodes: their instance role); the region from ``$SELFIMPROVE_AWS_REGION`` (the bucket's, us-west-2).
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse


def split(uri: str) -> tuple[str, str]:
    u = urlparse(uri)
    if u.scheme != "s3":
        raise ValueError(f"not an s3:// uri: {uri}")
    return u.netloc, u.path.lstrip("/")


def configure(profile: str | None, region: str | None) -> None:
    """Set the profile / region for this process (and the ``aws`` commands it starts)."""
    if profile is not None:
        os.environ["SELFIMPROVE_AWS_PROFILE"] = profile
    if region:
        os.environ["SELFIMPROVE_AWS_REGION"] = region
    client.cache_clear()


def _profile() -> str | None:
    return os.environ.get("SELFIMPROVE_AWS_PROFILE") or None


def _region() -> str:
    return os.environ.get("SELFIMPROVE_AWS_REGION") or "us-west-2"


@functools.lru_cache(maxsize=1)
def client():
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name=_profile(), region_name=_region())
    return session.client("s3", config=Config(retries={"max_attempts": 10, "mode": "adaptive"},
                                              max_pool_connections=64))


def exists(uri: str) -> bool:
    bucket, key = split(uri)
    try:
        client().head_object(Bucket=bucket, Key=key)
        return True
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def read_bytes(uri: str) -> bytes:
    bucket, key = split(uri)
    return client().get_object(Bucket=bucket, Key=key)["Body"].read()


def read_json(uri: str, default: Any = None) -> Any:
    try:
        return json.loads(read_bytes(uri))
    except Exception as exc:  # noqa: BLE001
        code = getattr(exc, "response", {}).get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return default
        raise


def write_bytes(uri: str, data: bytes, content_type: str = "application/octet-stream") -> None:
    bucket, key = split(uri)
    client().put_object(Bucket=bucket, Key=key, Body=data, ContentType=content_type)


def write_json(uri: str, obj: Any) -> None:
    write_bytes(uri, json.dumps(obj, indent=1, default=str).encode(), "application/json")


def iter_keys(prefix_uri: str) -> Iterator[tuple[str, int]]:
    """(uri, size) of every object under the prefix."""
    bucket, prefix = split(prefix_uri)
    for page in client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield f"s3://{bucket}/{obj['Key']}", int(obj["Size"])


def list_dirs(prefix_uri: str) -> list[str]:
    """The immediate 'sub-folders' of a prefix (names, without the trailing slash)."""
    bucket, prefix = split(prefix_uri.rstrip("/") + "/")
    out = []
    for page in client().get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for p in page.get("CommonPrefixes", []):
            out.append(p["Prefix"][len(prefix):].rstrip("/"))
    return out


def _aws(*args: str) -> list[str]:
    cmd = ["aws", *args, "--region", _region(), "--only-show-errors"]
    if _profile():
        cmd += ["--profile", _profile()]
    return cmd


def sync_up(local: str | os.PathLike, uri: str, *, delete: bool = False, excludes: tuple[str, ...] = ()) -> None:
    cmd = _aws("s3", "sync", str(local), uri.rstrip("/") + "/")
    if delete:
        cmd.append("--delete")
    for pattern in excludes:
        cmd += ["--exclude", pattern]
    subprocess.run(cmd, check=True)


def sync_down(uri: str, local: str | os.PathLike, *, includes: tuple[str, ...] = (),
              excludes: tuple[str, ...] = ()) -> None:
    Path(local).mkdir(parents=True, exist_ok=True)
    cmd = _aws("s3", "sync", uri.rstrip("/") + "/", str(local))
    for pattern in excludes:
        cmd += ["--exclude", pattern]
    for pattern in includes:
        cmd += ["--include", pattern]
    subprocess.run(cmd, check=True)
