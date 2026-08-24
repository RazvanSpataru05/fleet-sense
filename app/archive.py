"""The S3 recording archive.

Recordings live in a bucket instead of being pushed through the browser on every analysis.
Measured on the deployed stack, a 119 MB recording costs ~5.2 s to upload from a laptop
and ~3.9 s to analyse; read from S3 in the same region the transfer is roughly a second
and happens inside AWS, so it does not depend on whatever network the user is sitting on.

Browser upload is still fully supported -- this is an additional source, not a replacement.

Degrades cleanly: with S3_BUCKET unset (local development, or a deployment without an
archive) is_enabled() returns False and the UI simply does not offer the picker.
"""
import os
from pathlib import Path

# Same allowlist the upload path enforces. Applied here too because a key arrives from the
# client and IAM only constrains *which bucket* can be read, not what is asked for inside it.
ALLOWED_SUFFIXES = {".csv", ".txt"}


class ArchiveError(RuntimeError):
    """Something went wrong talking to S3. The message is written to be safe to show a
    user -- no bucket internals, no boto stack detail."""


def bucket_name():
    return os.environ.get("S3_BUCKET", "").strip()


def is_enabled():
    return bool(bucket_name())


def _client():
    # Imported lazily: boto3 is only needed when an archive is configured, and a missing
    # dependency should disable a feature rather than stop the whole app from starting.
    try:
        import boto3
    except ImportError as exc:
        raise ArchiveError("The recording archive requires boto3, which is not installed.") from exc
    # No credentials passed deliberately. On Fargate boto3 reads the task role from the
    # container credentials endpoint; anywhere that needs explicit keys is misconfigured.
    return boto3.client("s3")


def list_recordings():
    """Every usable recording in the bucket, newest first."""
    if not is_enabled():
        return []
    try:
        items = []
        # Paginated rather than a single list_objects_v2: that call caps at 1000 keys and
        # silently truncates, which would quietly hide recordings as the archive grows.
        for page in _client().get_paginator("list_objects_v2").paginate(Bucket=bucket_name()):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or Path(key).suffix.lower() not in ALLOWED_SUFFIXES:
                    continue
                items.append({"key": key,
                              "name": key.rsplit("/", 1)[-1],
                              "size_bytes": obj["Size"],
                              "last_modified": obj["LastModified"].isoformat()})
    except ArchiveError:
        raise
    except Exception as exc:
        raise ArchiveError(f"Could not list the recording archive: {_reason(exc)}") from exc
    items.sort(key=lambda i: i["last_modified"], reverse=True)
    return items


def download_to(key, dest_path):
    """Stream one object to a local path.

    Streamed to disk rather than read into memory: a recording is ~119 MB, and with two
    gunicorn workers a pair of concurrent requests would otherwise be ~240 MB of RSS on a
    task sized for far less.
    """
    if not is_enabled():
        raise ArchiveError("No recording archive is configured.")
    if Path(key).suffix.lower() not in ALLOWED_SUFFIXES:
        raise ArchiveError(f"{key!r} is not a .csv or .txt recording.")
    try:
        with open(dest_path, "wb") as fh:
            _client().download_fileobj(bucket_name(), key, fh)
    except ArchiveError:
        raise
    except Exception as exc:
        raise ArchiveError(f"Could not read {key!r} from the archive: {_reason(exc)}") from exc


def _reason(exc):
    """Boto exceptions stringify into a wall of detail that is no use to a plant manager
    and leaks account internals into a browser response. Translate the codes worth acting
    on and fall back to the class name."""
    response = getattr(exc, "response", None)
    code = (response or {}).get("Error", {}).get("Code") if isinstance(response, dict) else None
    return {"NoSuchKey": "the recording no longer exists",
            "404": "the recording no longer exists",
            "AccessDenied": "access denied -- check the task role's S3 permissions",
            "403": "access denied -- check the task role's S3 permissions",
            "NoSuchBucket": "the bucket does not exist",
            }.get(code, code or exc.__class__.__name__)
