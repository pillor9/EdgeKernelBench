"""Bounded read-only evidence after SSH transport failure; never rerun work."""
from __future__ import annotations

import json
import shlex
import subprocess
import uuid

from ..operation_control import OperationCancelled, run_cancellable

# New independent connection: a stale multiplex master must not be the observer.
DIAGNOSTIC_SSH_OPTS = (
    '-o', 'BatchMode=yes', '-o', 'ConnectionAttempts=1',
    '-o', 'ConnectTimeout=5', '-o', 'ServerAliveInterval=5',
    '-o', 'ServerAliveCountMax=1', '-o', 'ControlMaster=no',
    '-o', 'ControlPath=none',
)
REMOTE_SNAPSHOT = r'''
import json,pathlib,shutil,time
out={'device_epoch_s':time.time()}
for name,path in [('boot_id','/proc/sys/kernel/random/boot_id'),('uptime','/proc/uptime'),('meminfo','/proc/meminfo')]:
 try:
  value=pathlib.Path(path).read_text().strip()
  if name=='meminfo':value={line.split(':')[0]:line.split(':')[1].strip() for line in value.splitlines() if line.startswith(('MemTotal:','MemAvailable:','SwapTotal:','SwapFree:'))}
  out[name]=value
 except OSError as e:out[name]={'error':type(e).__name__}
try:
 d=shutil.disk_usage('/');out['disk']={'total_bytes':d.total,'free_bytes':d.free}
except OSError as e:out['disk']={'error':type(e).__name__}
out['network']={}
for name in ('eth0','eth1'):
 p=pathlib.Path('/sys/class/net')/name
 if not p.exists():continue
 out['network'][name]={}
 for field in ('operstate','carrier','carrier_changes'):
  try:out['network'][name][field]=(p/field).read_text().strip()
  except OSError as e:out['network'][name][field]={'error':type(e).__name__}
print(json.dumps(out))
'''


def boot_id_from_text(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    # Accept exactly one UUID line: banners/noisy startup scripts aren't identity.
    value = text.strip()
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError):
        return None
    return parsed if value.lower() == parsed else None


def collect_transport_evidence(alias: str, *, before_boot_id: str | None, control=None) -> dict:
    evidence = {
        'schema_version': 'leka.transport_failure_evidence.v1',
        'before_boot_id': before_boot_id,
        'scope': 'one fresh read-only SSH observation after failure; not failure-time root cause',
        'gpu_workloads': 0, 'worker_retries': 0,
    }
    argv = ('ssh', *DIAGNOSTIC_SSH_OPTS, alias,
            'python3 -c ' + shlex.quote(REMOTE_SNAPSHOT))
    try:
        proc = run_cancellable(argv, control=control, capture_output=True, text=True, timeout=12)
    except OperationCancelled:
        evidence.update(status='observation_skipped_operation_cancelled', root_cause='unknown')
        return evidence
    except (OSError, subprocess.TimeoutExpired) as error:
        evidence.update(status='observation_unavailable', error_type=type(error).__name__, root_cause='unknown')
        return evidence
    evidence['probe_returncode'] = proc.returncode
    if proc.returncode != 0:
        # Keep raw stderr out: callers retain the original transport diagnostic.
        evidence.update(status='observation_unavailable', root_cause='unknown')
        return evidence
    try:
        if not isinstance(proc.stdout, str) or len(proc.stdout) > 32768:
            raise ValueError('invalid snapshot size')
        snapshot = json.loads(proc.stdout)
        if not isinstance(snapshot, dict):
            raise ValueError('snapshot must be an object')
        after = boot_id_from_text(snapshot.get('boot_id'))
        if after is None:
            raise ValueError('missing device identity')
    except (ValueError, TypeError):
        evidence.update(status='invalid_observation', root_cause='unknown')
        return evidence
    evidence['snapshot'] = snapshot
    if before_boot_id is None:
        evidence['status'] = 'reachable_after_failure_prior_boot_unknown'
    elif before_boot_id != after:
        evidence['status'] = 'device_reboot_observed'
    else:
        evidence['status'] = 'reachable_after_failure_same_boot'
    evidence['root_cause'] = 'unknown'
    return evidence
