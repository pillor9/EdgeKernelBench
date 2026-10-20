"""Read-only PMU attachment to the owned Vivo RPC process generation.

The listener forks sessions from a background thread. /proc/PID/task/PID/
children is therefore insufficient; use a bounded PID/PPID census and include
existing descendants explicitly. Default simpleperf inheritance covers only
children created after attachment. The remaining attach race is reported,
never promoted into exact candidate or kernel attribution.
"""

from __future__ import annotations

from pathlib import Path
import re
import shlex

from ...backends import BACKENDS


def read_rpc_owner(serial: str) -> dict | None:
    cfg = BACKENDS["vivo_cpu"]
    path = Path(cfg["rpc_state_dir"]) / "rpc_owner.tsv"
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 4096:
            return None
        fields = path.read_text().strip().split("\t")
    except (OSError, UnicodeError):
        return None
    if len(fields) != 6:
        return None
    device, port, binary, pid, ticks, boot = fields
    expected_binary = str(cfg["device_lib_dir"]).rstrip("/") + "/tvm_rpc"
    if (device != serial or port != str(cfg["rpc_port"]) or binary != expected_binary
            or not pid.isdecimal() or int(pid) <= 0
            or not ticks.isdecimal() or int(ticks) <= 0
            or re.fullmatch(r"[A-Za-z0-9-]+", boot) is None):
        return None
    return {"pid": pid, "start_ticks": ticks, "boot_id": boot,
            "binary": binary, "port": int(port)}


def runtime_snapshot_command(owner: dict | None) -> str:
    """One bounded read-only command: owned-root tasks, then global sysfs.

    A failed process proof does not suppress unrelated device-global readings.
    No process-name search, descendant discovery, or affinity mutation occurs.
    """
    prefix = "printf 'LEKA_SNAPSHOT_UNAVAILABLE owner_missing\\n'\n"
    if owner is not None:
        expected = (f"{owner['binary']} server --host=0.0.0.0 --port={owner['port']} "
                    f"--port-end={owner['port'] + 1}")
        values = {"root": owner["pid"], "root_ticks": owner["start_ticks"],
                  "boot": owner["boot_id"], "expected": expected}
        prefix = "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items()) + r'''
snapshot_ticks() {
    IFS= read -r record < "/proc/$1/stat" || return 1
    rest=${record##*) }; set -- $rest
    [ "$#" -ge 20 ] || return 1
    shift 19; printf '%s\n' "$1"
}
snapshot_root_matches() {
    IFS= read -r actual_boot < /proc/sys/kernel/random/boot_id || return 1
    [ "$actual_boot" = "$boot" ] || return 1
    [ "$(snapshot_ticks "$root")" = "$root_ticks" ] || return 1
    command=$(tr '\000' ' ' < "/proc/$root/cmdline") || return 1
    [ "${command% }" = "$expected" ]
}
snapshot_owned_tasks() {
    snapshot_root_matches || return 91
    printf 'LEKA_SNAPSHOT_BEGIN %s %s %s\n' "$root" "$root_ticks" "$boot"
    task_count=0
    for status_file in /proc/$root/task/[0-9]*/status; do
        [ -r "$status_file" ] || return 92
        task_count=$((task_count + 1)); [ "$task_count" -le 256 ] || return 92
        tid=${status_file%/status}; tid=${tid##*/}
        ticks=$(snapshot_ticks "$root/task/$tid") || return 93
        allowed=$(awk '/^Cpus_allowed_list:/{print $2}' "$status_file") || return 93
        [ -n "$allowed" ] || return 93
        [ "$(snapshot_ticks "$root/task/$tid")" = "$ticks" ] || return 93
        printf 'LEKA_SNAPSHOT_TASK %s %s %s %s\n' "$root" "$tid" "$ticks" "$allowed"
    done
    [ "$task_count" -gt 0 ] || return 92
    snapshot_root_matches || return 91
    printf 'LEKA_SNAPSHOT_END %s %s %s %s\n' "$root" "$root_ticks" "$boot" "$task_count"
}
snapshot_owned_tasks || printf 'LEKA_SNAPSHOT_UNAVAILABLE generation_or_task_proof_failed\n'
'''
    return prefix + r'''
for cpu in 0 1 2 3 4 5 6 7; do
    path=/sys/devices/system/cpu/cpu${cpu}/cpufreq/scaling_cur_freq
    if [ -r "$path" ] && IFS= read -r value < "$path"; then
        printf 'cpu%s_freq_khz=%s\n' "$cpu" "$value"
    fi
done
for zone in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    base=/sys/class/thermal/thermal_zone${zone}
    if [ -r "$base/type" ] && [ -r "$base/temp" ] &&
        IFS= read -r kind < "$base/type" && IFS= read -r value < "$base/temp"; then
        printf 'thermal%s=%s:%s\n' "$zone" "$kind" "$value"
    fi
done
'''


def parse_cpu_list(value: str) -> set[int] | None:
    if not isinstance(value, str) or not value or len(value) > 1024:
        return None
    result = set()
    for item in value.split(","):
        ends = item.split("-")
        if len(ends) not in (1, 2) or any(not p.isascii() or not p.isdecimal() for p in ends):
            return None
        first, last = int(ends[0]), int(ends[-1])
        if not 0 <= first <= last < 1024:
            return None
        cpus = set(range(first, last + 1))
        if cpus & result:
            return None
        result.update(cpus)
    return result


def parse_runtime_snapshot_proof(report: str, owner: dict | None) -> dict | None:
    """Reject every process row if either root or any task proof is incomplete."""
    if owner is None or len(report.encode("utf-8")) > 256 * 1024:
        return None
    rows = [line.split() for line in report.splitlines() if line.startswith("LEKA_SNAPSHOT_")]
    expected = [owner["pid"], owner["start_ticks"], owner["boot_id"]]
    if (not 3 <= len(rows) <= 258 or rows[0] != ["LEKA_SNAPSHOT_BEGIN", *expected]
            or rows[-1][:4] != ["LEKA_SNAPSHOT_END", *expected] or len(rows[-1]) != 5
            or rows[-1][4] != str(len(rows) - 2)):
        return None
    affinities, generations = {}, {}
    for row in rows[1:-1]:
        if (len(row) != 5 or row[0] != "LEKA_SNAPSHOT_TASK" or row[1] != owner["pid"]
                or not re.fullmatch(r"[1-9][0-9]{0,19}", row[2])
                or not re.fullmatch(r"[1-9][0-9]{0,19}", row[3])
                or row[2] in affinities or parse_cpu_list(row[4]) is None):
            return None
        affinities[row[2]], generations[row[2]] = row[4], row[3]
    if generations.get(owner["pid"]) != owner["start_ticks"]:
        return None
    return {"root_generation_verified": True, "expected_rpc_owner": dict(owner),
            "thread_start_ticks": generations, "thread_cpus_allowed_list": affinities,
            "main_thread_cpus_allowed_list": affinities[owner["pid"]],
            "rpc_task_count": len(affinities),
            "scope": "post_measure_owned_rpc_root_tasks_only",
            "task_census_atomic": False, "measured_execution_affinity_proven": False}


def scoped_stat_command(owner: dict, *, duration_s: float, events: str) -> str:
    expected = (f"{owner['binary']} server --host=0.0.0.0 --port={owner['port']} "
                f"--port-end={owner['port'] + 1}")
    assignments = {
        "root": owner["pid"], "root_ticks": owner["start_ticks"],
        "boot": owner["boot_id"], "expected": expected,
        "duration": str(duration_s), "events": events,
    }
    prefix = "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in assignments.items())
    return prefix + r'''
start_ticks() {
    record=$(cat "/proc/$1/stat" 2>/dev/null) || return 1
    rest=${record##*) }; set -- $rest
    [ "$#" -ge 20 ] || return 1
    shift 19; printf '%s\n' "$1"
}
root_matches() {
    [ "$(cat /proc/sys/kernel/random/boot_id)" = "$boot" ] || return 1
    [ "$(start_ticks "$root")" = "$root_ticks" ] || return 1
    command=$(tr '\000' ' ' < "/proc/$root/cmdline") || return 1
    [ "${command% }" = "$expected" ]
}
root_matches || exit 91
table=$(ps -A -o PID,PPID) || exit 92
descendants=$(printf '%s\n' "$table" | awk -v root="$root" '
    NR==1 {if ($1!="PID" || $2!="PPID") exit 1; next}
    NF!=2 || $1!~/^[0-9]+$/ || $2!~/^[0-9]+$/ {bad=1; exit}
    {parent[$1]=$2; lines++; if (lines>32768) {bad=1; exit}}
    END {
        if (bad || !(root in parent)) exit 1
        selected[root]=1
        for (step=0; step<16; step++) {
            changed=0
            for (pid in parent) if (selected[parent[pid]] && !selected[pid]) {
                selected[pid]=1; changed=1
            }
            if (!changed) break
        }
        count=0; for (pid in selected) if (selected[pid]) count++
        if (count>64 || changed) exit 1
        for (pid in selected) if (selected[pid]) print pid
    }') || exit 92
targets=''; identities=''
for pid in $descendants; do
    ticks=$(start_ticks "$pid") || continue
    case "$ticks" in ''|*[!0-9]*) exit 93;; esac
    targets=${targets:+$targets,}$pid
    identities="$identities $pid:$ticks"
done
[ -n "$targets" ] || exit 93
root_matches || exit 91
printf 'LEKA_PMU_IDENTITIES%s\n' "$identities"
task_count=0
for identity in $identities; do
    pid=${identity%:*}
    for status_file in /proc/$pid/task/[0-9]*/status; do
        [ -r "$status_file" ] || continue
        task_count=$((task_count + 1)); [ "$task_count" -le 256 ] || break
        tid=${status_file%/status}; tid=${tid##*/}
        allowed=$(awk '/^Cpus_allowed_list:/{print $2}' "$status_file")
        [ -n "$allowed" ] && printf 'LEKA_PMU_AFFINITY %s %s %s\n' "$pid" "$tid" "$allowed"
    done
done
simpleperf stat -p "$targets" --duration "$duration" -e "$events" --csv
status=$?
root_matches || exit 91
exited=0
for identity in $identities; do
    pid=${identity%:*}; ticks=${identity#*:}
    if [ -d "/proc/$pid" ]; then
        [ "$(start_ticks "$pid")" = "$ticks" ] || exit 94
    else
        exited=$((exited + 1))
    fi
done
printf 'LEKA_PMU_GENERATION_VERIFIED %s %s %s %s\n' "$root" "$root_ticks" "$boot" "$exited"
exit "$status"
'''


def parse_generation_proof(report: str, owner: dict) -> dict | None:
    starts = [line.split()[1:] for line in report.splitlines() if line.startswith("LEKA_PMU_IDENTITIES ")]
    ends = [line.split()[1:] for line in report.splitlines() if line.startswith("LEKA_PMU_GENERATION_VERIFIED ")]
    if len(starts) != 1 or len(ends) != 1 or len(ends[0]) != 4:
        return None
    ids = starts[0]
    if not 1 <= len(ids) <= 64 or any(re.fullmatch(r"[0-9]+:[0-9]+", value) is None for value in ids):
        return None
    if f"{owner['pid']}:{owner['start_ticks']}" not in ids:
        return None
    if ends[0][:3] != [owner["pid"], owner["start_ticks"], owner["boot_id"]] or not ends[0][3].isdigit():
        return None
    affinity = [line.split()[1:] for line in report.splitlines() if line.startswith("LEKA_PMU_AFFINITY ")]
    pids = {item.split(":")[0] for item in ids}
    if len(affinity) > 256 or any(len(item) != 3 or item[0] not in pids or not item[1].isdigit()
        or re.fullmatch(r"[0-9,-]{1,1024}", item[2]) is None for item in affinity):
        return None
    return {"attached_process_generations": ids, "root_generation_verified": True,
            "exited_processes": int(ends[0][3]),
            "thread_affinity_before_window": [dict(zip(("pid", "tid", "cpus_allowed_list"), item)) for item in affinity],
            "affinity_scope": "allowed_cpu_masks_before_window_not_executed_cpu_utilization",
            "process_coverage": "owned_listener_and_censused_descendants_plus_default_inherit",
            "attachment_race_excluded": False, "measurement_phase_overlap": "unproven"}
