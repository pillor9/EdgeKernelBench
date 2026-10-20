#!/usr/bin/env bash
# Refresh the RPC process recorded in this run's state directory.
set -euo pipefail

THREADS="${VIVO_NUM_THREADS:?VIVO_NUM_THREADS is required}"
[[ "${THREADS}" == "8" ]] || { echo 'Vivo requires exactly eight workers' >&2; exit 1; }
SERIAL="${VIVO_SERIAL:?VIVO_SERIAL is required}"
PORT="${VIVO_RPC_PORT:-9490}"
[[ "${PORT}" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65535 )) || exit 1
PORT_END="$((PORT + 1))"
STATE_DIR="${LEKA_VIVO_STATE_DIR:-${TMPDIR:-/tmp}/LEKA/vivo_rpc}"
HOST_PID_FILE="${STATE_DIR}/adb_shell.pid"
HOST_LOG_FILE="${STATE_DIR}/adb_shell.log"
OWNER_FILE="${STATE_DIR}/rpc_owner.tsv"
DEVICE_LIB_DIR="${LEKA_VIVO_DEVICE_LIB_DIR:-/data/local/tmp/vivo_tvm}"
[[ "${DEVICE_LIB_DIR}" =~ ^/[a-zA-Z0-9_./-]+$ ]] || {
    echo 'LEKA_VIVO_DEVICE_LIB_DIR must be an absolute path without shell characters' >&2; exit 1;
}
[[ "${SERIAL}" =~ ^[a-zA-Z0-9_.:-]+$ ]] || exit 1
DEVICE_RPC_BIN=${DEVICE_LIB_DIR}/tvm_rpc
SERVING_CPU_ID=7
SERVING_TASKSET_MASK=80

# Validate remote arguments and match device PID, start time, and boot ID.
remote_rpc() {
    local action="$1" pid="${2:-0}" ticks="${3:-0}" boot="${4:-none}"
    [[ "${action}" =~ ^[a-z_]+$ && "${pid}" =~ ^[0-9]+$ && "${ticks}" =~ ^[0-9]+$ && "${boot}" =~ ^[a-zA-Z0-9-]+$ ]] || return 1
    adb -s "${SERIAL}" shell "sh -s -- ${action} ${DEVICE_RPC_BIN} ${PORT} ${PORT_END} ${pid} ${ticks} ${boot}" <<'REMOTE_RPC'
set -eu
action=$1; binary=$2; port=$3; port_end=$4; owner_pid=$5; owner_ticks=$6; owner_boot=$7
expected="$binary server --host=0.0.0.0 --port=$port --port-end=$port_end"
fail() { echo "Vivo RPC ownership: $*" >&2; exit 1; }
start_ticks() {
    record=$(cat "/proc/$1/stat") || return 1
    rest=${record##*) }; set -- $rest
    [ "$#" -ge 20 ] || return 1
    shift 19; printf '%s\n' "$1"
}
identity() {
    [ -d "/proc/$owner_pid" ] || return 2
    [ "$(cat /proc/sys/kernel/random/boot_id)" = "$owner_boot" ] || fail 'device boot changed; refusing PID cleanup'
    [ "$(start_ticks "$owner_pid")" = "$owner_ticks" ] || fail 'device PID was reused; refusing cleanup'
    command=$(tr '\000' ' ' < "/proc/$owner_pid/cmdline") || fail 'cannot read command identity'
    command=${command% }
    [ "$command" = "$expected" ] || fail 'owned PID command changed; refusing cleanup'
    owner_uid=$(awk '/^Uid:/{print $2}' "/proc/$owner_pid/status")
    [ "$owner_uid" = "$(id -u)" ] || fail 'owned PID has another UID'
}
case "$action" in
    launch)
        boot=$(cat /proc/sys/kernel/random/boot_id)
        ticks=$(start_ticks "$$")
        printf 'LEKA_RPC_GENERATION %s %s %s\n' "$$" "$ticks" "$boot"
        cd "${binary%/*}"
        exec env TVM_NUM_THREADS=8 TVM_BIND_THREADS=1 LD_LIBRARY_PATH="${binary%/*}:/vendor/lib64:/system/lib64:/odm/lib64" taskset 80 "$binary" server --host=0.0.0.0 --port="$port" --port-end="$port_end"
        ;;
    terminate)
        if identity; then :; else rc=$?; [ "$rc" -eq 2 ] && exit 0; exit "$rc"; fi
        # Refuse cleanup while any RPC session child is running. TVM can fork
        # from a listener thread, so inspect all processes rather than one thread.
        process_table=$(ps -A -o PID,PPID) || fail 'cannot inspect RPC session children'
        printf '%s\n' "$process_table" | awk 'NR==1 {if ($1!="PID" || $2!="PPID") exit 1; next} NF!=2 || $1!~/^[0-9]+$/ || $2!~/^[0-9]+$/ {exit 1}' || fail 'invalid process table'
        children=$(printf '%s\n' "$process_table" | awk -v p="$owner_pid" 'NR>1 && $2==p {print $1}')
        [ -z "$children" ] || fail 'owned RPC has active session children'
        identity || exit 0
        kill -TERM "$owner_pid"
        attempts=0
        while [ -d "/proc/$owner_pid" ] && [ "$attempts" -lt 10 ]; do
            sleep 0.2; attempts=$((attempts + 1))
        done
        if [ -d "/proc/$owner_pid" ]; then identity && kill -KILL "$owner_pid"; fi
        ;;
    inspect)
        identity
        awk '/^Cpus_allowed_list:/{print $2}' "/proc/$owner_pid/task/$owner_pid/status"
        ;;
    port_free)
        # An occupied port may belong to another run; never replace its listener.
        sockets=$(ss -ltn) || fail 'cannot inspect listening TCP ports'
        if printf '%s\n' "$sockets" | awk -v p="$port" 'NR>1 && $4 ~ (":" p "$") {found=1} END {exit !found}'; then
            fail "configured port $port is occupied by an unowned listener"
        fi
        ;;
    *) fail 'invalid lifecycle action' ;;
esac
REMOTE_RPC
}

mkdir -p "${STATE_DIR}"
[[ ! -L "${OWNER_FILE}" && ! -L "${HOST_PID_FILE}" && ! -L "${HOST_LOG_FILE}" ]] || {
    echo 'Vivo RPC state must not contain symlinks' >&2; exit 1;
}
rm -f "${HOST_PID_FILE}"
adb -s "${SERIAL}" get-state >/dev/null
adb -s "${SERIAL}" shell 'command -v taskset >/dev/null && command -v ss >/dev/null'
for cpu_id in 0 1 2 3 4 5 6 7; do
    adb -s "${SERIAL}" shell "test -r /sys/devices/system/cpu/cpu${cpu_id}/cpufreq/scaling_cur_freq && test -r /sys/devices/system/cpu/cpu${cpu_id}/cpufreq/scaling_governor"
done
thermal_count="$(adb -s "${SERIAL}" shell 'for z in /sys/class/thermal/thermal_zone*; do [ -r "$z/type" ] && [ -r "$z/temp" ] && echo "$z"; done' | wc -l | tr -d ' ')"
[[ "${thermal_count}" =~ ^[0-9]+$ ]] && (( thermal_count > 0 )) || exit 1

if [[ -f "${OWNER_FILE}" ]]; then
    IFS=$'\t' read -r old_serial old_port old_binary phone_pid start_ticks boot_id extra < "${OWNER_FILE}"
    [[ "${old_serial}" == "${SERIAL}" && "${old_port}" == "${PORT}" && "${old_binary}" == "${DEVICE_RPC_BIN}" && -z "${extra}" ]] || {
        echo 'Vivo RPC owner receipt does not match this device/binary/port' >&2; exit 1;
    }
    remote_rpc terminate "${phone_pid}" "${start_ticks}" "${boot_id}"
    rm "${OWNER_FILE}"
fi
remote_rpc port_free
# Refuse to replace another serial's host forward (or another remote endpoint).
forwards=$(adb forward --list)
foreign_forward=$(printf '%s\n' "${forwards}" | awk -v s="${SERIAL}" -v p="tcp:${PORT}" '$2==p && ($1!=s || $3!=p) {print}')
[[ -z "${foreign_forward}" ]] || { echo 'Vivo RPC host port has a foreign ADB forward' >&2; exit 1; }
adb -s "${SERIAL}" forward --no-rebind "tcp:${PORT}" "tcp:${PORT}" >/dev/null || {
    # An existing identical forward is safe to reuse; never remove it blindly.
    printf '%s\n' "${forwards}" | awk -v s="${SERIAL}" -v p="tcp:${PORT}" '$1==s && $2==p && $3==p {found=1} END {exit !found}' || exit 1
}

ready=false
for launch_attempt in 1 2 3 4; do
    remote_rpc port_free
    touch "${HOST_LOG_FILE}"
    log_offset=$(wc -c < "${HOST_LOG_FILE}")
    printf '\n[vivo_refresh_rpc] launching owned device RPC attempt %s\n' "${launch_attempt}" >> "${HOST_LOG_FILE}"
    # The background function owns a new remote sh whose PID is retained by exec.
    remote_rpc launch </dev/null >>"${HOST_LOG_FILE}" 2>&1 &
    host_pid=$!
    sleep 2
    generation_log="$(tail -c +$((log_offset + 1)) "${HOST_LOG_FILE}")"
    marker=$(printf '%s\n' "${generation_log}" | tr -d '\r' | awk '$1=="LEKA_RPC_GENERATION" {print $2, $3, $4; exit}')
    read -r phone_pid start_ticks boot_id <<< "${marker}"
    [[ "${phone_pid}" =~ ^[0-9]+$ && "${start_ticks}" =~ ^[0-9]+$ && "${boot_id}" =~ ^[a-zA-Z0-9-]+$ ]] || {
        echo 'Vivo RPC launch did not publish its process identity; no process was signalled' >&2; exit 1;
    }
    # Record process identity before readiness, so failed launches remain traceable.
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${SERIAL}" "${PORT}" "${DEVICE_RPC_BIN}" "${phone_pid}" "${start_ticks}" "${boot_id}" > "${OWNER_FILE}.pending"
    mv "${OWNER_FILE}.pending" "${OWNER_FILE}"
    if kill -0 "${host_pid}" 2>/dev/null &&
       grep -Eq "bind to 0[.]0[.]0[.]0:${PORT}([[:space:]]|$)" <<< "${generation_log}" &&
       ! grep -Eq 'Bind failed|bind to 0[.]0[.]0[.]0:-1' <<< "${generation_log}"; then
        serving_cpus=$(remote_rpc inspect "${phone_pid}" "${start_ticks}" "${boot_id}" | tr -d '\r')
        [[ "${serving_cpus}" == "${SERVING_CPU_ID}" ]] || { echo 'Vivo RPC serving affinity mismatch' >&2; exit 1; }
        ready=true; break
    fi
    printf '[vivo_refresh_rpc] attempt %s not ready; retaining evidence\n' "${launch_attempt}" >> "${HOST_LOG_FILE}"
    remote_rpc terminate "${phone_pid}" "${start_ticks}" "${boot_id}"
    rm "${OWNER_FILE}"
    if (( launch_attempt < 4 )); then sleep 20; fi
done
[[ "${ready}" == true ]] || { echo 'Vivo RPC did not bind the requested port after four attempts' >&2; exit 1; }
echo "${host_pid}" > "${HOST_PID_FILE}"
printf '{"serial":"%s","port":%s,"host_pid":%s,"phone_pid":%s,"same_uid":true,"serving_cpu_id":%s,"taskset_mask":"%s","thermal_zone_count":%s}\n' \
    "${SERIAL}" "${PORT}" "${host_pid}" "${phone_pid}" "${SERVING_CPU_ID}" "${SERVING_TASKSET_MASK}" "${thermal_count}"
