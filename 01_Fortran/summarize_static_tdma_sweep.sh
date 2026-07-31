#!/bin/bash

set -euo pipefail

RESULT_ROOT="${1:?usage: summarize_static_tdma_sweep.sh RESULT_ROOT}"
AB_CSV="${RESULT_ROOT}/static_tdma_ab_comparison.csv"
COARSE_CSV="${RESULT_ROOT}/coarse_phase_comparison.csv"
TDMA_CSV="${RESULT_ROOT}/tdma_phase_ab_comparison.csv"

CASES=(
    "1:default"
    "2:default"
    "4:pair01_23"
    "4:pair02_13"
    "4:pair03_12"
)

extract_time()
{
    awk '/Average Poisson solve time:/ {value=$NF} END {print value}' "$1"
}

extract_rms()
{
    awk '/RMS error against analytic solution:/ {value=$NF} END {print value}' "$1"
}

median()
{
    sort -g | awk '
        {x[NR]=$1}
        END {
            if (NR == 0) exit 1
            if (NR % 2) print x[(NR+1)/2]
            else print 0.5*(x[NR/2]+x[NR/2+1])
        }
    '
}

variant_median()
{
    local root="$1"
    mapfile -t values < <(
        find "${root}/perf" -name perf_solver.log -type f -print \
            | sort \
            | while read -r log; do extract_time "${log}"; done
    )
    [[ "${#values[@]}" -gt 0 ]] || return 1
    printf '%s\n' "${values[@]}" | median
}

variant_rms()
{
    local root="$1"
    local log
    log="$(
        find "${root}/perf" -name perf_solver.log -type f -print \
            | sort | head -n 1
    )"
    [[ -n "${log}" ]] || return 1
    extract_rms "${log}"
}

printf '%s\n' \
    'np,placement,dynamic_median_s,static_median_s,speedup,improvement_pct,dynamic_rms,static_rms,rms_abs_delta,correctness' \
    > "${AB_CSV}"
printf '%s\n' \
    'np,placement,operator,phase_id,phase_label,rank_min_s,rank_avg_s,rank_max_s' \
    > "${COARSE_CSV}"
printf '%s\n' \
    'np,placement,dynamic_tdma_rankmax_s,static_tdma_rankmax_s,tdma_speedup,tdma_improvement_pct' \
    > "${TDMA_CSV}"

echo
echo "================ Static-operator TDMA A/B ================="
printf '%-4s %-12s %13s %13s %10s %10s %11s\n' \
    NP PLACEMENT DYNAMIC_S STATIC_S SPEEDUP IMPROVE CORRECTNESS

np1_static=""
best_np4_static=""
best_np4_placement=""

for item in "${CASES[@]}"; do
    np="${item%%:*}"
    placement="${item#*:}"
    case_root="${RESULT_ROOT}/np${np}/${placement}"
    dynamic_root="${case_root}/dynamic"
    static_root="${case_root}/static"

    dynamic_median="$(variant_median "${dynamic_root}")" || {
        echo "[ERROR] Missing dynamic performance logs: ${case_root}" >&2
        exit 2
    }
    static_median="$(variant_median "${static_root}")" || {
        echo "[ERROR] Missing static performance logs: ${case_root}" >&2
        exit 3
    }
    dynamic_rms="$(variant_rms "${dynamic_root}")"
    static_rms="$(variant_rms "${static_root}")"

    read -r speedup improvement rms_delta correctness <<EOF
$(awk -v old="${dynamic_median}" -v new="${static_median}" \
      -v r1="${dynamic_rms}" -v r2="${static_rms}" '
    BEGIN {
        speedup = old/new
        improvement = 100.0*(old-new)/old
        delta = r1-r2
        if (delta < 0.0) delta = -delta
        correctness = (delta <= 1.0e-12 && r1 < 1.0e-3 && r2 < 1.0e-3) \
            ? "PASS" : "FAIL"
        printf "%.8f %.6f %.12e %s\n", \
            speedup, improvement, delta, correctness
    }
')
EOF

    printf '%-4s %-12s %13.6e %13.6e %9.4fx %9.2f%% %11s\n' \
        "${np}" "${placement}" "${dynamic_median}" "${static_median}" \
        "${speedup}" "${improvement}" "${correctness}"

    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${np}" "${placement}" "${dynamic_median}" "${static_median}" \
        "${speedup}" "${improvement}" "${dynamic_rms}" "${static_rms}" \
        "${rms_delta}" "${correctness}" >> "${AB_CSV}"

    [[ "${correctness}" == PASS ]] || {
        echo "[ERROR] RMS mismatch: np=${np} placement=${placement}" >&2
        exit 4
    }

    for operator in dynamic static; do
        coarse_log="${case_root}/${operator}/coarse/coarse_solver.log"
        [[ -f "${coarse_log}" ]] || {
            echo "[ERROR] Missing coarse log: ${coarse_log}" >&2
            exit 5
        }
        awk -v np="${np}" -v placement="${placement}" \
            -v operator="${operator}" '
            /^\[COARSE\] [0-9][0-9]/ {
                label=$3
                for (i=4; i<=NF-3; i++) label=label " " $i
                gsub(/,/, ";", label)
                print np "," placement "," operator "," $2 "," label "," \
                      $(NF-2) "," $(NF-1) "," $NF
            }
        ' "${coarse_log}" >> "${COARSE_CSV}"
    done

    if [[ "${np}" == 1 ]]; then
        np1_static="${static_median}"
    elif [[ "${np}" == 4 ]]; then
        if [[ -z "${best_np4_static}" ]] \
            || awk -v new="${static_median}" -v old="${best_np4_static}" \
                'BEGIN {exit !(new < old)}'; then
            best_np4_static="${static_median}"
            best_np4_placement="${placement}"
        fi
    fi
done

echo
echo "================ Coarse TDMA-Z phase A/B =================="
printf '%-4s %-12s %13s %13s %10s %10s\n' \
    NP PLACEMENT DYNAMIC_S STATIC_S SPEEDUP IMPROVE
for item in "${CASES[@]}"; do
    np="${item%%:*}"
    placement="${item#*:}"
    case_root="${RESULT_ROOT}/np${np}/${placement}"
    dynamic_log="${case_root}/dynamic/coarse/coarse_solver.log"
    static_log="${case_root}/static/coarse/coarse_solver.log"
    dynamic_tdma="$(
        awk '/^\[COARSE\] 04 / {value=$NF} END {print value}' \
            "${dynamic_log}"
    )"
    static_tdma="$(
        awk '/^\[COARSE\] 04 / {value=$NF} END {print value}' \
            "${static_log}"
    )"
    [[ -n "${dynamic_tdma}" && -n "${static_tdma}" ]] || {
        echo "[ERROR] Missing coarse TDMA phase: ${case_root}" >&2
        exit 7
    }
    read -r tdma_speedup tdma_improvement <<EOF
$(awk -v old="${dynamic_tdma}" -v new="${static_tdma}" '
    BEGIN {
        printf "%.8f %.6f\n", old/new, 100.0*(old-new)/old
    }
')
EOF
    printf '%-4s %-12s %13.6e %13.6e %9.4fx %9.2f%%\n' \
        "${np}" "${placement}" "${dynamic_tdma}" "${static_tdma}" \
        "${tdma_speedup}" "${tdma_improvement}"
    printf '%s,%s,%s,%s,%s,%s\n' \
        "${np}" "${placement}" "${dynamic_tdma}" "${static_tdma}" \
        "${tdma_speedup}" "${tdma_improvement}" >> "${TDMA_CSV}"
done

[[ -n "${np1_static}" && -n "${best_np4_static}" ]] || {
    echo "[ERROR] Could not determine NP1 or best NP4 time." >&2
    exit 6
}

read -r scaling efficiency np4_faster <<EOF
$(awk -v t1="${np1_static}" -v t4="${best_np4_static}" '
    BEGIN {
        scaling=t1/t4
        efficiency=100.0*scaling/4.0
        faster=(t4 < t1) ? "PASS" : "NOT_YET"
        printf "%.8f %.6f %s\n", scaling, efficiency, faster
    }
')
EOF

echo "-----------------------------------------------------------"
echo "Best NP4 static placement : ${best_np4_placement}"
printf 'NP1 static               : %.6e s/solve\n' "${np1_static}"
printf 'Best NP4 static          : %.6e s/solve\n' "${best_np4_static}"
printf 'NP1/NP4 scaling          : %.4fx\n' "${scaling}"
printf 'Four-GPU efficiency      : %.2f%%\n' "${efficiency}"
echo "Primary target NP4 < NP1 : ${np4_faster}"
echo "A/B CSV                   : ${AB_CSV}"
echo "Coarse phase CSV          : ${COARSE_CSV}"
echo "TDMA phase A/B CSV        : ${TDMA_CSV}"
echo "==========================================================="
