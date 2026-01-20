#!/bin/bash
#
# check_jobs.sh - Monitor running jobs via .live log files
#
# Usage:
#   bash scripts/check_jobs.sh              # Show running jobs
#   bash scripts/check_jobs.sh --cleanup    # Finalize stale .live logs
#

set -euo pipefail

LOGS_DIR="${LOGS_DIR:-logs}"
ACTION="${1:-status}"

count_live_logs() {
    find "${LOGS_DIR}" -name "*.live" 2>/dev/null | wc -l | tr -d ' '
}

show_running_jobs() {
    local count=$(count_live_logs)

    if [[ $count -eq 0 ]]; then
        echo "No running jobs (no .live logs found)"
        return 0
    fi

    echo "Running jobs: $count"
    echo ""
    echo "Live logs:"
    find "${LOGS_DIR}" -name "*.live" -exec ls -lh {} \; 2>/dev/null | \
        awk '{printf "  %s %s  %s\n", $9, $5, $6" "$7" "$8}'
}

cleanup_live_logs() {
    local count=$(count_live_logs)

    if [[ $count -eq 0 ]]; then
        echo "No .live logs to clean up"
        return 0
    fi

    echo "Found $count .live log file(s)"
    echo "Renaming to final names..."

    local renamed=0
    for live_log in "${LOGS_DIR}"/*.live; do
        [[ -f "$live_log" ]] || continue

        # Remove .live extension
        final_log="${live_log%.live}"

        echo "  $live_log -> $final_log"
        mv "$live_log" "$final_log"
        renamed=$((renamed + 1))
    done

    echo "Renamed $renamed file(s)"
}

case "$ACTION" in
    status|--status|-s)
        show_running_jobs
        ;;
    cleanup|--cleanup|-c)
        cleanup_live_logs
        ;;
    *)
        echo "Usage: $0 [status|cleanup]"
        echo ""
        echo "Commands:"
        echo "  status   - Show running jobs (default)"
        echo "  cleanup  - Finalize all .live logs"
        exit 1
        ;;
esac
