#!/usr/bin/env bash
set -euo pipefail

READY_URL="http://localhost:8000/health"
TIMEOUT=60

usage() {
    echo "Usage: $0 --action start|terminate"
    exit 1
}

ACTION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
    --action) ACTION="${2:-}"; shift 2 || usage;;
    -h|--help) usage;;
    *) echo "Unknown option: $1"; usage;;
    esac
done

command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1 || {
    echo "Error: Docker with compose V2 is not installed or not in PATH.";
    exit 1;
}

command -v curl >/dev/null 2>&1 || {
    echo "Error: curl is not installed or not in PATH.";
    exit 1;
}

start() {
    echo "Building and starting the Docker containers..."
    docker compose up --build -d
    echo "Waiting for the services to be ready (max ${TIMEOUT}s)..."
    for ((i = 1; i <= TIMEOUT; i++)); do
        if curl -sf "$READY_URL" >/dev/null; then
            echo "Ready after $i seconds."
            return 0
        fi
        sleep 1
    done
    echo "Error: Services did not become ready within ${TIMEOUT} seconds. Recent logs for debugging:"
    docker compose logs --tail=50
    exit 1
}

terminate() {
    echo "Stopping and removing the Docker containers, volumes, and networks..."
    docker compose down -v --remove-orphans
    echo "Cleanup complete."
}

case "$ACTION" in
start) start ;;
terminate) terminate ;;
*) usage ;;
esac
