#!/bin/bash

# Determine script location and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Corrected to 3 levels up: models/albert-base-v2 -> models -> experiment_new_solution -> FedAvgLS
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"

echo "Project Root: $PROJECT_ROOT"

# Experiment directory
WORK_DIR="$SCRIPT_DIR/fl-mtl-slms-alberbase-v2-sts-qqp-sst2"

# Install dependencies if requirements.txt exists
if [ -f "$WORK_DIR/requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r "$WORK_DIR/requirements.txt"
fi

# Function to handle cleanup on exit
cleanup() {
    echo "Stopping all processes..."
    # Kill background jobs
    kill $(jobs -p) 2>/dev/null
    wait
    echo "Done."
}
trap cleanup SIGINT SIGTERM EXIT

# Set PYTHONPATH
export PYTHONPATH=$PROJECT_ROOT

# Navigate to work directory
cd "$WORK_DIR" || exit

# Terminal 1: Server
echo "Starting Server..."
python federated_main.py --mode server --config federated_config.yaml > server.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

sleep 20

# Terminal 2: SST2 Client
echo "Starting SST2 Client..."
python federated_main.py --mode client --client_id sst2_client --tasks sst2 > sst2_client.log 2>&1 &
CLIENT1_PID=$!
echo "SST2 Client PID: $CLIENT1_PID"

# Terminal 3: QQP Client
echo "Starting QQP Client..."
python federated_main.py --mode client --client_id qqp_client --tasks qqp > qqp_client.log 2>&1 &
CLIENT2_PID=$!
echo "QQP Client PID: $CLIENT2_PID"

# Terminal 4: STSB Client
echo "Starting STSB Client..."
python federated_main.py --mode client --client_id stsb_client --tasks stsb > stsb_client.log 2>&1 &
CLIENT3_PID=$!
echo "STSB Client PID: $CLIENT3_PID"

echo "All processes launched! Logs are being written to server.log, sst2_client.log, qqp_client.log, and stsb_client.log."

# Wait for server process to finish
wait $SERVER_PID
