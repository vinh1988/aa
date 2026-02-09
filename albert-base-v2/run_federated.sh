#!/bin/bash

# Determine script location and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
# Corrected to 3 levels up: models/albert-base-v2 -> models -> experiment_new_solution -> FedAvgLS
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../../..")"

echo "Project Root: $PROJECT_ROOT"

# Experiment directory
WORK_DIR="$SCRIPT_DIR/fl-mtl-slms-alberbase-v2-sts-qqp-sst2"

# Kaggle-specific setup
if [ -d "/kaggle" ]; then
    echo "Kaggle environment detected."
    # Define writable directory in Kaggle with the structure user expects
    # fl-mtl... is inside models/albert-base-v2, so let's replicate that structure if needed
    # But user specifically mentioned: /kaggle/working/aa/albert-base-v2
    
    KAGGLE_PROJECT_ROOT="/kaggle/working/aa/albert-base-v2"
    KAGGLE_WORK_DIR="$KAGGLE_PROJECT_ROOT/fl-mtl-slms-alberbase-v2-sts-qqp-sst2"
    
    echo "Copying experiment files to writable directory: $KAGGLE_WORK_DIR"
    
    # Create the directory structure
    mkdir -p "$KAGGLE_WORK_DIR"
    
    # Copy contents of the current WORK_DIR (fl-mtl...) to the new KAGGLE_WORK_DIR
    # Note: cp -r source/* dest/ copies contents. Ensure hidden files are copied if needed.
    # To be safe, copy the directory itself if we were one level up, but we are inside script which is outside work_dir
    # cp -r "$WORK_DIR/." "$KAGGLE_WORK_DIR/"
    cp -r "$WORK_DIR"/* "$KAGGLE_WORK_DIR/"
    
    # Update WORK_DIR to the writable location
    WORK_DIR="$KAGGLE_WORK_DIR"
    
    # Update PROJECT_ROOT to maintain relative path integrity if needed
    # If original project root was ../../.., relative to new work dir it is still ../../.. ?
    # /kaggle/working/aa/albert-base-v2/fl-mtl...
    # ../../.. -> /kaggle/working/aa ? No.
    # ../../.. from fl-mtl is albert-base-v2 -> aa -> working.
    # So PROJECT_ROOT would be /kaggle/working.
    # But source code (other libs) might be in /kaggle/input...
    
    # CRITICAL: We need PYTHONPATH to include the original source code if it's not copied.
    # If src is INSIDE work_dir (which it is: fl-mtl/src), then copying work_dir is sufficient for src.
    # But if there are other dependencies in project root...
    # Let's keep PYTHONPATH pointing to original input for safety, OR add new root.
    
    export PYTHONPATH="$PYTHONPATH:$PROJECT_ROOT"
fi

# Install dependencies if requirements.txt exists
# Install dependencies if requirements.txt exists
if [ -f "$WORK_DIR/requirements.txt" ]; then
    echo "Installing dependencies from $WORK_DIR/requirements.txt..."
    
    # Create a local library directory to ensure isolation from system packages
    LIB_DIR="$WORK_DIR/lib"
    mkdir -p "$LIB_DIR"
    
    # Install dependencies into the local directory
    # using --ignore-installed to ensuring we ignore system packages
    pip install --upgrade --ignore-installed --target "$LIB_DIR" -r "$WORK_DIR/requirements.txt"
    
    # Add local lib to PYTHONPATH (prepend it so it takes precedence)
    export PYTHONPATH="$LIB_DIR:$PYTHONPATH"
    
    # Verify installation
    echo "Verifying installation:"
    # We need to set PYTHONPATH for pip show to work on target dir? No, pip show looks at site-packages.
    # But for python usage:
    python3 -c "import sys; print(sys.path); import transformers; print(f'Transformers file: {transformers.__file__}'); print(f'Transformers version: {transformers.__version__}')"
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

echo "All processes launched!"
echo "Logs are being written to directory: $(pwd)"
echo "Log files:"
ls -l *.log

echo "Tailing server log to check for immediate errors..."
sleep 5
head -n 20 server.log

# Wait for server process to finish
wait $SERVER_PID
