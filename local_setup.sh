#!/bin/bash

echo "========================================================="
echo "Setting up local environment"

# Initialize variables
VENV_DIR="masai"
INSTALL_REQS=false
EXEC_TYPE=""

# Parse command-line flags (fixed: r does not take an argument)
echo "Parsing flags: $@"
while getopts "re:v:" flag; do
    echo "Flag: -$flag, Argument: $OPTARG"
    case $flag in
        r)
            INSTALL_REQS=true
            ;;
        e)
            EXEC_TYPE="$OPTARG"
            ;;
        v)
            VENV_DIR="$OPTARG"
            ;;
        *)
            echo "Usage: $0 [-r] -e <execution_type> [-v <venv_name>]"
            echo "  -r: Install dependencies from requirements.txt"
            echo "  -e: Specify execution type (hierarchical, decentralized, sequential)"
            echo "  -v: Specify virtual environment name (default: masai)"
            exit 1
            ;;
    esac
done

# Check parsed values
echo "INSTALL_REQS=$INSTALL_REQS"
echo "EXEC_TYPE=$EXEC_TYPE"
echo "VENV_DIR=$VENV_DIR"

# Check if execution type is provided
if [ -z "$EXEC_TYPE" ]; then
    echo "Error: Execution type not specified. Use -e flag with 'hierarchical', 'decentralized', or 'sequential'."
    echo "Usage: $0 [-r] -e <execution_type> [-v <venv_name>]"
    deactivate 2>/dev/null
    exit 1
fi

# Validate execution type
case "$EXEC_TYPE" in
    "hierarchical"|"decentralized"|"sequential")
        echo "Execution type set to: $EXEC_TYPE"
        ;;
    *)
        echo "Invalid execution type: $EXEC_TYPE. Must be 'hierarchical', 'decentralized', or 'sequential'."
        deactivate 2>/dev/null
        exit 1
        ;;
esac

# Check if virtual environment directory exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists, activating it..."
    . "$VENV_DIR/bin/activate"
else
    echo "Virtual environment '$VENV_DIR' not found, creating it..."
    python3 -m venv "$VENV_DIR"
    if [ $? -eq 0 ]; then
        echo "Virtual environment created successfully, activating it..."
        . "$VENV_DIR/bin/activate"
    else
        echo "Failed to create virtual environment '$VENV_DIR'"
        exit 1
    fi
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment '$VENV_DIR' could not be activated"
    exit 1
else
    echo "Virtual environment '$VENV_DIR' activated"
fi

# Install requirements if flag is set
if [ "$INSTALL_REQS" = true ]; then
    if [ -f "requirements.txt" ]; then
        echo "Installing dependencies from requirements.txt..."
        pip install --upgrade pip
        pip install -r requirements.txt
        if [ $? -eq 0 ]; then
            echo "Dependencies installed successfully"
        else
            echo "Failed to install dependencies"
            deactivate
            exit 1
        fi
    else
        echo "requirements.txt not found, skipping installation"
    fi
else
    echo "Skipping installation of requirements.txt (use -r flag to install)"
fi

# Run masai.py with the specified execution type
if [ -f "masai.py" ]; then
    echo "Running masai.py with execution type '$EXEC_TYPE'..."
    python masai.py "$EXEC_TYPE"
    if [ $? -eq 0 ]; then
        echo "masai.py executed successfully"
    else
        echo "Error running masai.py"
    fi
else
    echo "masai.py not found"
fi

# Deactivate virtual environment
deactivate
echo "Virtual environment '$VENV_DIR' deactivated"
echo "========================================================="