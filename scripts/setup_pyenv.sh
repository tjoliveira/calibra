#!/usr/bin/env bash
set -euo pipefail

# Minimal pyenv setup:
# - Install pyenv (if missing)
# - Install Python 3.10
# - Create a project virtualenv
# - Run setup.py in that environment

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_VERSION="3.10.14"
PYENV_NAME="$(basename "$PROJECT_DIR")-py310"

#Update and install required libraries
sudo apt-get update -y
sudo apt-get install -y \
  build-essential make \
  libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev \
  libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev \
  libffi-dev liblzma-dev libgdbm-dev libnss3-dev ca-certificates curl

# Install pyenv if missing
if [ ! -d "${HOME}/.pyenv" ]; then
  curl -fsSL https://pyenv.run | bash
fi

# Initialize pyenv for this shell only
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

# Persist pyenv init for future shells (idempotent)
if ! grep -qs 'PYENV_ROOT=.*\.pyenv' "${HOME}/.bashrc"; then
  {
    echo ''
    echo '# >>> pyenv initialization >>>'
    echo 'export PYENV_ROOT="$HOME/.pyenv"'
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"'
    echo 'eval "$(pyenv init -)"'
    echo 'eval "$(pyenv virtualenv-init -)"'
    echo '# <<< pyenv initialization <<<'
  } >> "${HOME}/.bashrc"
fi

# Sanity check
if ! command -v pyenv >/dev/null 2>&1; then
  echo "ERROR: pyenv not found on PATH after initialization. Aborting." >&2
  exit 1
fi

# Ensure the requested Python is available
pyenv install -s "$PYTHON_VERSION"

# Create the virtualenv if it doesn't exist
if ! pyenv virtualenvs --bare | grep -Fxq "$PYENV_NAME"; then
  pyenv virtualenv "$PYTHON_VERSION" "$PYENV_NAME"
fi

# Use this env for the project and current shell
(cd "$PROJECT_DIR" && pyenv local "$PYENV_NAME")
pyenv shell "$PYENV_NAME"

# Get the Python path directly from the virtualenv location
# pyenv virtualenvs are stored in $PYENV_ROOT/versions/$VIRTUALENV_NAME/bin/python
PYTHON_PATH="$PYENV_ROOT/versions/$PYENV_NAME/bin/python"

echo "PYTHON_PATH: $PYTHON_PATH"
if [[ ! -f "$PYTHON_PATH" ]]; then
    echo "ERROR: Could not find Python for virtualenv '$PYENV_NAME'" >&2
    echo "Expected path: $PYTHON_PATH" >&2
    echo "Available virtualenvs:" >&2
    pyenv virtualenvs >&2
    exit 1
fi

echo "Using Python: $PYTHON_PATH"

# Verify what 'which python' resolves to (should be shim, but shim resolves to our Python)
if command -v python >/dev/null 2>&1; then
    WHICH_PYTHON=$(which python)
    echo "Note: 'which python' shows: $WHICH_PYTHON (this is the pyenv shim, which is correct)"
    # Verify the shim resolves to our Python
    RESOLVED_PYTHON=$(python -c "import sys; print(sys.executable)" 2>/dev/null || echo "")
    if [[ -n "$RESOLVED_PYTHON" ]]; then
        echo "The shim resolves to: $RESOLVED_PYTHON"
        if [[ "$RESOLVED_PYTHON" == "$PYTHON_PATH" ]]; then
            echo "✓ Shims are correctly configured"
        else
            echo "⚠ WARNING: Shims may not be resolving correctly!"
        fi
    fi
fi

# Verify Python is from the virtualenv
echo ""
echo "Direct Python path verification:"
"$PYTHON_PATH" -c "import sys; print(f'Python executable: {sys.executable}')"
"$PYTHON_PATH" -c "import site; print(f'Site-packages: {site.getsitepackages()}')"

# Get pip path from the virtualenv
PIP_PATH="$PYENV_ROOT/versions/$PYENV_NAME/bin/pip"
if [[ ! -f "$PIP_PATH" ]]; then
    echo "WARNING: pip not found at $PIP_PATH, will use python -m pip"
    PIP_PATH="$PYTHON_PATH -m pip"
else
    echo "Using pip: $PIP_PATH"
fi

# Install with pip using the full path to ensure we're in the right environment
cd "$PROJECT_DIR"

# Disable user site-packages to ensure everything installs in the virtualenv
export PYTHONNOUSERSITE=1

# Verify where pip will install packages
echo "Checking pip installation location..."
"$PYTHON_PATH" -m pip show pip | grep -i "location" || "$PYTHON_PATH" -c "import pip; import os; print(f'Pip location: {os.path.dirname(pip.__file__)}')"

"$PYTHON_PATH" -m pip install --upgrade pip setuptools wheel
"$PYTHON_PATH" -m pip install -r requirements.txt

MAX_JOBS=4 "$PYTHON_PATH" -m pip -v install "flash-attn==2.3.6" --no-build-isolation
