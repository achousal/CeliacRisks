#!/bin/bash
#
# HPC Setup Script for CeliacRisks v1.0.0
#
# Purpose: Automated setup of Python environment and package installation on HPC
#
# Usage:
#   bash scripts/hpc_setup.sh
#
# Requirements:
#   - Python 3.8+
#   - Git (for version tracking)
#   - ~2 GB disk space for virtual environment
#
# Author: Andres Chousal
# Date: 2026-01-19

set -e  # Exit on error
set -u  # Exit on undefined variable

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

# Logging functions
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Print header
echo "=========================================="
echo "  CeliacRisks HPC Setup (v1.0.0)"
echo "=========================================="
echo ""

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found. Run this script from the analysis/ directory."
fi

info "Current directory: $(pwd)"

# Check Python version
info "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]; }; then
    error "Python 3.8+ required, found $PYTHON_VERSION. Load a newer Python module."
fi

success "Python $PYTHON_VERSION detected"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    warning "Virtual environment 'venv' already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        info "Removing existing virtual environment..."
        rm -rf venv
    else
        info "Keeping existing virtual environment."
        info "Activating environment..."
        source venv/bin/activate
        info "Upgrading pip..."
        pip install --upgrade pip setuptools wheel > /dev/null 2>&1
        info "Installing/updating package..."
        pip install -e . > /dev/null 2>&1
        success "Package updated successfully"
        echo ""
        echo "=========================================="
        echo "  Setup Complete!"
        echo "=========================================="
        echo ""
        echo "To activate the environment:"
        echo "  source venv/bin/activate"
        echo ""
        echo "To verify installation:"
        echo "  ced --help"
        echo ""
        exit 0
    fi
fi

# Create virtual environment
info "Creating virtual environment..."
python3 -m venv venv
success "Virtual environment created"

# Activate virtual environment
info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
info "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
success "pip upgraded to version $(pip --version | awk '{print $2}')"

# Install package
info "Installing CeliacRisks package (this may take 2-3 minutes)..."
pip install -e . > install.log 2>&1

if [ $? -eq 0 ]; then
    success "Package installed successfully"
    rm install.log
else
    error "Package installation failed. Check install.log for details."
fi

# Verify installation
info "Verifying installation..."
if command -v ced &> /dev/null; then
    success "ced command is available"
else
    error "ced command not found. Installation may have failed."
fi

# Check CLI functionality
info "Testing CLI..."
ced --version > /dev/null 2>&1
if [ $? -eq 0 ]; then
    success "CLI is functional"
else
    error "CLI test failed"
fi

# Create required directories
info "Creating output directories..."
mkdir -p logs
mkdir -p splits_production
mkdir -p results_production
success "Output directories created"

# Check for data file
info "Checking for input data file..."
if [ -f "../Celiac_dataset_proteomics.csv" ]; then
    DATA_SIZE=$(du -h ../Celiac_dataset_proteomics.csv | cut -f1)
    success "Data file found (size: $DATA_SIZE)"
else
    warning "Data file not found at ../Celiac_dataset_proteomics.csv"
    echo "  You need to copy it from shared storage before running the pipeline."
fi

# Record environment
info "Recording package versions..."
pip freeze > requirements_frozen_$(date +%Y%m%d).txt
success "Package versions saved to requirements_frozen_$(date +%Y%m%d).txt"

# Record git state
if command -v git &> /dev/null; then
    info "Recording git state..."
    git log -1 --oneline > git_version.txt 2>/dev/null || true
    if [ -f "git_version.txt" ]; then
        success "Git version recorded: $(cat git_version.txt)"
    fi
fi

# Run optional tests
echo ""
read -p "Run test suite to verify installation? (recommended, takes ~2 min) (Y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    info "Running test suite..."
    if pytest tests/ -v --tb=short > test_results.log 2>&1; then
        TEST_PASS=$(grep -c "passed" test_results.log || echo "0")
        success "Tests passed: $TEST_PASS"
        rm test_results.log
    else
        warning "Some tests failed. Check test_results.log for details."
        echo "  This may not prevent pipeline execution."
    fi
fi

# Print summary
echo ""
echo "=========================================="
echo "  Setup Complete!"
echo "=========================================="
echo ""
echo "Environment details:"
echo "  Python: $PYTHON_VERSION"
echo "  Virtual env: venv/"
echo "  Working dir: $(pwd)"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate the environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Verify installation:"
echo "   ced --help"
echo ""
echo "3. Check configuration files:"
echo "   ls configs/*.yaml"
echo ""
echo "4. Run production pipeline:"
echo "   Option A (Recommended): ./run_production.sh"
echo "   Option B (Manual):"
echo "     # Generate splits first:"
echo "     ced save-splits --config configs/splits_config.yaml \\"
echo "       --infile ../Celiac_dataset_proteomics.csv \\"
echo "       --n-splits 10"
echo "     # Then submit training:"
echo "     bsub < CeD_production.lsf"
echo ""
echo "Documentation:"
echo "  - Setup guide: docs/HPC_SETUP.md"
echo "  - Project overview: CLAUDE.md"
echo "  - Migration guide: docs/HPC_MIGRATION_GUIDE.md"
echo ""
echo "=========================================="
