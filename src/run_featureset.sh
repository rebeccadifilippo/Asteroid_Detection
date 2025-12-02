#!/bin/bash

# Stop the script if any command fails
set -e

echo "Running Physical Features..."
python main.py -fs physical

echo "Running Orbital Core Features..."
python main.py -fs orbital_core

echo "Running Orbital Derived Features..."
python mai