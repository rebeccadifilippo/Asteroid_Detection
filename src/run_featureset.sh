#!/bin/bash

# Stop the script if any command fails
set -e

echo "Running Physical Features..."
python main.py -fs physical

echo "Running Orbital Core Features..."
python main.py -fs orbital_core

echo "Running Orbital Derived Features..."
python main.py -fs orbital_derived

echo "Running Motion & Timing Features..."
python main.py -fs motion_timing

echo "Running Uncertainties Features..."
python main.py -fs uncertainties

echo "Running Model Fit Features..."
python main.py -fs model_fit

echo "Running All Features..."
python main.py -fs all

echo "Running Good Features..."
python main.py -fs orbital_derived,physical,orbital_core