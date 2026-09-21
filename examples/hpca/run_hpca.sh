#!/bin/bash
set -e

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Create build directory if it doesn't exist
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir build
fi

# Go into build directory
cd build

# Run CMake
echo "Running CMake..."
cmake ..

# Run Make
if [ -z "$1" ]; then
    echo "Building all targets..."
    make -j$(nproc)
else
    # Map numbers to targets
    case "$1" in
        0) TARGET="00_basic_workflow" ;;
        1) TARGET="01_bootstrapping" ;;
        2) TARGET="02_polynomials" ;;
        3) TARGET="03_simd" ;;
        4) TARGET="04_optimizations" ;;
        5) TARGET="05_gaussian" ;;
        *) TARGET=$1 ;;
    esac

    echo "Building target: $TARGET"
    make -j$(nproc) $TARGET
    
    # Check if executable exists and run it
    if [ -f "./$TARGET" ]; then
        echo "--------------------------------------------------"
        echo "Running $TARGET..."
        echo "--------------------------------------------------"
        
        # Specific execution logic
        if [ "$TARGET" = "05_gaussian" ]; then
            ./$TARGET ../images/input.png ../output.png
        elif [ "$TARGET" = "02_polynomials" ]; then
            ./$TARGET
            echo "--------------------------------------------------"
            echo "Running python validation script..."
            echo "--------------------------------------------------"
            
            # Move CSV to root
            if [ -f "polynomial_results.csv" ]; then
                mv polynomial_results.csv ../
            fi
            
            # Run python script with root CSV
            python3 ../src/02_polynomials.py ../polynomial_results.csv
            
            # Move plot if generated
            if [ -f "polynomial_plot.png" ]; then
                mv polynomial_plot.png ../
            fi
        else
            ./$TARGET 2> /dev/null
        fi
    else
        echo "Error: Executable '$TARGET' not found."
        exit 1
    fi
fi
