#!/bin/bash
# Basic validation tests for GPU Bidirectional Ray Tracer
# These tests verify basic functionality without requiring GPU execution

echo "====================================="
echo "GPU Bidirectional Ray Tracer Tests"
echo "====================================="

# Test 1: Check if source files exist
echo ""
echo "Test 1: Verifying source files..."
PASS=0
FAIL=0

check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $1 exists"
        ((PASS++))
    else
        echo "  ✗ $1 missing"
        ((FAIL++))
    fi
}

check_file "src/device.cu"
check_file "src/smallptCPU.c"
check_file "src/displayfunc.c"
check_file "src/MersenneTwister_kernel.cu"
check_file "include/vec.h"
check_file "include/geom.h"
check_file "include/camera.h"
check_file "include/scene.h"
check_file "Makefile"

# Test 2: Check if asset files exist
echo ""
echo "Test 2: Verifying asset files..."

check_file "assets/data/MersenneTwister.dat"
check_file "assets/scenes/cornell.scn"
check_file "assets/scenes/simple.scn"

# Test 3: Verify header guards
echo ""
echo "Test 3: Verifying header guards..."

check_header_guard() {
    FILE=$1
    GUARD=$2
    if grep -q "#ifndef $GUARD" "$FILE" && grep -E "#define[[:space:]]+$GUARD" "$FILE" > /dev/null && grep -q "#endif" "$FILE"; then
        echo "  ✓ $FILE has proper header guards"
        ((PASS++))
    else
        echo "  ✗ $FILE missing or improper header guards"
        ((FAIL++))
    fi
}

check_header_guard "include/vec.h" "_VEC_H"
check_header_guard "include/geom.h" "_GEOM_H"
check_header_guard "include/camera.h" "_CAMERA_H"

# Test 4: Check scene file format
echo ""
echo "Test 4: Validating scene file format..."

check_scene_file() {
    FILE=$1
    if [ -f "$FILE" ]; then
        if grep -q "^camera " "$FILE" && grep -q "^size " "$FILE" && grep -q "^sphere " "$FILE"; then
            echo "  ✓ $FILE has valid format"
            ((PASS++))
        else
            echo "  ✗ $FILE has invalid format"
            ((FAIL++))
        fi
    else
        echo "  ✗ $FILE not found"
        ((FAIL++))
    fi
}

check_scene_file "assets/scenes/cornell.scn"
check_scene_file "assets/scenes/simple.scn"

# Test 5: Check for common code issues
echo ""
echo "Test 5: Checking for common code issues..."

# Check for proper include paths
if grep -r "#include \"" src/ include/ | grep -v "include/" | grep -v "\.h\"" > /dev/null; then
    echo "  ⚠ Warning: Some includes might need path adjustment"
else
    echo "  ✓ Include paths look correct"
    ((PASS++))
fi

# Test 6: Verify Makefile targets
echo ""
echo "Test 6: Verifying Makefile..."

if grep -q "^smallptCPU:" Makefile; then
    echo "  ✓ Makefile has smallptCPU target"
    ((PASS++))
else
    echo "  ✗ Makefile missing smallptCPU target"
    ((FAIL++))
fi

if grep -q "^clean:" Makefile; then
    echo "  ✓ Makefile has clean target"
    ((PASS++))
else
    echo "  ✗ Makefile missing clean target"
    ((FAIL++))
fi

# Summary
echo ""
echo "====================================="
echo "Test Summary"
echo "====================================="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed."
    exit 1
fi
