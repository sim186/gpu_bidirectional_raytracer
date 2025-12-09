# Tests

This directory contains tests for the GPU Bidirectional Ray Tracer project.

## Running Tests

To run all validation tests:

```bash
./tests/run_tests.sh
```

## Test Suite

### Current Tests

1. **Source File Verification**: Checks that all required source files exist
2. **Asset File Verification**: Verifies asset files are present
3. **Header Guard Verification**: Ensures header files have proper include guards
4. **Scene File Format Validation**: Validates scene file structure
5. **Code Quality Checks**: Basic code quality and include path verification
6. **Makefile Verification**: Checks that required build targets exist

### Test Results

All tests should pass in a properly configured repository. If any tests fail, check:

- All files are in the correct directories
- No files were accidentally deleted
- File permissions are correct

## Future Test Additions

Since this is a GPU-accelerated graphics application, comprehensive automated testing requires:

1. **GPU Environment**: Tests need access to CUDA-capable GPU
2. **Build Tests**: Verify compilation succeeds
3. **Render Tests**: Compare rendered output against reference images
4. **Performance Tests**: Benchmark rendering performance
5. **Unit Tests**: Test individual functions in isolation

### Adding New Tests

To add new tests, edit `run_tests.sh` and follow the existing pattern:

```bash
echo ""
echo "Test N: Description..."

# Your test logic here
if [ condition ]; then
    echo "  ✓ Test passed"
    ((PASS++))
else
    echo "  ✗ Test failed"
    ((FAIL++))
fi
```

## Notes

- These tests are validation tests that don't require GPU execution
- For full functionality testing, manual testing with a CUDA-capable GPU is required
- Tests are designed to be fast and suitable for CI/CD pipelines
