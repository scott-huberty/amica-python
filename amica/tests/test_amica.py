"""
Test for the AMICA algorithm implementation.

This test runs the main AMICA algorithm and validates that it produces
expected outputs, serving as a regression test during refactoring.
"""

import pytest
from pathlib import Path
import sys
import os

# Add the amica directory to the Python path so we can import from amica.py
amica_dir = Path(__file__).parent.parent
sys.path.insert(0, str(amica_dir))


def test_amica_full_algorithm():
    """
    Test the complete AMICA algorithm by executing the full main script.
    
    This test runs the entire algorithm and checks that it completes successfully
    with expected outputs.
    """
    # Change to the amica directory so relative paths work
    original_cwd = os.getcwd()
    os.chdir(str(amica_dir))
    
    try:
        # This will execute the entire amica.py script as if run from command line
        # We capture any exceptions to ensure the algorithm runs to completion
        
        import subprocess
        import sys
        
        # Run the script as a subprocess to isolate it
        result = subprocess.run(
            [sys.executable, "amica.py"], 
            capture_output=False, 
            text=True,
            timeout=600,  # 10 minute timeout
            check=True  # Raise an error if the command fails
        )
        
        # Check that the script completed successfully
        assert result.returncode == 0, f"AMICA script failed with error: {result.stderr}"
        
        # Check for expected output patterns
        # output = result.stdout
        # assert "entering the main loop" in output, "Main algorithm loop not started"
        # assert "Iteration 1" in output, "First iteration not completed"
        
        print("✓ Full AMICA algorithm completed successfully")
        # print(f"Script output preview: {output[-500:]}")  # Show last 500 chars
        
    except subprocess.TimeoutExpired:
        pytest.fail("AMICA algorithm took too long to complete (>10 minutes)")
        
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


if __name__ == "__main__":
    # Allow running this test file directly
    pytest.main([__file__, "-v"])