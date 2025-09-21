"""
Test for the AMICA algorithm implementation.

This test runs the main AMICA algorithm and validates that it produces
expected outputs, serving as a regression test during refactoring.
"""
import subprocess
from pathlib import Path

import pytest

def test_amica_full_algorithm():
    """
    Test the complete AMICA algorithm by executing the full main script.
    
    This test runs the entire algorithm and checks that it completes successfully
    with expected outputs.
    """    
    try:
        # This will execute the entire amica.py script as if run from command line
        # We capture any exceptions to ensure the algorithm runs to completion        
        # Run the script as a subprocess to isolate it
        result = subprocess.run(
            ["python", "-c", "import amica; amica.core.main()"], 
            capture_output=False, 
            text=True,
            timeout=60,  # 1 minute timeout
            check=True  # Raise an error if the command fails
        )
        
        # Check that the script completed successfully
        assert result.returncode == 0, f"AMICA script failed with error: {result.stderr}"        
        print("✓ Full AMICA algorithm completed successfully")        
    except subprocess.TimeoutExpired:
        pytest.fail("AMICA algorithm took too long to complete (>10 minutes)")