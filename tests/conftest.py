import pytest
import os
import subprocess

#@pytest.fixture(scope='session', autouse=True)
def setup_r_environment():
    # Set the path to the R installation directory
    r_bin_path = r'C:\Program Files\R\R-4.2.1\bin\x64'

    # Add the R bin directory to the PATH
    os.environ['PATH'] = f"{r_bin_path};{os.environ['PATH']}"

    # Now you can run your R-related commands
    subprocess.run(["R", "--version"], check=True)
