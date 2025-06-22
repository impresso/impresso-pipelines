import pytest
import jpype
import glob
import os

@pytest.fixture(scope="session", autouse=True)
def start_jvm_once_for_all_tests():
    """
    Start the JVM once for all tests with all JARs in lucene_jars.
    This ensures all Java-dependent tests have the required classes available.
    """
    jar_dir = "lucene_jars"
    jar_paths = glob.glob(os.path.join(jar_dir, "*.jar"))
    if not jpype.isJVMStarted():
        if not jar_paths:
            pytest.skip("No Lucene JARs found in lucene_jars/", allow_module_level=True)
        jpype.startJVM(classpath=jar_paths)
