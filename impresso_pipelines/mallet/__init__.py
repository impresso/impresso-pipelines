import importlib
import sys

REQUIRED_PACKAGES = [
    "jpype",
    "smart_open",
    "boto3",
    "dotenv",
    "spacy",
]

MISSING_PACKAGES = []

for package in REQUIRED_PACKAGES:
    if importlib.util.find_spec(package) is None:
        MISSING_PACKAGES.append(package)

if MISSING_PACKAGES:
    missing = ", ".join(MISSING_PACKAGES)
    sys.stderr.write(
        f"Error: The following required packages are missing for the Mallet module: {missing}\n"
        "Please install them with: pip install 'impresso-pipelines[mallet]'\n"
    )
    sys.exit(1)

