#!/usr/bin/env bash
#
# Local test runner script for impresso-pipelines
# This script helps developers run tests in an environment similar to CI
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== impresso-pipelines Local Test Runner ===${NC}\n"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo -e "${RED}Error: Poetry is not installed${NC}"
    echo "Install it with: curl -sSL https://install.python-poetry.org | python3 -"
    exit 1
fi

# Check if package is installed in editable mode or via poetry
if ! python -c "import impresso_pipelines" &> /dev/null; then
    echo -e "${YELLOW}Package not installed. Installing in editable mode...${NC}"
    pip install -e ".[all]"
elif [[ $(python -c "import impresso_pipelines; print(impresso_pipelines.__file__)") == *"site-packages"* ]]; then
    echo -e "${YELLOW}Warning: Package appears to be installed from site-packages, not in editable mode.${NC}"
    echo -e "${YELLOW}For development, it's recommended to use: pip install -e '.[all]'${NC}"
    echo ""
fi

# Parse arguments
TEST_TYPE="${1:-quick}"

case "$TEST_TYPE" in
    quick)
        echo -e "${GREEN}Running quick tests (skipping JVM)...${NC}"
        IMPRESSO_SKIP_JVM=1 poetry run pytest
        ;;
    
    full)
        echo -e "${GREEN}Running full test suite...${NC}"
        
        # Check for Lucene setup
        if [ ! -d "lucene_jars" ]; then
            echo -e "${YELLOW}Lucene JARs not found. Setting up...${NC}"
            make setup-lucene
        fi
        
        # Source Lucene environment
        if [ -f "lucene_env.sh" ]; then
            source lucene_env.sh
        fi
        
        poetry run pytest
        ;;
    
    ocrqa)
        echo -e "${GREEN}Running OCRQA tests only...${NC}"
        IMPRESSO_SKIP_JVM=1 poetry run pytest tests/ocrqa/
        ;;
    
    langident)
        echo -e "${GREEN}Running language identification tests only...${NC}"
        IMPRESSO_SKIP_JVM=1 poetry run pytest tests/langident/
        ;;
    
    coverage)
        echo -e "${GREEN}Running tests with coverage report...${NC}"
        IMPRESSO_SKIP_JVM=1 poetry run pytest --cov=impresso_pipelines --cov-report=html --cov-report=term
        echo -e "\n${GREEN}Coverage report generated in htmlcov/index.html${NC}"
        ;;
    
    qa)
        echo -e "${GREEN}Running full QA suite (tests + linting + type checking)...${NC}"
        
        echo -e "\n${YELLOW}1. Running tests...${NC}"
        IMPRESSO_SKIP_JVM=1 poetry run pytest
        
        echo -e "\n${YELLOW}2. Running linting...${NC}"
        poetry run pylint impresso_pipelines/ || true
        
        echo -e "\n${YELLOW}3. Running type checking...${NC}"
        poetry run mypy impresso_pipelines/ || true
        
        echo -e "\n${GREEN}QA suite complete!${NC}"
        ;;
    
    help|--help|-h)
        echo "Usage: ./run_tests.sh [TEST_TYPE]"
        echo ""
        echo "Available test types:"
        echo "  quick      - Run tests skipping JVM-dependent tests (default)"
        echo "  full       - Run all tests including JVM-dependent tests"
        echo "  ocrqa      - Run only OCRQA tests"
        echo "  langident  - Run only language identification tests"
        echo "  coverage   - Run tests with coverage report"
        echo "  qa         - Run full QA suite (tests + linting + type checking)"
        echo "  help       - Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run_tests.sh              # Quick tests"
        echo "  ./run_tests.sh full         # All tests"
        echo "  ./run_tests.sh coverage     # With coverage"
        exit 0
        ;;
    
    *)
        echo -e "${RED}Unknown test type: $TEST_TYPE${NC}"
        echo "Run './run_tests.sh help' for usage information"
        exit 1
        ;;
esac

echo -e "\n${GREEN}✓ Tests complete!${NC}"
