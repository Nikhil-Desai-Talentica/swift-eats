#!/bin/bash

# Generate coverage reports for all services
set -e

echo "=========================================="
echo "Generating Coverage Reports"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create coverage directory
mkdir -p coverage

# Order Service
echo -e "\n${BLUE}Running coverage for order-service...${NC}"
cd order-service
if [ -d "htmlcov" ]; then
    rm -rf htmlcov
fi
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v || true
if [ -d "htmlcov" ]; then
    cp -r htmlcov ../coverage/order-service-htmlcov
    cp .coverage.json ../coverage/order-service-coverage.json 2>/dev/null || true
fi
cd ..

# Menu Service
echo -e "\n${BLUE}Running coverage for menu-service...${NC}"
cd menu-service
if [ -d "htmlcov" ]; then
    rm -rf htmlcov
fi
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v || true
if [ -d "htmlcov" ]; then
    cp -r htmlcov ../coverage/menu-service-htmlcov
    cp .coverage.json ../coverage/menu-service-coverage.json 2>/dev/null || true
fi
cd ..

# Logistics Service
echo -e "\n${BLUE}Running coverage for logistics-service...${NC}"
cd logistics-service
if [ -d "htmlcov" ]; then
    rm -rf htmlcov
fi
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v || true
if [ -d "htmlcov" ]; then
    cp -r htmlcov ../coverage/logistics-service-htmlcov
    cp .coverage.json ../coverage/logistics-service-coverage.json 2>/dev/null || true
fi
cd ..

echo -e "\n${GREEN}Coverage reports generated!${NC}"
echo -e "\nHTML reports available at:"
echo "  - order-service: coverage/order-service-htmlcov/index.html"
echo "  - menu-service: coverage/menu-service-htmlcov/index.html"
echo "  - logistics-service: coverage/logistics-service-htmlcov/index.html"

