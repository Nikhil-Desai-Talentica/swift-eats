# Coverage Report

This document describes how to generate and view coverage reports for all three microservices.

## Generating Coverage Reports

### Option 1: Using the Python Script (Recommended)

```bash
python3 generate_coverage.py
```

This script will:
1. Run pytest with coverage for each service (order-service, menu-service, logistics-service)
2. Generate HTML coverage reports
3. Copy all HTML reports to the `coverage/` directory
4. Display a summary of coverage results

### Option 2: Using the Shell Script

```bash
./generate_coverage_report.sh
```

### Option 3: Manual Generation

Run coverage for each service individually:

```bash
# Order Service
cd order-service
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v

# Menu Service
cd menu-service
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v

# Logistics Service
cd logistics-service
pytest --cov=app --cov-report=html --cov-report=term-missing --cov-report=json -v
```

## Viewing Coverage Reports

After generating coverage reports, HTML reports are available at:

- **Order Service**: `coverage/order-service-htmlcov/index.html`
- **Menu Service**: `coverage/menu-service-htmlcov/index.html`
- **Logistics Service**: `coverage/logistics-service-htmlcov/index.html`

Open these HTML files in your browser to view detailed coverage information, including:
- Overall coverage percentage
- Line-by-line coverage highlighting
- Missing lines and branches
- Coverage by module/file

## Coverage Configuration

Each service uses `pytest-cov` with the following configuration:

- **Coverage source**: `app/` directory (all application code)
- **Report formats**: HTML, terminal (with missing lines), and JSON
- **Coverage tool**: `coverage.py` (via pytest-cov)

## Coverage Goals

While there are no strict coverage requirements, the following are recommended targets:

- **Critical paths**: 90%+ coverage (API endpoints, business logic, error handling)
- **Service layer**: 85%+ coverage
- **Models**: 80%+ coverage
- **Workers**: 80%+ coverage (with mocked external dependencies)

## Interpreting Coverage Reports

### Terminal Output

When running pytest with coverage, you'll see output like:

```
---------- coverage: platform linux, python 3.11.0 -----------
Name                          Stmts   Miss  Cover   Missing
------------------------------------------------------------
app/__init__.py                   0      0   100%
app/api/routes/orders.py         45      5    89%   23-27, 45
app/services/order_service.py    78     12    85%   45-50, 78-82
------------------------------------------------------------
TOTAL                           123     17    86%
```

### HTML Reports

The HTML reports provide:
- **Summary page**: Overall coverage statistics
- **File-by-file breakdown**: Click on any file to see line-by-line coverage
- **Missing lines**: Highlighted in red
- **Covered lines**: Highlighted in green
- **Branch coverage**: Shows which code branches were tested

## Excluding Code from Coverage

To exclude certain code from coverage (e.g., type stubs, test utilities), you can add a `.coveragerc` file in each service directory:

```ini
[run]
omit =
    */tests/*
    */test_*
    */__pycache__/*
    */venv/*
    */env/*
```

## Continuous Integration

For CI/CD pipelines, you can generate coverage reports and fail the build if coverage drops below a threshold:

```bash
pytest --cov=app --cov-report=xml --cov-fail-under=80
```

This will:
- Generate an XML report (compatible with CI tools)
- Fail if coverage is below 80%

## Notes

- Coverage reports are generated in each service's `htmlcov/` directory
- Combined reports are copied to the root `coverage/` directory for easy access
- Coverage data files (`.coverage`) are stored in each service directory
- HTML reports can be served via a simple HTTP server if needed: `python3 -m http.server 8000`

