@echo off
REM Build QUASAR-SUBNET Docker images (Challenge + Validator only)

echo ========================================
echo QUASAR-SUBNET Docker Build Script
echo ========================================
echo.
echo NOTE: Miners run locally, not in Docker
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker is not installed or not running.
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    exit /b 1
)

echo Docker is installed!
echo.

REM Build Challenge Container
echo [1/2] Building Challenge Container...
docker build -t quasar-subnet/challenge:latest -f challenge/Dockerfile .
if errorlevel 1 (
    echo ERROR: Failed to build Challenge Container
    exit /b 1
)
echo SUCCESS: Challenge Container built
echo.

REM Build Validator Node
echo [2/2] Building Validator Node...
docker build -t quasar-subnet/validator:latest -f validator/Dockerfile .
if errorlevel 1 (
    echo ERROR: Failed to build Validator Node
    exit /b 1
)
echo SUCCESS: Validator Node built
echo.

echo ========================================
echo All images built successfully!
echo ========================================
echo.
echo Images:
echo   - quasar-subnet/challenge:latest
echo   - quasar-subnet/validator:latest
echo.
echo To run Docker services:
echo   docker-compose up
echo.
echo To run miner locally:
echo   python miner/server.py --port 8000
echo.
echo To test endpoints:
echo   curl http://localhost:8080/health      # Challenge container
echo   curl http://localhost:8000/health      # Local Miner
echo.
