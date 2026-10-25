@echo off
rem ============================================================================
rem   CarDamage AI - Autonomous Vehicle Inspection and Repair Cost Estimator
rem   Unified Application Launcher (Windows)
rem ============================================================================

setlocal EnableExtensions EnableDelayedExpansion
title CarDamage AI - Autonomous Vehicle Inspection and Cost Estimator
cd /d "%~dp0"

set "HOST=127.0.0.1"
set "PORT=8000"
set "RELOAD=--reload"
set "PYTHON_EXE="

rem Parse CLI arguments
set "ACTION=server"
:parse_args
if "%~1"=="" goto after_args
if /i "%~1"=="server"      set "ACTION=server"& shift & goto parse_args
if /i "%~1"=="start"       set "ACTION=server"& shift & goto parse_args
if /i "%~1"=="--server"    set "ACTION=server"& shift & goto parse_args
if /i "%~1"=="-s"          set "ACTION=server"& shift & goto parse_args
if /i "%~1"=="menu"        set "ACTION=menu"& shift & goto parse_args
if /i "%~1"=="--menu"      set "ACTION=menu"& shift & goto parse_args
if /i "%~1"=="-m"          set "ACTION=menu"& shift & goto parse_args
if /i "%~1"=="check"       set "ACTION=check"& shift & goto parse_args
if /i "%~1"=="--check"     set "ACTION=check"& shift & goto parse_args
if /i "%~1"=="-c"          set "ACTION=check"& shift & goto parse_args
if /i "%~1"=="diag"        set "ACTION=check"& shift & goto parse_args
if /i "%~1"=="test"        set "ACTION=test"& shift & goto parse_args
if /i "%~1"=="--test"      set "ACTION=test"& shift & goto parse_args
if /i "%~1"=="-t"          set "ACTION=test"& shift & goto parse_args
if /i "%~1"=="install"     set "ACTION=install"& shift & goto parse_args
if /i "%~1"=="--install"   set "ACTION=install"& shift & goto parse_args
if /i "%~1"=="-i"          set "ACTION=install"& shift & goto parse_args
if /i "%~1"=="clean"       set "ACTION=clean"& shift & goto parse_args
if /i "%~1"=="--clean"     set "ACTION=clean"& shift & goto parse_args
if /i "%~1"=="help"        set "ACTION=help"& shift & goto parse_args
if /i "%~1"=="--help"      set "ACTION=help"& shift & goto parse_args
if /i "%~1"=="-h"          set "ACTION=help"& shift & goto parse_args
if /i "%~1"=="--port"      set "PORT=%~2"& shift & shift & goto parse_args
if /i "%~1"=="-p"          set "PORT=%~2"& shift & shift & goto parse_args
if /i "%~1"=="--no-reload" set "RELOAD="& shift & goto parse_args
shift
goto parse_args
:after_args

rem Banner Display
call :print_banner

rem Python Detection
call :detect_python
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python environment initialization failed.
    echo.
    pause
    exit /b 1
)

rem Route Actions
if /i "%ACTION%"=="menu"    goto show_menu
if /i "%ACTION%"=="help"    goto show_help
if /i "%ACTION%"=="check"   goto run_check
if /i "%ACTION%"=="test"    goto run_test
if /i "%ACTION%"=="install" goto run_install
if /i "%ACTION%"=="clean"   goto run_clean
if /i "%ACTION%"=="server"  goto run_server

goto run_server

rem ============================================================================
rem   BANNER
rem ============================================================================
:print_banner
cls
echo ============================================================================
echo   [ CAR-DAMAGE AI ]
echo   Autonomous Vehicle Damage Inspection and Repair Cost Estimator
echo   B.Tech CSE Capstone Project -- 5-Stage Vision-Actuarial Pipeline
echo ============================================================================
echo.
exit /b 0

rem ============================================================================
rem   HELP
rem ============================================================================
:show_help
echo USAGE:
echo   run.bat [COMMAND] [OPTIONS]
echo.
echo COMMANDS:
echo   start, server    Start FastAPI backend and Web UI [Default]
echo   menu             Interactive numbered action menu
echo   check, diag      Run system, environment and model weights diagnostics
echo   test             Run PyTest automated test suite
echo   install          Install or update requirements [pip install]
echo   clean            Remove __pycache__ and pytest cache directories
echo   help             Show this help screen
echo.
echo OPTIONS:
echo   --port PORT      Specify HTTP port [Default: 8000]
echo   --no-reload      Run uvicorn without live auto-reload
echo.
echo EXAMPLES:
echo   run.bat
echo   run.bat --port 8080
echo   run.bat check
echo   run.bat test
echo.
pause
exit /b 0

rem ============================================================================
rem   INTERACTIVE MENU
rem ============================================================================
:show_menu
echo Select an action to execute:
echo.
echo   [1] Start Web Server [FastAPI + Web UI on port %PORT%]
echo   [2] Run Model Weights and Environment Health Diagnostics
echo   [3] Run Automated Test Suite [PyTest]
echo   [4] Install / Update Dependencies [requirements.txt]
echo   [5] Clean Bytecode Caches [__pycache__]
echo   [6] Exit
echo.
set /p "MENU_CHOICE=Enter choice 1-6 (Default 1): "
if "%MENU_CHOICE%"=="" set "MENU_CHOICE=1"
if "%MENU_CHOICE%"=="1" goto run_server
if "%MENU_CHOICE%"=="2" goto run_check
if "%MENU_CHOICE%"=="3" goto run_test
if "%MENU_CHOICE%"=="4" goto run_install
if "%MENU_CHOICE%"=="5" goto run_clean
if "%MENU_CHOICE%"=="6" exit /b 0
echo [WARN] Invalid choice '%MENU_CHOICE%'. Defaulting to start server.
goto run_server

rem ============================================================================
rem   PYTHON AND VIRTUAL ENVIRONMENT DETECTION
rem ============================================================================
:detect_python
echo [*] Initializing Python runtime environment...

if defined VIRTUAL_ENV (
    echo [OK] Using active virtual environment: %VIRTUAL_ENV%
    set "PYTHON_EXE=python"
    goto check_python_ver
)

if exist "%~dp0.venv\Scripts\activate.bat" (
    echo [OK] Activating project virtual environment: .venv
    call "%~dp0.venv\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto check_python_ver
)
if exist "%~dp0venv\Scripts\activate.bat" (
    echo [OK] Activating project virtual environment: venv
    call "%~dp0venv\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto check_python_ver
)
if exist "%~dp0env\Scripts\activate.bat" (
    echo [OK] Activating project virtual environment: env
    call "%~dp0env\Scripts\activate.bat"
    set "PYTHON_EXE=python"
    goto check_python_ver
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=python"
    goto check_python_ver
)

where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "PYTHON_EXE=py -3"
    goto check_python_ver
)

echo [ERROR] No Python installation found on system PATH!
echo         Please install Python 3.10+ from https://www.python.org/
exit /b 1

:check_python_ver
%PYTHON_EXE% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python 3.10 or higher is required.
    %PYTHON_EXE% --version 2>nul
    exit /b 1
)

for /f "delims=" %%v in ('%PYTHON_EXE% --version 2^>^&1') do set "PY_VER=%%v"
echo [OK] Python runtime: %PY_VER%
echo.
exit /b 0

rem ============================================================================
rem   DIAGNOSTICS AND HEALTH CHECK
rem ============================================================================
:run_check
echo ============================================================================
echo   Running System and Model Integrity Diagnostics
echo ============================================================================
echo.
%PYTHON_EXE% -c "import sys, platform; print(f'  OS Platform:     {platform.system()} {platform.release()} ({platform.machine()})'); print(f'  Python Path:     {sys.executable}'); print(f'  Python Version:  {platform.python_version()}')"
echo.

echo [Packages]
%PYTHON_EXE% -c "import torch; print(f'  [+] PyTorch:       {torch.__version__} (CUDA Available: {torch.cuda.is_available()})')" 2>nul || echo   [-] PyTorch:       MISSING
%PYTHON_EXE% -c "import ultralytics; print(f'  [+] Ultralytics:   {ultralytics.__version__}')" 2>nul || echo   [-] Ultralytics:   MISSING
%PYTHON_EXE% -c "import fastapi; print(f'  [+] FastAPI:       {fastapi.__version__}')" 2>nul || echo   [-] FastAPI:       MISSING
%PYTHON_EXE% -c "import uvicorn; print(f'  [+] Uvicorn:       {uvicorn.__version__}')" 2>nul || echo   [-] Uvicorn:       MISSING
%PYTHON_EXE% -c "import sklearn; print(f'  [+] Scikit-Learn:  {sklearn.__version__}')" 2>nul || echo   [-] Scikit-Learn:  MISSING
%PYTHON_EXE% -c "import cv2; print(f'  [+] OpenCV:        {cv2.__version__}')" 2>nul || echo   [-] OpenCV:        MISSING
%PYTHON_EXE% -c "import xgboost; print(f'  [+] XGBoost:       {xgboost.__version__}')" 2>nul || echo   [-] XGBoost:       OPTIONAL / NOT INSTALLED
echo.

echo [Model Weights and Artifacts]
if exist "%~dp0yolov8n.pt" (
    echo   [+] YOLOv8 Base:          yolov8n.pt [EXISTS]
) else (
    echo   [WARN] YOLOv8 Base:       yolov8n.pt [NOT FOUND]
)
if exist "%~dp0models\detection\phase4_yolov8n_baseline\weights\best.pt" (
    echo   [+] YOLOv8 Detector:      phase4_yolov8n_baseline\weights\best.pt [EXISTS]
) else (
    echo   [WARN] YOLOv8 Detector:   best.pt [NOT FOUND in models\detection]
)
if exist "%~dp0models\classification\phase5_damage_classifier\best_classifier.pt" (
    echo   [+] Damage Classifier:    phase5_damage_classifier\best_classifier.pt [EXISTS]
) else (
    echo   [WARN] Damage Classifier: best_classifier.pt [NOT FOUND]
)
if exist "%~dp0models\severity\phase7_severity_model\best_severity_model.pt" (
    echo   [+] Severity Assessor:    phase7_severity_model\best_severity_model.pt [EXISTS]
) else (
    echo   [WARN] Severity Assessor: best_severity_model.pt [NOT FOUND]
)
if exist "%~dp0models\cost\best_cost_model.joblib" (
    echo   [+] Cost Regressor:       models\cost\best_cost_model.joblib [EXISTS]
) else (
    echo   [WARN] Cost Regressor:    best_cost_model.joblib [NOT FOUND]
)
if exist "%~dp0models\cost\quantile_models.joblib" (
    echo   [+] Quantile Bounds:      models\cost\quantile_models.joblib [EXISTS]
) else (
    echo   [WARN] Quantile Bounds:   quantile_models.joblib [NOT FOUND]
)
echo.

echo [Port Status]
set "PORT_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set "PORT_PID=%%a"
if defined PORT_PID (
    echo   [WARN] Port %PORT% occupied by PID: !PORT_PID!
) else (
    echo   [OK] Port %PORT% is FREE and ready for binding.
)
echo.
echo ============================================================================
echo   Diagnostics completed.
echo ============================================================================
pause
exit /b 0

rem ============================================================================
rem   TEST SUITE
rem ============================================================================
:run_test
echo ============================================================================
echo   Executing Automated PyTest Suite
echo ============================================================================
echo.
%PYTHON_EXE% -m pytest tests/ -v
if %ERRORLEVEL% equ 0 (
    echo.
    echo [OK] All automated tests passed successfully!
) else (
    echo.
    echo [FAIL] One or more tests failed. Check log output above.
)
echo.
pause
exit /b %ERRORLEVEL%

rem ============================================================================
rem   INSTALL DEPENDENCIES
rem ============================================================================
:run_install
echo ============================================================================
echo   Installing Dependencies from requirements.txt
echo ============================================================================
echo.
%PYTHON_EXE% -m pip install --upgrade pip
%PYTHON_EXE% -m pip install -r "%~dp0requirements.txt"
if %ERRORLEVEL% equ 0 (
    echo.
    echo [OK] Dependencies installed successfully!
) else (
    echo.
    echo [ERROR] Package installation encountered errors.
)
echo.
pause
exit /b %ERRORLEVEL%

rem ============================================================================
rem   CLEAN BYTECODE
rem ============================================================================
:run_clean
echo [*] Cleaning Python cache files and test artifacts...
for /d /r "%~dp0" %%d in (__pycache__) do (
    if exist "%%d" rd /s /q "%%d" 2>nul
)
for /d /r "%~dp0" %%d in (.pytest_cache) do (
    if exist "%%d" rd /s /q "%%d" 2>nul
)
del /s /q "%~dp0*.pyc" >nul 2>nul
del /s /q "%~dp0*.pyo" >nul 2>nul
echo [OK] Cleaned cache files.
pause
exit /b 0

rem ============================================================================
rem   START APPLICATION SERVER
rem ============================================================================
:run_server
echo ============================================================================
echo   Starting CarDamage AI Server
echo   Target URL: http://%HOST%:%PORT%/
echo ============================================================================
echo.

rem Pre-flight: Check if port is already listening
set "PORT_PID="
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do set "PORT_PID=%%a"

if not defined PORT_PID goto launch_server

echo [INFO] Port %PORT% is currently occupied by PID: !PORT_PID!
%PYTHON_EXE% -c "import urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:%PORT%/api/health', timeout=1).getcode() == 200 else 1)" >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [OK] CarDamage AI is already running at http://%HOST%:%PORT%/
    echo [*] Opening browser...
    start http://%HOST%:%PORT%/
    exit /b 0
) else (
    echo [WARN] Stale process [PID: !PORT_PID!] detected on port %PORT%.
    echo Terminating stale process...
    taskkill /F /PID !PORT_PID! >nul 2>&1
    ping 127.0.0.1 -n 2 >nul
    goto launch_server
)

:launch_server
rem Ensure required packages are present
%PYTHON_EXE% -c "import fastapi, uvicorn, multipart" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo [WARN] Installing missing web dependencies...
    %PYTHON_EXE% -m pip install fastapi uvicorn python-multipart
)

echo [*] Starting browser launcher thread...
start "" /b %PYTHON_EXE% -c "import time, urllib.request, webbrowser; time.sleep(1.2); [time.sleep(0.5) for _ in range(40) if not urllib.request.urlopen('http://127.0.0.1:%PORT%/api/health', timeout=2).getcode() == 200]; webbrowser.open('http://127.0.0.1:%PORT%/')"

echo [*] Starting Uvicorn ASGI server...
echo     Local URL: http://%HOST%:%PORT%/
echo     Press Ctrl+C anytime to stop.
echo.

%PYTHON_EXE% -m uvicorn app.backend.main:app --host %HOST% --port %PORT% %RELOAD%

if %ERRORLEVEL% neq 0 (
    echo.
    echo ============================================================================
    echo [ERROR] Server exited with code %ERRORLEVEL%.
    echo         To check dependencies, run: run.bat install
    echo         To check system status, run: run.bat check
    echo ============================================================================
    echo.
    pause
)
exit /b %ERRORLEVEL%
