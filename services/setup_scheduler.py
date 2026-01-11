@echo off
SET PROJECT_DIR=%~dp0

echo === Checking Python ===
python --version >nul 2>&1
IF ERRORLEVEL 1 (
    echo Python not found. Please install Python 3.10+
    pause
    exit /b
)

echo === Creating virtual environment ===
IF NOT EXIST "%PROJECT_DIR%venv" (
    python -m venv "%PROJECT_DIR%venv"
)

echo === Activating venv ===
call "%PROJECT_DIR%venv\Scripts\activate.bat"

echo === Installing dependencies ===
pip install --upgrade pip
pip install -r "%PROJECT_DIR%requirements.txt"

echo === Creating Scheduler Task ===
schtasks /create ^
 /tn "VectorDB_Monthly_Update" ^
 /tr "\"%PROJECT_DIR%venv\Scripts\python.exe\" \"%PROJECT_DIR%vectordb.py\"" ^
 /sc monthly ^
 /mo second ^
 /d sat ^
 /st 22:00 ^
 /f

echo === DONE ===
echo Scheduler set for Second Saturday Night (10 PM)
pause
