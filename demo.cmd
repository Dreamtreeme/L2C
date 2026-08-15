@echo off
setlocal
set "REPO_ROOT=%~dp0"
set "PYTHON=%REPO_ROOT%.venv-app\Scripts\python.exe"

if not exist "%PYTHON%" goto missing

"%PYTHON%" "%REPO_ROOT%scripts\seed_demo_db.py"
if errorlevel 1 exit /b %ERRORLEVEL%

set "DB_PATH=%REPO_ROOT%data\demo_jobs.db"
call "%REPO_ROOT%run.cmd" %*
exit /b %ERRORLEVEL%

:missing
echo 앱 Python 환경을 찾을 수 없습니다. setup.cmd를 먼저 실행하세요.
exit /b 1
