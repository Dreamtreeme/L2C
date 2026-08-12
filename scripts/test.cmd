@echo off
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
set REPO_ROOT=%~dp0..
set PYTHON=%REPO_ROOT%\.venv-app\Scripts\python.exe
set TEST_ARGS=%*
if "%~1"=="" set TEST_ARGS=agent\tests -q -p no:cacheprovider
if not exist "%PYTHON%" goto launcher
"%PYTHON%" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 13) else 1)" > nul 2>&1
if errorlevel 1 goto launcher
"%PYTHON%" -m pytest %TEST_ARGS%
exit /b %ERRORLEVEL%

:launcher
py -3.13 -c "import sys" > nul 2>&1
if errorlevel 1 goto missing
py -3.13 -m pytest %TEST_ARGS%
exit /b %ERRORLEVEL%

:missing
echo Python 3.13 실행 환경을 찾을 수 없습니다. setup.cmd -Development를 먼저 실행하세요.
exit /b 1
