@echo off
setlocal
chcp 65001 > nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set PYTHONLEGACYWINDOWSSTDIO=0
set REPO_ROOT=%~dp0..
set PYTHON=%REPO_ROOT%\.venv-app\Scripts\python.exe
if not exist "%PYTHON%" set PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe
if not exist "%PYTHON%" goto launcher
"%PYTHON%" -m pytest %*
exit /b %ERRORLEVEL%

:launcher
py -3.13 -m pytest %*
exit /b %ERRORLEVEL%
