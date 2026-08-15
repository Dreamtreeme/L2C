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
if not exist "%PYTHON%" goto missing
"%PYTHON%" -m pytest %TEST_ARGS%
exit /b %ERRORLEVEL%

:missing
echo 앱 Python 환경을 찾을 수 없습니다. setup.cmd를 먼저 실행하세요.
exit /b 1
