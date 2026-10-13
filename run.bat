@echo off
rem Quick launcher for CarDamage AI
setlocal
cd /d "%~dp0"
call "%~dp0run_car_damage_ai.bat" %*
exit /b %ERRORLEVEL%
