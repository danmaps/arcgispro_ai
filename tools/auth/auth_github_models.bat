@echo off
setlocal

set "TOKEN_DIR=%USERPROFILE%\.arcgispro_ai"
set "TOKEN_FILE=%TOKEN_DIR%\github_models_token.txt"

echo.
echo ArcGIS Pro AI Toolbox - GitHub Models Authentication
echo.
echo Opening the GitHub token page in your browser...
start "" "https://github.com/settings/personal-access-tokens/new"
echo.
echo Create a token with models access, then paste it below.
set /p GITHUB_TOKEN=GitHub token: 

if "%GITHUB_TOKEN%"=="" (
    echo.
    echo No token entered. Nothing was saved.
    exit /b 1
)

if not exist "%TOKEN_DIR%" (
    mkdir "%TOKEN_DIR%"
)

> "%TOKEN_FILE%" (
    set /p="%GITHUB_TOKEN%"
)

icacls "%TOKEN_DIR%" /inheritance:r >nul
icacls "%TOKEN_DIR%" /grant:r "%USERNAME%:(OI)(CI)F" >nul
icacls "%TOKEN_FILE%" /inheritance:r >nul
icacls "%TOKEN_FILE%" /grant:r "%USERNAME%:F" >nul

echo.
echo Saved token to:
echo %TOKEN_FILE%
echo.
echo Setup complete.
endlocal
