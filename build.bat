@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

rem --- locate Visual Studio's x64 build environment --------------------------
set "VCVARS="
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
rem  note: !VSWHERE! (delayed) not %VSWHERE% -- the path contains "(x86)" and a
rem  literal ")" inside a parenthesised block would terminate it early.
if exist "!VSWHERE!" (
    "!VSWHERE!" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath > "%TEMP%\_vspath.txt" 2>nul
    set /p VSPATH=<"%TEMP%\_vspath.txt"
    del /q "%TEMP%\_vspath.txt" 2>nul
    if exist "!VSPATH!\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=!VSPATH!\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
    for %%v in (2022 2019) do (
        for %%e in (Enterprise Professional Community BuildTools) do (
            if exist "%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat" (
                set "VCVARS=%ProgramFiles%\Microsoft Visual Studio\%%v\%%e\VC\Auxiliary\Build\vcvars64.bat"
            )
        )
    )
)
if not defined VCVARS (
    echo ERROR: could not find vcvars64.bat - install Visual Studio 2019/2022
    echo        with the "Desktop development with C++" workload.
    exit /b 1
)

echo Using %VCVARS%
rem  vcvars64.bat itself prints harmless noise on some installs; keep it quiet.
call "%VCVARS%" >nul 2>nul
if errorlevel 1 ( echo ERROR: vcvars64 failed & exit /b 1 )

rem --- compile ---------------------------------------------------------------
rem  No CUDA / OpenCL SDK needed: the OpenCL entry points are resolved at run
rem  time from the driver's OpenCL.dll.
cl /nologo /O2 /EHsc /std:c++17 /W3 /MT /D_CRT_SECURE_NO_WARNINGS /Fe:mersenne_tf0.9.exe mersenne_tf0.9.cpp
if errorlevel 1 ( echo. & echo BUILD FAILED & exit /b 1 )

del /q mersenne_tf0*.obj 2>nul
echo.
echo Built mersenne_tf0.9.exe
echo   mersenne_tf0.9.exe --list-devices    show GPUs
echo   mersenne_tf0.9.exe --selftest        verify against known factorisations
echo   mersenne_tf0.9.exe                   run the job in config.txt
