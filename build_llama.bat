@echo off
setlocal EnableExtensions
rem ===========================================================================
rem build_llama.bat -- Windows build for llama.cpp (HIP / ROCm) with an optional
rem                    ck_tile FMHA (composable_kernel) integration.
rem
rem Windows counterpart of build_llama.sh. The Linux flow is left untouched; this
rem mirrors its option surface.
rem
rem The ck path is OPT-IN via --ck. When NOT given, nothing CK-related is built,
rem linked or staged (GGML_HIP_CK_FMHA stays OFF, fattn-ck compiles to stubs).
rem Building the CK library is controlled separately by --build_ck:
rem   * --build_ck (implies --ck): compile the ck_tile_fmha SHARED target in the
rem     sibling composable_kernel checkout, then vendor ck_tile_fmha.{dll,lib} into
rem     ggml/src/ggml-cuda/ck-fmha/lib/.
rem   * --ck without --build_ck: reuse the already-vendored ck_tile_fmha.dll/.lib.
rem
rem Usage:
rem   build_llama.bat [--archs "gfx1151;gfx1100"] [--ck] [--build_ck] [--native]
rem                   [--jobs N] [--rocm "C:\Program Files\AMD\ROCm\7.2"]
rem                   [--ck-dir DIR] [--rocm-cmake DIR] [--clean] [--ck-clean]
rem
rem   --archs LIST   Semicolon/comma separated GPU targets for -DGPU_TARGETS,
rem                  e.g. "gfx1151" or "gfx1151;gfx1100". Default: gfx1151.
rem                  Supported: gfx1100 gfx1101 gfx1102 gfx1150 gfx1151 gfx1152
rem                  gfx1201 gfx1202.
rem   --ck           Link the ck_tile FMHA path (-DGGML_HIP_CK_FMHA=ON).
rem   --build_ck     (implies --ck) compile + vendor the ck_tile_fmha lib first.
rem   --native       Bake -march=native into the CPU backend (host-only binaries).
rem   --jobs N       Parallel build jobs (default: %NUMBER_OF_PROCESSORS%).
rem   --rocm PATH    ROCm / HIP SDK install (default: C:\Program Files\AMD\ROCm\7.2).
rem                  If that clang's resource dir lacks its intrinsic headers, a
rem                  sibling ROCm install with a self-consistent clang is used.
rem   --ck-dir DIR   composable_kernel checkout (default: sibling of llama.cpp).
rem   --rocm-cmake D ROCmCMakeBuildTools cmake config dir (auto-provisioned if omitted).
rem   --clean        Remove the llama build dir before configuring.
rem   --ck-clean     Remove the CK build dir before configuring.
rem   -h / --help    Show this help.
rem
rem Env overrides: ARCHS, JOBS, ROCM, CK_DIR.
rem ===========================================================================

rem ---- defaults (env overrides honored, same as the .sh) ----
if not defined ARCHS set "ARCHS=gfx1151"
if not defined ROCM  set "ROCM=C:\Program Files\AMD\ROCm\7.2"
if not defined JOBS  set "JOBS=%NUMBER_OF_PROCESSORS%"

set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
set "LLAMA_DIR=%SCRIPT_DIR%"
for %%I in ("%LLAMA_DIR%\..") do set "REPO_ROOT=%%~fI"
if not defined CK_DIR set "CK_DIR=%REPO_ROOT%\composable_kernel"
if not defined ROCM_CMAKE_DIR set "ROCM_CMAKE_DIR=%REPO_ROOT%\_tools\rocm-cmake-install"

set "WITH_CK=0"
set "BUILD_CK=0"
set "NATIVE=OFF"
set "DO_CLEAN=0"
set "CK_CLEAN=0"
set "ROCM_CMAKE_ARG="

rem ---- parse args ----
:parse
if "%~1"=="" goto after_parse
if /I "%~1"=="--archs"      ( set "ARCHS=%~2"   & shift & shift & goto parse )
if /I "%~1"=="--arch"       ( set "ARCHS=%~2"   & shift & shift & goto parse )
if /I "%~1"=="--ck"         ( set "WITH_CK=1"   & shift & goto parse )
if /I "%~1"=="--build_ck"   ( set "BUILD_CK=1"  & set "WITH_CK=1" & shift & goto parse )
if /I "%~1"=="--native"     ( set "NATIVE=ON"   & shift & goto parse )
if /I "%~1"=="--jobs"       ( set "JOBS=%~2"    & shift & shift & goto parse )
if /I "%~1"=="--rocm"       ( set "ROCM=%~2"    & shift & shift & goto parse )
if /I "%~1"=="--ck-dir"     ( set "CK_DIR=%~2"  & shift & shift & goto parse )
if /I "%~1"=="--rocm-cmake" ( set "ROCM_CMAKE_ARG=%~2" & shift & shift & goto parse )
if /I "%~1"=="--clean"      ( set "DO_CLEAN=1"  & shift & goto parse )
if /I "%~1"=="--ck-clean"   ( set "CK_CLEAN=1"  & shift & goto parse )
if /I "%~1"=="-h"           goto usage
if /I "%~1"=="--help"       goto usage
echo [build_llama] ERROR: unknown argument: %~1 1>&2
goto usage

:after_parse

rem Normalize separators: accept both ';' and ',' -> CMake wants ';'.
set "ARCHS=%ARCHS:,=;%"
set "ROCM_PATH=%ROCM%"

rem ---- pick a ROCm whose clang matches its own intrinsic headers ----
rem Some installs ship a clang whose resource dir (lib/clang/<N>) has no headers
rem (mismatched compiler/headers), which breaks the host x86 + HIP device builds.
call :select_rocm
if errorlevel 1 goto err_no_consistent_rocm

if not exist "%ROCM_PATH%\bin\clang++.exe" (
    echo [build_llama] ERROR: clang++ not found under "%ROCM_PATH%\bin". Pass --rocm. 1>&2
    exit /b 1
)

rem ---- build/output layout ----
set "LLAMA_BUILD=%LLAMA_DIR%\build"
rem CK's generated FMHA kernel filenames are very long; keep the CK build root
rem short (and on the same drive) so object paths stay within Windows limits.
set "CK_BUILD=%REPO_ROOT%\ckb"
set "CK_VENDOR_DIR=%LLAMA_DIR%\ggml\src\ggml-cuda\ck-fmha\lib"
set "BIN=%LLAMA_BUILD%\bin"

echo ============================================================================
echo   llama.cpp         : %LLAMA_DIR%
echo   composable_kernel : %CK_DIR%
echo   ROCm / HIP SDK    : %ROCM_PATH%
echo   GPU targets       : %ARCHS%
echo   ck FMHA           : with_ck=%WITH_CK% build_ck=%BUILD_CK%
echo   native / jobs     : %NATIVE% / %JOBS%
echo ============================================================================

rem ---- Visual Studio dev environment (Windows SDK libs + Ninja + lld) ----
call :setup_msvc
if errorlevel 1 exit /b 1

rem ---- prepend ROCm to PATH so clang finds the hip runtime + device libs ----
set "PATH=%ROCM_PATH%\bin;%PATH%"
set "CC=clang"
set "CXX=clang++"

rem ---- resolve HIP toolchain details (short paths, resource dir, flags) ----
call :setup_hip
if errorlevel 1 exit /b 1

rem ---- locate a Ninja generator ----
set "NINJA="
for /f "delims=" %%N in ('where ninja 2^>nul') do if not defined NINJA set "NINJA=%%N"
if not defined NINJA (
    echo [build_llama] ERROR: ninja.exe not found on PATH -- install Ninja or run from a VS dev env. 1>&2
    exit /b 1
)
echo [build_llama] using ninja: %NINJA%

if "%DO_CLEAN%"=="1" (
    echo [build_llama] cleaning llama build dir...
    if exist "%LLAMA_BUILD%" rmdir /s /q "%LLAMA_BUILD%"
)

rem ========================================================================
rem  Step A: build + vendor the ck_tile FMHA shared library (only --build_ck)
rem ========================================================================
if not "%BUILD_CK%"=="1" goto skip_build_ck
call :build_ck
if errorlevel 1 exit /b 1
:skip_build_ck

rem ========================================================================
rem  Step B: configure + build llama.cpp
rem ========================================================================
set "CK_CMAKE_FLAG=OFF"
if "%WITH_CK%"=="1" set "CK_CMAKE_FLAG=ON"

echo [build_llama] configuring llama.cpp (GGML_HIP_CK_FMHA=%CK_CMAKE_FLAG%)...
cmake -S "%LLAMA_DIR%" -B "%LLAMA_BUILD%" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DGGML_HIP=ON ^
    -DGGML_HIP_CK_FMHA=%CK_CMAKE_FLAG% ^
    "-DGPU_TARGETS=%ARCHS%" ^
    "-DAMDGPU_TARGETS=%ARCHS%" ^
    "-DCMAKE_HIP_ARCHITECTURES=%ARCHS%" ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DCMAKE_HIP_COMPILER=clang++ ^
    -DCMAKE_HIP_PLATFORM=amd ^
    -DGGML_NATIVE=%NATIVE% ^
    "-DCMAKE_C_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_CXX_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_HIP_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_EXE_LINKER_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_SHARED_LINKER_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_MODULE_LINKER_FLAGS=%HIP_EXTRA_FLAGS%"
if errorlevel 1 ( echo [build_llama] ERROR: llama.cpp cmake configure failed. 1>&2 & exit /b 1 )

echo [build_llama] building llama.cpp...
cmake --build "%LLAMA_BUILD%" --target llama-bench llama-server llama-cli test-backend-ops -j %JOBS%
if errorlevel 1 ( echo [build_llama] ERROR: llama.cpp build failed. 1>&2 & exit /b 1 )

echo.
echo [build_llama] DONE. binaries in %BIN%
if "%WITH_CK%"=="1" (
    if exist "%CK_VENDOR_DIR%\ck_tile_fmha.dll" (
        if not exist "%BIN%" mkdir "%BIN%"
        copy /y "%CK_VENDOR_DIR%\ck_tile_fmha.dll" "%BIN%\ck_tile_fmha.dll" >nul
        echo [build_llama] staged ck_tile_fmha.dll next to the binaries -- auto-loaded when GGML_CK_FA=1.
    ) else (
        echo [build_llama] WARN: %CK_VENDOR_DIR%\ck_tile_fmha.dll missing; run with --build_ck. 1>&2
    )
    echo.
    echo Run with ck FMHA ^(dense d128, e.g. Llama^):
    echo   set GGML_CK_FA=1 ^&^& set GGML_CK_FA_CAUSAL=1 ^&^& set GGML_CK_FA_DECODE=1 ^&^& set GGML_CUDA_FATTN_FORCE=ck
    echo   "%BIN%\llama-bench.exe" -m model.gguf -fa 1 -ngl 99
) else (
    echo [build_llama] built without ck_tile FMHA; pass --ck to enable it
)
exit /b 0


rem ========================================================================
rem  Subroutine: build the CK ck_tile FMHA shared library and vendor it
rem ========================================================================
:build_ck
if not exist "%CK_DIR%\CMakeLists.txt" (
    echo [build_llama] ERROR: composable_kernel not found at "%CK_DIR%" -- use --ck-dir. 1>&2
    exit /b 1
)
if not exist "%CK_DIR%\example\ck_tile\01_fmha\ck_tile_fmha_c_api.cpp" (
    echo [build_llama] ERROR: ck C-ABI shim missing in %CK_DIR%; apply the CK patch first. 1>&2
    exit /b 1
)
call :find_rocm_cmake
if defined ROCM_CMAKE_CFG goto have_rocm_cmake
call :provision_rocm_cmake
if errorlevel 1 exit /b 1
call :find_rocm_cmake
:have_rocm_cmake
if not defined ROCM_CMAKE_CFG (
    echo [build_llama] ERROR: ROCmCMakeBuildTools / rocm-cmake not found and could not be provisioned. 1>&2
    exit /b 1
)
echo [build_llama] rocm-cmake: %ROCM_CMAKE_DIRFS%

if "%CK_CLEAN%"=="1" if exist "%CK_BUILD%" (
    echo [build_llama] cleaning CK build dir...
    rmdir /s /q "%CK_BUILD%"
)

rem Reuse a previously built CK DLL (avoids recompiling ~2000 kernels). Pass
rem --ck-clean (or delete the CK build dir) to force a full CK rebuild.
if exist "%CK_BUILD%\b\bin\ck_tile_fmha.dll" (
    echo [build_llama] reusing existing CK DLL under %CK_BUILD%
    goto vendor_ck
)

rem CK's generated FMHA object filenames are ~185 chars; the full object path
rem blows past Windows' 260-char limit at link time. Map the (in-repo) CK build
rem dir to a short virtual drive via subst so paths stay short while the files
rem remain inside D:\code\Staging\llama. Reconfiguring an existing tree with this
rem CMake/ROCmCMakeBuildTools combo is unreliable, so always start clean.
if exist "%CK_BUILD%" rmdir /s /q "%CK_BUILD%"
mkdir "%CK_BUILD%"
call :subst_ck
if errorlevel 1 exit /b 1

echo [build_llama] configuring composable_kernel (ck_tile FMHA) on %CK_DRV%...
cmake -S "%CK_DIR%" -B "%CK_DRV%/b" -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_C_COMPILER=clang ^
    -DCMAKE_CXX_COMPILER=clang++ ^
    -DCMAKE_HIP_COMPILER=clang++ ^
    -DCMAKE_HIP_PLATFORM=amd ^
    "-DCMAKE_HIP_ARCHITECTURES=%ARCHS%" ^
    "-DGPU_TARGETS=%ARCHS%" ^
    -DCMAKE_OBJECT_PATH_MAX=1000 ^
    "-DROCmCMakeBuildTools_DIR=%ROCM_CMAKE_DIRFS%" ^
    "-DCMAKE_C_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_CXX_FLAGS=%HIP_EXTRA_FLAGS% -Wno-gnu-line-marker -fbracket-depth=1024" ^
    "-DCMAKE_HIP_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_EXE_LINKER_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_SHARED_LINKER_FLAGS=%HIP_EXTRA_FLAGS%" ^
    "-DCMAKE_MODULE_LINKER_FLAGS=%HIP_EXTRA_FLAGS%" ^
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON ^
    -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON ^
    -DENABLE_CLANG_CPP_CHECKS=OFF ^
    -DBUILD_DEV=OFF ^
    -DBUILD_TESTING=OFF ^
    -DBUILD_CK_EXAMPLES=ON ^
    -DBUILD_CK_TUTORIALS=OFF ^
    -DBUILD_CK_TILE_ENGINE=OFF ^
    -DBUILD_CK_PROFILER=OFF ^
    -DBUILD_CK_DEVICE_INSTANCES=OFF ^
    -DFMHA_FWD_ENABLE_APIS=fwd ^
    -DFMHA_FWD_RECEIPT=2 ^
    "-DFMHA_FWD_OPTDIM=128,256"
if errorlevel 1 ( call :unsubst_ck & echo [build_llama] ERROR: composable_kernel cmake configure failed. 1>&2 & exit /b 1 )

echo [build_llama] building ck_tile_fmha target (this compiles the FMHA instances)...
cmake --build "%CK_DRV%/b" --target ck_tile_fmha -j %JOBS%
set "CK_RC=%ERRORLEVEL%"
call :unsubst_ck
if not "%CK_RC%"=="0" ( echo [build_llama] ERROR: ck_tile_fmha build failed. 1>&2 & exit /b 1 )

:vendor_ck
set "CK_DLL="
for /r "%CK_BUILD%" %%F in (ck_tile_fmha.dll) do if not defined CK_DLL if exist "%%F" set "CK_DLL=%%F"
if not defined CK_DLL ( echo [build_llama] ERROR: ck_tile_fmha.dll not found under "%CK_BUILD%". 1>&2 & exit /b 1 )
set "CK_LIB="
for /r "%CK_BUILD%" %%F in (ck_tile_fmha.lib) do if not defined CK_LIB if exist "%%F" set "CK_LIB=%%F"
if not defined CK_LIB ( echo [build_llama] ERROR: ck_tile_fmha.lib import library not found under "%CK_BUILD%". 1>&2 & exit /b 1 )

echo [build_llama] vendoring ck_tile_fmha.dll/.lib -^> %CK_VENDOR_DIR%
if not exist "%CK_VENDOR_DIR%" mkdir "%CK_VENDOR_DIR%"
copy /y "%CK_DLL%" "%CK_VENDOR_DIR%\ck_tile_fmha.dll" >nul
if errorlevel 1 ( echo [build_llama] ERROR: failed to vendor ck_tile_fmha.dll. 1>&2 & exit /b 1 )
copy /y "%CK_LIB%" "%CK_VENDOR_DIR%\ck_tile_fmha.lib" >nul
if errorlevel 1 ( echo [build_llama] ERROR: failed to vendor ck_tile_fmha.lib. 1>&2 & exit /b 1 )
exit /b 0


rem ========================================================================
rem  Subroutine: map / unmap the CK build dir to a short virtual drive (subst)
rem ========================================================================
:subst_ck
set "CK_DRV="
for %%L in (X Y Z W V U T S R Q P) do if not defined CK_DRV if not exist "%%L:\" set "CK_DRV=%%L:"
if not defined CK_DRV ( echo [build_llama] ERROR: no free drive letter for subst. 1>&2 & exit /b 1 )
subst "%CK_DRV%" "%CK_BUILD%"
if errorlevel 1 ( echo [build_llama] ERROR: subst %CK_DRV% failed. 1>&2 & set "CK_DRV=" & exit /b 1 )
echo [build_llama] mapped %CK_DRV% -^> %CK_BUILD%
exit /b 0

:unsubst_ck
if defined CK_DRV subst "%CK_DRV%" /D >nul 2>&1
set "CK_DRV="
exit /b 0


rem ========================================================================
rem  Subroutine: initialise the MSVC / Windows SDK developer environment
rem ========================================================================
:setup_msvc
call :find_vsdir
if not defined VSDIR (
    echo [build_llama] ERROR: could not locate a VS 2022 install with the C++ workload. 1>&2
    exit /b 1
)
if not defined VCINSTALLDIR (
    set "VCVARS=%VSDIR%\VC\Auxiliary\Build\vcvars64.bat"
    if not exist "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" (
        echo [build_llama] ERROR: vcvars64.bat not found under "%VSDIR%". 1>&2
        exit /b 1
    )
    echo [build_llama] using VS env: %VSDIR%
    call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul
    if errorlevel 1 ( echo [build_llama] ERROR: vcvars64.bat failed. 1>&2 & exit /b 1 )
)
set "VS_NINJA=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja"
if exist "%VS_NINJA%\ninja.exe" set "PATH=%VS_NINJA%;%PATH%"
rem Some VS installs don't register the Windows SDK (kernel32.lib missing from LIB).
if not defined WindowsSdkDir call :setup_winsdk
exit /b 0

:find_vsdir
if defined VSDIR exit /b 0
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" goto find_vsdir_fallback
for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2^>nul`) do set "VSDIR=%%I"
if defined VSDIR exit /b 0
:find_vsdir_fallback
for %%E in (Enterprise Professional Community BuildTools) do if not defined VSDIR if exist "%ProgramFiles%\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VSDIR=%ProgramFiles%\Microsoft Visual Studio\2022\%%E"
exit /b 0

:setup_winsdk
rem Use the 8.3 short path for the Kits root so the "(x86)" parentheses don't
rem break cmd parsing, then inject the newest SDK that has both headers and libs.
for %%I in ("%ProgramFiles(x86)%\Windows Kits\10") do set "WSDK_ROOT=%%~sI"
if not exist "%WSDK_ROOT%\Lib" exit /b 0
set "WSDK_VER="
for /f "delims=" %%V in ('dir /b /ad /o-n "%WSDK_ROOT%\Lib" 2^>nul') do if not defined WSDK_VER if exist "%WSDK_ROOT%\Lib\%%V\um\x64\kernel32.lib" if exist "%WSDK_ROOT%\Include\%%V\um\windows.h" set "WSDK_VER=%%V"
if not defined WSDK_VER ( echo [build_llama] WARNING: no complete Windows SDK found. 1>&2 & exit /b 0 )
echo [build_llama] using Windows SDK: %WSDK_VER%
set "INCLUDE=%WSDK_ROOT%\Include\%WSDK_VER%\ucrt;%WSDK_ROOT%\Include\%WSDK_VER%\um;%WSDK_ROOT%\Include\%WSDK_VER%\shared;%WSDK_ROOT%\Include\%WSDK_VER%\winrt;%INCLUDE%"
set "LIB=%WSDK_ROOT%\Lib\%WSDK_VER%\ucrt\x64;%WSDK_ROOT%\Lib\%WSDK_VER%\um\x64;%LIB%"
set "PATH=%WSDK_ROOT%\bin\%WSDK_VER%\x64;%PATH%"
exit /b 0


rem ========================================================================
rem  Subroutine: choose a ROCm whose clang resource dir has the headers
rem ========================================================================
:select_rocm
call :rocm_consistent "%ROCM_PATH%"
if "%ROCM_OK%"=="1" exit /b 0
for %%I in ("%ROCM_PATH%") do set "ROCM_LEAF=%%~nxI"
echo [build_llama] clang at "%ROCM_PATH%" lacks matching intrinsic headers; searching...
set "ROCM_SEL="
for /d %%D in ("%ProgramFiles%\AMD\ROCm\%ROCM_LEAF%*") do call :rocm_pick "%%~fD"
if not defined ROCM_SEL for /d %%D in ("%ProgramFiles%\AMD\ROCm\*") do call :rocm_pick "%%~fD"
if not defined ROCM_SEL exit /b 1
set "ROCM_PATH=%ROCM_SEL%"
echo [build_llama] selected ROCm: %ROCM_PATH%
exit /b 0

:rocm_pick
if defined ROCM_SEL exit /b 0
call :rocm_consistent "%~1"
if "%ROCM_OK%"=="1" set "ROCM_SEL=%~1"
exit /b 0

:rocm_consistent
set "ROCM_OK=0"
if not exist "%~1\bin\clang.exe" exit /b 0
for /f "delims=" %%R in ('"%~1\bin\clang.exe" -print-resource-dir 2^>nul') do if exist "%%R\include\immintrin.h" set "ROCM_OK=1"
exit /b 0


rem ========================================================================
rem  Subroutine: resolve HIP toolchain flags (short path + resource dir)
rem ========================================================================
:setup_hip
set "HIP_PATH=%ROCM_PATH%"
rem 8.3 short path (forward slashes): no spaces, and dodges the CMake backslash
rem escape bug that mangles the HIP root when baked into generated cmake files.
for %%I in ("%ROCM_PATH%") do set "ROCM_SHORT=%%~sI"
set "ROCM_FS=%ROCM_SHORT:\=/%"
rem Point clang at the resource dir that actually ships the HIP wrapper header.
set "HIP_RESDIR="
for /f "delims=" %%D in ('dir /b /ad "%ROCM_PATH%\lib\clang" 2^>nul') do if exist "%ROCM_PATH%\lib\clang\%%D\include\__clang_hip_runtime_wrapper.h" set "HIP_RESDIR=%ROCM_FS%/lib/clang/%%D"
if defined HIP_RESDIR (
    set "HIP_EXTRA_FLAGS=-resource-dir %HIP_RESDIR% --rocm-path=%ROCM_FS%"
) else (
    set "HIP_EXTRA_FLAGS=--rocm-path=%ROCM_FS%"
)
set "HIP_PATH=%ROCM_FS%"
set "ROCM_PATH=%ROCM_FS%"
echo [build_llama] HIP root (short): %ROCM_FS%
if defined HIP_RESDIR echo [build_llama] clang resource dir: %HIP_RESDIR%
exit /b 0


rem ========================================================================
rem  Subroutine: locate the ROCmCMakeBuildTools (rocm-cmake) cmake config
rem ========================================================================
:find_rocm_cmake
set "ROCM_CMAKE_CFG="
set "ROCM_CMAKE_DIRFS="
if defined ROCM_CMAKE_ARG call :try_rocm_cmake "%ROCM_CMAKE_ARG%"
if defined ROCM_CMAKE_CFG exit /b 0
call :try_rocm_cmake "%ROCM_CMAKE_DIR%\share\rocmcmakebuildtools\cmake"
if defined ROCM_CMAKE_CFG exit /b 0
call :try_rocm_cmake "%ROCM_CMAKE_DIR%"
if defined ROCM_CMAKE_CFG exit /b 0
call :try_rocm_cmake "%ROCM%\share\rocmcmakebuildtools\cmake"
if defined ROCM_CMAKE_CFG exit /b 0
call :try_rocm_cmake "%ROCM%\lib\cmake\ROCmCMakeBuildTools"
exit /b 0

:try_rocm_cmake
if defined ROCM_CMAKE_CFG exit /b 0
if not exist "%~1\ROCmCMakeBuildToolsConfig.cmake" exit /b 0
set "ROCM_CMAKE_CFG=%~1\ROCmCMakeBuildToolsConfig.cmake"
set "ROCM_CMAKE_DIRFS=%~1"
set "ROCM_CMAKE_DIRFS=%ROCM_CMAKE_DIRFS:\=/%"
exit /b 0

:provision_rocm_cmake
echo [build_llama] provisioning rocm-cmake (ROCmCMakeBuildTools) into %ROCM_CMAKE_DIR%...
where git >nul 2>&1
if errorlevel 1 ( echo [build_llama] ERROR: git not found; needed to fetch rocm-cmake. 1>&2 & exit /b 1 )
set "RCM_SRC=%REPO_ROOT%\_tools\rocm-cmake"
if not exist "%RCM_SRC%\CMakeLists.txt" git clone --depth 1 https://github.com/ROCm/rocm-cmake.git "%RCM_SRC%"
if errorlevel 1 ( echo [build_llama] ERROR: rocm-cmake clone failed. 1>&2 & exit /b 1 )
cmake -S "%RCM_SRC%" -B "%RCM_SRC%\build" "-DCMAKE_INSTALL_PREFIX=%ROCM_CMAKE_DIR%"
if errorlevel 1 ( echo [build_llama] ERROR: rocm-cmake configure failed. 1>&2 & exit /b 1 )
cmake --build "%RCM_SRC%\build" --target install --config Release
if errorlevel 1 ( echo [build_llama] ERROR: rocm-cmake install failed. 1>&2 & exit /b 1 )
exit /b 0


:err_no_consistent_rocm
echo [build_llama] ERROR: no ROCm install found whose clang matches its own headers. 1>&2
echo               The clang resource dir "lib\clang\NN\include" is missing its intrinsics. 1>&2
echo               Repair the HIP SDK or pass --rocm to a consistent install. 1>&2
exit /b 1

:usage
echo.
echo Usage: build_llama.bat [--archs "gfx1151;gfx1100"] [--ck] [--build_ck] [--native]
echo                        [--jobs N] [--rocm "C:\Program Files\AMD\ROCm\7.2"]
echo                        [--ck-dir DIR] [--rocm-cmake DIR] [--clean] [--ck-clean]
echo.
echo   --archs LIST   Semicolon/comma separated GPU targets for -DGPU_TARGETS.
echo   --ck           Link the ck_tile FMHA path (-DGGML_HIP_CK_FMHA=ON).
echo   --build_ck     (implies --ck) compile + vendor the ck_tile_fmha lib first.
echo   --native       Bake -march=native into the CPU backend.
echo   --jobs N       Parallel build jobs.
echo   --rocm PATH    ROCm / HIP SDK install path.
echo   --ck-dir DIR   composable_kernel checkout.
echo   --rocm-cmake D ROCmCMakeBuildTools cmake config dir.
echo   --clean        Remove the llama build dir first.
echo   --ck-clean     Remove the CK build dir first.
exit /b 0
