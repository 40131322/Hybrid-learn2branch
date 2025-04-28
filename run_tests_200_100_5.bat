@echo off
echo Running specific tests for facilities models...

REM Create output folders if missing
mkdir test_results\experiment\200_50_3
mkdir test_results\experiment\200_100_5

echo.
echo ========== Running MLP test ==========
echo.

REM python 10_test_mlp.py facilities --facility_config 200_100_5 --model_path trained_models/facilities/baseline_torch --start_file 5000 --end_file 15000
REM if %ERRORLEVEL% neq 0 (
REM    echo Error running MLP test
REM    pause
REM    exit /b %ERRORLEVEL%
REM)

echo.
echo ========== Running GNN test ==========
echo.

python 10_test_gcnn_torch.py facilities -g -1 --config 200_100_5 --model_name baseline_torch --max_samples 10000
if %ERRORLEVEL% neq 0 (
    echo Error running GNN test
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========== Running Hybrid tests ==========
echo.

FOR %%M IN (
    "concat"
    "film"
    "hybridsvm"
    "hybridsvm-film"
) DO (
    echo Testing hybrid model: %%~M
    python 10_test_hybrid.py facilities --config 200_100_5 --model_name %%~M --start_idx 0  --max_samples 10000
    if %ERRORLEVEL% neq 0 (
        echo Error running hybrid test for %%~M
        pause
        exit /b %ERRORLEVEL%
    )
)

echo.
echo All tests completed successfully!
pause
