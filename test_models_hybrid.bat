@echo off
REM Run all specified models for 'facilities' on CPU using 09_test_hybrid.py

SET PYTHON_SCRIPT=09_test_hybrid.py
SET PROBLEM=facilities
SET GPU=-1

FOR %%M IN (
    "baseline_torch"
    "concat"
    "film"
    "film_distilled"
    "film_distilled_ED_0.1_l2_0.001"
    "film-pre"
    "hybridsvm"
    "hybridsvm-film"
    "mlp_sigmoidal_decay"
) DO (
    echo Running model: %%~M
    call python %PYTHON_SCRIPT% %PROBLEM% -g %GPU% --model_name %%~M
)

echo Done testing all models!
pause
