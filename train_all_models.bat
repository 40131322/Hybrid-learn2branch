@echo off
setlocal enabledelayedexpansion

:: === Config ===
set PROBLEM=facilities
set SEED=0
set GPU=-1

echo Training all models end-to-end for Top-1 accuracy on problem: %PROBLEM%

:: === 1. Train CONCAT ===
echo.
echo ==== Training CONCAT ====
python 08_train_hybrid.py %PROBLEM% --model concat --seed %SEED% --gpu %GPU%

:: === 2. Train BASE MLP ===
echo.
echo ==== Training BASE MLP ====
python 08_train_hybrid.py %PROBLEM% --model mlp --seed %SEED% --gpu %GPU%

:: === 3. Train HyperSVM (GNN + MLP, end-to-end) ===
echo.
echo ==== Training HYPERSVM (end-to-end) ====
python 08_train_hybrid.py %PROBLEM% --model hybridsvm --seed %SEED% --gpu %GPU%

:: === 4. Train HyperSVM-FiLM (end-to-end) ===
echo.
echo ==== Training HYPERSVM-FILM (end-to-end) ====
python 08_train_hybrid.py %PROBLEM% --model hybridsvm-film --seed %SEED% --gpu %GPU%

echo.
echo ==== All models trained end-to-end ====
pause
