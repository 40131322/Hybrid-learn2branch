@echo off
setlocal enabledelayedexpansion

:: === Config ===
set PROBLEM=setcover
set SEED=0
set GPU=-1

echo.
echo ==== Pre-training GCNN teacher (baseline_torch) ====
python 03_train_gcnn_torch.py %PROBLEM% --model baseline_torch --seed %SEED% --gpu %GPU%

echo.
echo ==== 1. FILM with pretrained GCNN (film-pre) ====
python 03_train_hybrid.py %PROBLEM% --model film --no_e2e --seed %SEED% --gpu %GPU%

echo.
echo ==== 2. FILM end-to-end ====
python 03_train_hybrid.py %PROBLEM% --model film --seed %SEED% --gpu %GPU%

echo.
echo ==== 3. FILM end-to-end + knowledge distillation ====
python 03_train_hybrid.py %PROBLEM% --model film --distilled --seed %SEED% --gpu %GPU%

echo.
echo ==== 4. FILM end-to-end + KD + auxiliary task (ED) ====
python 03_train_hybrid.py %PROBLEM% --model film --distilled --at ED --beta_at 0.1 --l2 0.001 --seed %SEED% --gpu %GPU%

echo.
echo ==== All FILM model variants for %PROBLEM% trained. ====
pause
