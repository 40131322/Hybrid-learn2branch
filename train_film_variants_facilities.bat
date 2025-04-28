@echo off
setlocal enabledelayedexpansion

:: === Config ===
set PROBLEM=facilities
set SEED=0
set GPU=-1




echo.
echo ==== Step 2: FILM with pretrained GCNN (film-pre) ====
python 08_train_hybrid.py %PROBLEM% --model film --no_e2e --seed %SEED% --gpu %GPU%

echo.
echo ==== Step 3: FILM end-to-end ====
python 08_train_hybrid.py %PROBLEM% --model film --seed %SEED% --gpu %GPU%

echo.
echo ==== Step 4: FILM end-to-end + knowledge distillation ====
python 08_train_hybrid.py %PROBLEM% --model film --distilled --seed %SEED% --gpu %GPU%

echo.
echo ==== Step 5: FILM end-to-end + KD + auxiliary task (ED) ====
python 08_train_hybrid.py %PROBLEM% --model film --distilled --at ED --beta_at 0.1 --l2 0.001 --seed %SEED% --gpu %GPU%

echo.
echo ==== All FILM model variants for %PROBLEM% completed. ====
pause
