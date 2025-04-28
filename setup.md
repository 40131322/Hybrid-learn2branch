Here’s the `.md` (Markdown) version of your uploaded `SCIP_6.0.14_Install_Guide_Windows.docx`: 

---

# SCIP Optimization Suite 6.0.1 Installation Guide (Windows)

##  Objective

Successfully build and install SCIP Optimization Suite 6.0.1 using CMake and MSBuild from the Visual Studio Build Tools, while avoiding common dependency pitfalls.

---

##  Installation Procedure

###  Step 1: Clean Previous Builds

Before starting, remove any leftover files from previous attempts:

```bash
rd /s /q D:\scip-build
rd /s /q D:\scip-install
del /q D:\scip\scipoptsuite-6.0.1\CMakeCache.txt
```

---

###  Step 2: Configure CMake Build

Create a fresh build directory:

```bash
mkdir D:\scip-build
```

Then configure the build using CMake, **disabling optional packages** that often cause build errors:

```bash
cmake -S D:\scip\scipoptsuite-6.0.1 -B D:\scip-build ^
  -DCMAKE_INSTALL_PREFIX=D:\scip-install ^
  -DZIMPL=OFF ^
  -DGMP=OFF ^
  -DREADLINE=OFF ^
  -DZLIB=OFF ^
  -DGCG=OFF
```

>  **Note:** `-DGCG=OFF` is **critical** to prevent errors related to GCG's GMP dependency.

---

###  Step 3: Build and Install SCIP

Use MSBuild via CMake to compile and install:

```bash
cmake --build D:\scip-build --config Release --target install -- /m
```

---

###  Step 4: Verify the Installation

Check that key folders exist:

```bash
dir D:\scip-install
dir D:\scip-install\include\scip
dir D:\scip-install\bin
```

Then try launching SCIP:

```bash
D:\scip-install\bin\scip.exe
```

You should see output like:

```
SCIP version 6.0.1 [release]
...
SCIP>
```

Exit by typing:

```bash
quit
```

---

###  Step 5: Install PySCIPOpt (Optional)

If you want to use the Python bindings:

```bash
rmdir /s /q trained_models\setcover\baseline_t