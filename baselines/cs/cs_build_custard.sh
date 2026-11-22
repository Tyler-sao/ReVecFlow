#!/bin/bash
# Build the Custard compiler (SAM's TACO fork) inside fuseflow-v3, isolated from everything else.
#   docker exec -i fuseflow-v3 bash -s < cs_build_custard.sh
# Source is copied from /fuseflow-artifact/sam/compiler/taco (read only) to /fuseflow-v3/third_party/custard/taco.
# Nothing is installed: no `make install`, no pip, Comal and the venvs are not touched.
# The build runs detached (nice 19, 2 jobs); progress: /fuseflow-v3/third_party/custard/build.log
set -e
W=/fuseflow-v3/third_party/custard
SRC=/fuseflow-artifact/sam/compiler/taco
mkdir -p "$W"
cd "$W"

if [ ! -d taco ]; then
  cp -a "$SRC" taco
  (cd taco && find . -type f \( -name '*.cpp' -o -name '*.h' -o -name 'CMakeLists.txt' \) -print0 | sort -z | xargs -0 md5sum) > SOURCE_MANIFEST.md5
  echo "copied $SRC -> $W/taco ($(wc -l < SOURCE_MANIFEST.md5) source files)"
fi

# TACO's CMake refuses to configure without the pybind11 submodule, even with PYTHON=OFF.
# The artifact copy has no .git, so the submodule may be missing. If so, disable only that check
# in OUR copy and record the change.
if [ ! -f taco/python_bindings/pybind11/CMakeLists.txt ]; then
  if ! grep -q 'CS_LOCAL_CHANGE' taco/CMakeLists.txt; then
    cp taco/CMakeLists.txt taco/CMakeLists.txt.orig
    sed -i 's|^if(NOT EXISTS "${TACO_PROJECT_DIR}/python_bindings/pybind11/CMakeLists.txt")|if(PYTHON AND NOT EXISTS "${TACO_PROJECT_DIR}/python_bindings/pybind11/CMakeLists.txt")  # CS_LOCAL_CHANGE|' taco/CMakeLists.txt
    diff -u taco/CMakeLists.txt.orig taco/CMakeLists.txt > LOCAL_CHANGES.diff || true
    echo "pybind11 submodule missing: check limited to PYTHON=ON, see $W/LOCAL_CHANGES.diff"
  fi
fi

echo "== toolchain"; cmake --version | head -1; g++ --version | head -1
cmake -S taco -B build -DCMAKE_BUILD_TYPE=Release -DPYTHON=OFF -DOPENMP=OFF -DCUDA=OFF > configure.log 2>&1 \
  || { echo "CONFIGURE FAILED, tail of $W/configure.log:"; tail -30 configure.log; exit 1; }
echo "configure ok"

setsid nohup nice -n 19 cmake --build build --target taco-tool -j 2 > build.log 2>&1 &
echo $! > build.pid
echo "build started, pid $(cat build.pid), log $W/build.log"
