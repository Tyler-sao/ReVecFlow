#!/bin/bash
# Installs this repository inside a container built from the FuseFlow artifact. Run once, inside the container:
#   bash /ReVecFlow/tools/install.sh
# Creates /fuseflow-v3 (patched compiler and Comal, their own venv, scripts) and
# patches scripts/ of /fuseflow-artifact. The artifact's own compiler, Comal and venv are not changed.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
V=/fuseflow-v3
A=/fuseflow-artifact
JOBS=${JOBS:-4}
[ -e $V ] && { echo "STOP: $V exists"; exit 1; }
[ -d $A/fuseflow-compiler/external/llvm-project/build ] || { echo "STOP: $A with its built LLVM not found"; exit 1; }
for t in cmake ninja cargo maturin patch; do command -v $t > /dev/null || { echo "STOP: $t not on PATH"; exit 1; }; done
mkdir -p $V/toolchain $V/logs

apply() {   # DIR PATCH
    (cd "$1" && for f in $(grep '^+++ b/' "$2" | sed 's|^+++ b/||'); do [ -f "$f" ] && sed -i 's/\r$//' "$f"; done
     patch -p1 -F0 --no-backup-if-mismatch < "$2" > /dev/null) || { echo "STOP: $2 does not apply"; exit 1; }
    echo "applied $(basename "$2")"
}

echo "== artifact scripts and kernels"
apply $A "$REPO/patches/artifact/0001-artifact-scripts.patch"
cp "$REPO"/artifact_scripts/*.py $A/scripts/
cp -r "$REPO"/kernels/. $A/fuseflow-compiler/tests/models/

echo "== compiler"
mkdir -p $V/toolchain/compiler
tar -C $A/fuseflow-compiler --exclude=./build --exclude=./external -cf - . | tar -C $V/toolchain/compiler -xf -
ln -s $A/fuseflow-compiler/external $V/toolchain/compiler/external
apply $V/toolchain/compiler "$REPO/patches/compiler/0016-compiler-locate-parent-ref.patch"
cd $V/toolchain/compiler
cmake -G Ninja -S . -B build -DLLVM_EXTERNAL_LIT="$(which lit || echo lit)" > $V/logs/cmake.log 2>&1 || { tail -30 $V/logs/cmake.log; exit 1; }
ninja -C build -j "$JOBS" sam-opt sam-translate > $V/logs/ninja.log 2>&1 || { tail -40 $V/logs/ninja.log; exit 1; }

echo "== Comal"
mkdir -p $V/toolchain/comal
tar -C $A/comal --exclude=./target -cf - . | tar -C $V/toolchain/comal -xf -
for p in "$REPO"/patches/comal/*.patch; do apply $V/toolchain/comal "$p"; done
cp "$REPO/patches/comal/Cargo.lock" $V/toolchain/comal/
PYV=$(/opt/fuseflow-venv/bin/python -c 'import sys; print(f"python{sys.version_info[0]}.{sys.version_info[1]}")')
/opt/fuseflow-venv/bin/python -m venv $V/venv
echo "import site; site.addsitedir('/opt/fuseflow-venv/lib/$PYV/site-packages')" > $V/venv/lib/$PYV/site-packages/zz_fuseflow_venv.pth
cd $V/toolchain/comal
CARGO_BUILD_JOBS=$JOBS VIRTUAL_ENV=$V/venv PATH=$V/venv/bin:$PATH maturin develop --release 2>&1 | tail -3

echo "== scripts"
mkdir -p $V/scripts $V/cs $V/gala $V/granii $V/ablation $V/exp3 $V/exp4 $V/baselines
cp "$REPO"/scripts/*.py $V/scripts/
cp -r "$REPO/baselines/cs" $V/cs/scripts
cp -r "$REPO/baselines/gala" $V/gala/scripts
cp -r "$REPO/baselines/granii" $V/granii/scripts
for e in ablation exp3 exp4; do cp -r "$REPO/experiments/$e" $V/$e/scripts; done
cd /tmp
$V/venv/bin/python -c "import comal; assert hasattr(comal, 'run_graph_vector16_outputs'); print('INSTALL OK')"
