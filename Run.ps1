python -m venv env
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
. .\env\Scripts\Activate.ps1
python -m pip install wheel cibuildwheel
$Env:THIRDAI_FEATURE_FLAGS='THIRDAI_EXPOSE_ALL'
$Env:CMAKE_ARGS='-DTHIRDAI_GTEST_DISCOVERY_TIMEOUT=30 -DVCPKG_TARGET_TRIPLET=arm64-windows-static -DCMAKE_TOOLCHAIN_FILE=C:\opt\vcpkg\scripts\buildsystems\vcpkg.cmake'
python setup.py bdist_wheel
python -m cibuildwheel --platform windows

