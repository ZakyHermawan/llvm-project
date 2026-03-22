// RUN: %clang -fclangir -emit-cir -cir-combine -nocudainc -nogpulib -I %S/Inputs -include cuda.h %s -o %t.combined.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-input %t.combined.cir -cir-host-output %t.host.cir -cir-device-output %t.device.cir
// RUN: cir-opt %t.host.cir -cir-to-llvm -o - | mlir-translate -allow-unregistered-dialect -mlir-to-llvmir -o %t.host.ll
// RUN: cir-opt %t.device.cir -cir-to-llvm -o - | mlir-translate -allow-unregistered-dialect -mlir-to-llvmir -o %t.device.ll
// RUN: FileCheck %s --check-prefix=LLVM-HOST -input-file=%t.host.ll
// RUN: FileCheck %s --check-prefix=LLVM-DEVICE -input-file=%t.device.ll

// LLVM-HOST: define{{.*}} i32 @main()
// LLVM-HOST-NOT: @_Z11device_funcPi

// LLVM-DEVICE: define{{.*}} i32 @_Z11device_funcPi
// LLVM-DEVICE-NOT: @main

__device__ int device_func(int *out) {
  if (out)
    *out = 1;
  return 7;
}

int main() {
  return 0;
}
