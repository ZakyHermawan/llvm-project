// RUN: %clang -fclangir -emit-cir -cir-combine -nocudainc -nogpulib -I %S/Inputs -include cuda.h %s -o %t.combined.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-input %t.combined.cir -cir-host-output %t.host.cir -cir-device-output %t.device.cir
// RUN: FileCheck %s --check-prefix=CHECK-HOST -input-file=%t.host.cir
// RUN: FileCheck %s --check-prefix=CHECK-DEVICE -input-file=%t.device.cir

// CHECK-HOST: cir.offload.kind = "host"
// CHECK-HOST: cir.func {{.*}} @main()

// CHECK-DEVICE: cir.offload.kind = "device"
// CHECK-DEVICE: cir.func {{.*}} @_Z13device_kernelPi

__global__ void device_kernel(int *out) {
  if (out)
    *out = 1;
}

int main() {
  return 0;
}
