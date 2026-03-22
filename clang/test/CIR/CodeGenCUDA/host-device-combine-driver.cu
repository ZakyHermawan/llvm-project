// RUN: %clang -fclangir -emit-cir -cir-combine %s -o %t.cir
// RUN: FileCheck %s -input-file=%t.cir

// CHECK: cir.offload.container

// CHECK: builtin.module attributes {{{.*}}cir.offload.kind = "host"{{.*}}}
// CHECK: cir.func {{.*}} @main()
// CHECK: %[[D_OUT:[0-9]+]] = cir.alloca {{.*}}["d_out", init]
// CHECK: %[[LOADED:[0-9]+]] = cir.load {{.*}} %[[D_OUT]]
// CHECK: cir.call @_Z28__device_stub__device_kernelPi(%[[LOADED]]) {cu.kernel_name = #cir.cu.kernel_name<_Z13device_kernelPi>}

// CHECK: builtin.module attributes {{{.*}}cir.offload.kind = "device"{{.*}}}
// CHECK: cir.func {{.*}} @_Z13device_kernelPi

__global__ void device_kernel(int *out) {
  if (out)
    *out = 1;
}

int main() {
  int *d_out = 0;
  device_kernel<<<1, 1>>>(d_out);
  return 0;
}
