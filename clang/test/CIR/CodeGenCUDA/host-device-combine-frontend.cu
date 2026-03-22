// RUN: printf "module {}\n" > %t.host.cir
// RUN: printf "module {}\n" > %t.device.cir
// RUN: %clang_cc1 -x c -fclangir -cir-combine -cir-host-input %t.host.cir -cir-device-input %t.device.cir -o %t.cir
// RUN: FileCheck %s -input-file=%t.cir

// CHECK: cir.offload.container
// CHECK: builtin.module attributes {cir.offload.kind = "host"}
// CHECK: builtin.module attributes {cir.offload.kind = "device"}
