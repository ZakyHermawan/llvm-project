// RUN: printf "module {}\n" > %t.host.cir
// RUN: printf "module {}\n" > %t.device.cir
// RUN: %clang_cc1 -x c -fclangir -cir-combine -cir-host-input %t.host.cir -cir-device-input %t.device.cir -o %t.combined.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-input %t.combined.cir -cir-host-output %t.split.host.cir -cir-device-output %t.split.device.cir
// RUN: %clang_cc1 -x c -fclangir -cir-combine -cir-host-input %t.split.host.cir -cir-device-input %t.split.device.cir -o %t.recombined.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-input %t.recombined.cir -cir-host-output %t.roundtrip.host.cir -cir-device-output %t.roundtrip.device.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-host-input %t.combined.cir -cir-host-output %t.compat.host.cir -cir-device-output %t.compat.device.cir
// RUN: not %clang_cc1 -x c -fclangir -cir-split -cir-input %t.combined.cir -cir-host-input %t.combined.cir -cir-host-output %t.err.host.cir -cir-device-output %t.err.device.cir 2>&1 | FileCheck %s --check-prefix=CHECK-ERR
// RUN: FileCheck %s --check-prefix=CHECK-COMBINED -input-file=%t.combined.cir
// RUN: FileCheck %s --check-prefix=CHECK-SPLIT-HOST -input-file=%t.split.host.cir
// RUN: FileCheck %s --check-prefix=CHECK-SPLIT-DEVICE -input-file=%t.split.device.cir
// RUN: FileCheck %s --check-prefix=CHECK-RECOMBINED -input-file=%t.recombined.cir
// RUN: FileCheck %s --check-prefix=CHECK-ROUNDTRIP-HOST -input-file=%t.roundtrip.host.cir
// RUN: FileCheck %s --check-prefix=CHECK-ROUNDTRIP-DEVICE -input-file=%t.roundtrip.device.cir
// RUN: FileCheck %s --check-prefix=CHECK-COMPAT-HOST -input-file=%t.compat.host.cir
// RUN: FileCheck %s --check-prefix=CHECK-COMPAT-DEVICE -input-file=%t.compat.device.cir

// CHECK-COMBINED: cir.offload.container
// CHECK-COMBINED: builtin.module attributes {cir.offload.kind = "host"}
// CHECK-COMBINED: builtin.module attributes {cir.offload.kind = "device"}

// CHECK-SPLIT-HOST: cir.offload.kind = "host"
// CHECK-SPLIT-DEVICE: cir.offload.kind = "device"

// CHECK-RECOMBINED: cir.offload.container
// CHECK-RECOMBINED: builtin.module attributes {cir.offload.kind = "host"}
// CHECK-RECOMBINED: builtin.module attributes {cir.offload.kind = "device"}

// CHECK-ROUNDTRIP-HOST: cir.offload.kind = "host"
// CHECK-ROUNDTRIP-DEVICE: cir.offload.kind = "device"

// CHECK-COMPAT-HOST: cir.offload.kind = "host"
// CHECK-COMPAT-DEVICE: cir.offload.kind = "device"

// CHECK-ERR: error: invalid argument '-cir-input' not allowed with '-cir-host-input'
