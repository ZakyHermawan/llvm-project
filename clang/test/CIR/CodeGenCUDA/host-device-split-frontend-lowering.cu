// NOTE: This file exists only to host RUN lines for split + lowering checks.
//
// RUN: printf "module {\n  cir.func @main() {\n    cir.return\n  }\n}\n" > %t.host.in.cir
// RUN: printf "module {\n  cir.func @device_only() {\n    cir.return\n  }\n}\n" > %t.device.in.cir
// RUN: %clang_cc1 -x c -fclangir -cir-combine -cir-host-input %t.host.in.cir -cir-device-input %t.device.in.cir -o %t.combined.cir
// RUN: %clang_cc1 -x c -fclangir -cir-split -cir-input %t.combined.cir -cir-host-output %t.host.out.cir -cir-device-output %t.device.out.cir
// RUN: cir-opt %t.host.out.cir -cir-to-llvm -o - | mlir-translate -allow-unregistered-dialect -mlir-to-llvmir -o %t.host.ll
// RUN: cir-opt %t.device.out.cir -cir-to-llvm -o - | mlir-translate -allow-unregistered-dialect -mlir-to-llvmir -o %t.device.ll
// RUN: FileCheck %s --check-prefix=LLVM-HOST -input-file=%t.host.ll
// RUN: FileCheck %s --check-prefix=LLVM-DEVICE -input-file=%t.device.ll

// LLVM-HOST: define{{.*}} void @main()
// LLVM-HOST-NOT: @device_only

// LLVM-DEVICE: define{{.*}} void @device_only()
// LLVM-DEVICE-NOT: @main
