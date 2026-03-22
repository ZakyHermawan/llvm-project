//===--- CIRCombineSplitUtils.h - CIR combine/split helpers ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Shared helpers for CIR combine/split frontend actions.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_FRONTENDACTION_CIRCOMBINESPLITUTILS_H
#define CLANG_CIR_FRONTENDACTION_CIRCOMBINESPLITUTILS_H

#include "clang/Basic/Diagnostic.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

#include <utility>

namespace clang {
class CompilerInstance;
} // namespace clang

namespace llvm {
class MemoryBufferRef;
class raw_ostream;
} // namespace llvm

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace cir {
// Load the dialect set expected by CIR combine/split actions.
void loadCIRCombineSplitDialects(mlir::MLIRContext &MlirContext);

// Read and parse a CIR module from disk and report diagnostics on failure.
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
parseCIRModuleFromFile(clang::CompilerInstance &Ci, llvm::StringRef InputPath,
                       mlir::MLIRContext &MlirContext);

// Extract host/device modules from a top-level offload container module.
mlir::FailureOr<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
getHostDeviceModules(mlir::ModuleOp CombinedModule,
                     clang::DiagnosticsEngine &Diags,
                     llvm::StringRef ActionName,
                     llvm::StringRef MissingContainerMessage);

// Create a new ModuleOp from memory buffer
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
parseCIRModuleFromBuffer(llvm::MemoryBufferRef BufferRef,
                         mlir::MLIRContext &MlirContext);

// Combine host and device module into a single offload container module
mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
buildCombinedOffloadModule(mlir::ModuleOp HostModule,
                           mlir::ModuleOp DeviceModule,
                           mlir::MLIRContext &MlirContext);

// Print Module to Output
bool emitCIRModuleToFile(mlir::ModuleOp Module, llvm::StringRef Output,
                         clang::DiagnosticsEngine &Diags);

// Print Module to stream
bool emitCIRModuleToStream(mlir::ModuleOp Module, llvm::raw_ostream &OS,
                           clang::DiagnosticsEngine &Diags);

} // namespace cir

#endif // CLANG_CIR_FRONTENDACTION_CIRCOMBINESPLITUTILS_H
