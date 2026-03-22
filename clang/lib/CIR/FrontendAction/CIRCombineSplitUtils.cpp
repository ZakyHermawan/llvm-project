//===--- CIRCombineSplitUtils.cpp - CIR combine/split helpers ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRCombineSplitUtils.h"

#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include <optional>

void cir::loadCIRCombineSplitDialects(mlir::MLIRContext &MlirContext) {
  MlirContext.getOrLoadDialect<mlir::BuiltinDialect>();
  MlirContext.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  MlirContext.getOrLoadDialect<mlir::DLTIDialect>();
  MlirContext.getOrLoadDialect<cir::CIRDialect>();
  MlirContext.getOrLoadDialect<mlir::func::FuncDialect>();
  MlirContext.getOrLoadDialect<mlir::memref::MemRefDialect>();
  MlirContext.getOrLoadDialect<mlir::arith::ArithDialect>();
  MlirContext.getOrLoadDialect<mlir::omp::OpenMPDialect>();
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> cir::parseCIRModuleFromFile(
    clang::CompilerInstance &Ci, llvm::StringRef InputPath,
    mlir::MLIRContext &MlirContext) {
  auto &Diags = Ci.getDiagnostics();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> InputOrErr =
      Ci.getFileManager().getBufferForFile(InputPath);
  if (!InputOrErr) {
    std::error_code Ec = InputOrErr.getError();
    Diags.Report(clang::diag::err_fe_error_reading) << InputPath << Ec.message();
    return mlir::failure();
  }

  std::unique_ptr<llvm::MemoryBuffer> Input = std::move(*InputOrErr);
  auto ModuleOr = parseCIRModuleFromBuffer(*Input, MlirContext);
  if (mlir::failed(ModuleOr)) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "failed to parse CIR module" << InputPath;
    return mlir::failure();
  }
  return ModuleOr;
}

mlir::FailureOr<std::pair<mlir::ModuleOp, mlir::ModuleOp>>
cir::getHostDeviceModules(mlir::ModuleOp CombinedModule,
                          clang::DiagnosticsEngine &Diags,
                          llvm::StringRef ActionName,
                          llvm::StringRef MissingContainerMessage) {
  auto Container = llvm::dyn_cast_or_null<cir::OffloadContainerOp>(
      CombinedModule.getBody()->empty() ? nullptr
                                        : &CombinedModule.getBody()->front());
  if (!Container) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << ActionName << MissingContainerMessage;
    return mlir::failure();
  }

  std::optional<mlir::ModuleOp> HostModule = Container.getHostModule();
  std::optional<mlir::ModuleOp> DeviceModule = Container.getDeviceModule();
  if (!HostModule || !DeviceModule) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << ActionName << "missing host/device module in offload container";
    return mlir::failure();
  }

  return std::make_pair(*HostModule, *DeviceModule);
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
cir::parseCIRModuleFromBuffer(llvm::MemoryBufferRef BufferRef,
                              mlir::MLIRContext &MlirContext) {
  auto NormalizeCudaKernelNameAttrs = [](llvm::StringRef Input) {
    std::string Sanitized = Input.str();
    constexpr llvm::StringLiteral Prefix = "#cir.cu.kernel_name<";

    size_t Pos = 0;
    while ((Pos = Sanitized.find(Prefix.str(), Pos)) != std::string::npos) {
      size_t PayloadBegin = Pos + Prefix.size();
      if (PayloadBegin >= Sanitized.size())
        break;

      if (Sanitized[PayloadBegin] == '"') {
        Pos = PayloadBegin + 1;
        continue;
      }

      size_t PayloadEnd = Sanitized.find('>', PayloadBegin);
      if (PayloadEnd == std::string::npos)
        break;

      Sanitized.insert(PayloadEnd, "\"");
      Sanitized.insert(PayloadBegin, "\"");
      Pos = PayloadEnd + 2;
    }
    return Sanitized;
  };

  auto ParseFromString = [&](llvm::StringRef Text) {
    auto SourceMgr = std::make_shared<llvm::SourceMgr>();
    SourceMgr->AddNewSourceBuffer(
        llvm::MemoryBuffer::getMemBufferCopy(Text, BufferRef.getBufferIdentifier()),
        llvm::SMLoc());
    return mlir::parseSourceFile<mlir::ModuleOp>(SourceMgr, &MlirContext);
  };

  std::string Normalized = NormalizeCudaKernelNameAttrs(BufferRef.getBuffer());
  auto Module = ParseFromString(Normalized);
  if (Module)
    return Module;

  // Last fallback: try original text for non-CUDA parse errors.
  Module = ParseFromString(BufferRef.getBuffer());

  if (!Module)
    return mlir::failure();
  return Module;
}

mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
cir::buildCombinedOffloadModule(mlir::ModuleOp HostModule,
                                mlir::ModuleOp DeviceModule,
                                mlir::MLIRContext &MlirContext) {
  auto Loc = mlir::UnknownLoc::get(&MlirContext);
  mlir::OwningOpRef<mlir::ModuleOp> Combined = mlir::ModuleOp::create(Loc);

  // Clone and sanitize nested modules before inserting under offload.container.
  // A nested builtin.module cannot carry sym_name here because the parent
  // container is not a SymbolTable op.
  auto *HostClone = HostModule->clone();
  HostClone->removeAttr(mlir::SymbolTable::getSymbolAttrName());
  HostClone->setAttr("cir.offload.kind",
                     mlir::StringAttr::get(&MlirContext, "host"));

  auto *DeviceClone = DeviceModule->clone();
  DeviceClone->removeAttr(mlir::SymbolTable::getSymbolAttrName());
  DeviceClone->setAttr("cir.offload.kind",
                       mlir::StringAttr::get(&MlirContext, "device"));

  mlir::OpBuilder Builder = mlir::OpBuilder::atBlockBegin(Combined->getBody());
  auto Container = cir::OffloadContainerOp::create(Builder, Loc);

  mlir::Region &Body = Container.getBody();
  if (Body.empty())
    Body.push_back(new mlir::Block());

  mlir::Block &Block = Body.front();
  mlir::OpBuilder Inserter(&Block, Block.end());

  Inserter.insert(HostClone);
  Inserter.insert(DeviceClone);

  // Validate container invariants early.
  if (mlir::failed(Container.verify()))
    return mlir::failure();

  return Combined;
}

bool cir::emitCIRModuleToFile(mlir::ModuleOp Module, llvm::StringRef Output,
                              clang::DiagnosticsEngine &Diags) {
  mlir::OpPrintingFlags Flags;

  std::error_code EC;
  llvm::raw_fd_ostream OS(Output, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    Diags.Report(clang::diag::err_fe_error_opening) << Output << EC.message();
    return false;
  }

  Module.print(OS, Flags);
  return true;
}

bool cir::emitCIRModuleToStream(mlir::ModuleOp Module, llvm::raw_ostream &OS,
                                clang::DiagnosticsEngine &Diags) {
  mlir::OpPrintingFlags Flags;

  Module.print(OS, Flags);
  return true;
}
