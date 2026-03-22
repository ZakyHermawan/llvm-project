//===--- CIRGenAction.cpp - LLVM Code generation Frontend Action ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/FrontendAction/CIRGenAction.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CIR/CIRGenerator.h"
#include "clang/CIR/CIRToCIRPasses.h"
#include "clang/CIR/FrontendAction/CIRCombineSplitUtils.h"
#include "clang/CIR/LowerToLLVM.h"
#include "clang/CodeGen/BackendUtil.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/IR/Module.h"
#include "llvm/Object/OffloadBinary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace cir;
using namespace clang;

namespace cir {

static BackendAction
getBackendActionFromOutputType(CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitCIR:
    assert(false &&
           "Unsupported output type for getBackendActionFromOutputType!");
    break; // Unreachable, but fall through to report that
  case CIRGenAction::OutputType::EmitAssembly:
    return BackendAction::Backend_EmitAssembly;
  case CIRGenAction::OutputType::EmitBC:
    return BackendAction::Backend_EmitBC;
  case CIRGenAction::OutputType::EmitLLVM:
    return BackendAction::Backend_EmitLL;
  case CIRGenAction::OutputType::EmitObj:
    return BackendAction::Backend_EmitObj;
  }
  // We should only get here if a non-enum value is passed in or we went through
  // the assert(false) case above
  llvm_unreachable("Unsupported output type!");
}

static std::unique_ptr<llvm::Module>
lowerFromCIRToLLVMIR(mlir::ModuleOp MLIRModule, llvm::LLVMContext &LLVMCtx) {
  return direct::lowerDirectlyFromCIRToLLVMIR(MLIRModule, LLVMCtx);
}

static bool emitEmbeddedOffloadPayload(raw_pwrite_stream &OS,
                                       CompilerInstance &CI,
                                       StringRef PayloadPath) {
  auto PayloadOrErr = CI.getVirtualFileSystem().getBufferForFile(
      PayloadPath, -1, false);
  if (std::error_code EC = PayloadOrErr.getError()) {
    CI.getDiagnostics().Report(diag::err_cannot_open_file)
        << PayloadPath << EC.message();
    return false;
  }

  StringRef Payload = PayloadOrErr.get()->getBuffer();
  OS << "\n// cir.offload_payload.begin\n";
  OS << "// path: " << PayloadPath << "\n";
  OS << "// size: " << Payload.size() << "\n";

  // Keep payload fallback binary-safe and format-agnostic.
  OS << "// format: hex\n";
  for (size_t I = 0, E = Payload.size(); I < E; I += 32) {
    size_t End = I + 32 < E ? I + 32 : E;
    OS << "// hex: " << llvm::toHex(Payload.slice(I, End), true) << "\n";
  }
  OS << "// cir.offload_payload.end\n";
  return true;
}

// Try to load device module from PayloadPath
static mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>>
tryLoadEmbeddedDeviceModule(CompilerInstance &CI, StringRef PayloadPath,
                            mlir::MLIRContext &MlirContext) {
  auto PayloadOrErr =
      CI.getVirtualFileSystem().getBufferForFile(PayloadPath, -1, false);
  if (std::error_code EC = PayloadOrErr.getError()) {
    CI.getDiagnostics().Report(diag::err_cannot_open_file)
        << PayloadPath << EC.message();
    return mlir::failure();
  }

  // Preferred path: decode standardized offload container entries and parse
  // their embedded image as CIR. This is language-agnostic (e.g. CUDA/HIP).
  if (auto BinariesOrErr = llvm::object::OffloadBinary::create(
          PayloadOrErr.get()->getMemBufferRef())) {
    llvm::object::OffloadKind ExpectedKind = llvm::object::OFK_None;
    if (CI.getLangOpts().HIP)
      ExpectedKind = llvm::object::OFK_HIP;
    else if (CI.getLangOpts().CUDA)
      ExpectedKind = llvm::object::OFK_Cuda;

    StringRef ExpectedTriple = CI.getTargetOpts().Triple;
    StringRef ExpectedHostTriple = CI.getTargetOpts().HostTriple;
    StringRef ExpectedArch = CI.getTargetOpts().CPU;

    auto ParseImageFromEntry = [&](const llvm::object::OffloadBinary &Binary)
        -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      StringRef Image = Binary.getImage();
      if (Image.empty())
        return mlir::failure();

      auto ImageBuffer = llvm::MemoryBuffer::getMemBufferCopy(
          Image, "embedded-device-image.cir");
      return parseCIRModuleFromBuffer(ImageBuffer->getMemBufferRef(),
                                      MlirContext);
    };

    auto KindMatches = [&](const llvm::object::OffloadBinary &Binary) {
      return ExpectedKind == llvm::object::OFK_None ||
             Binary.getOffloadKind() == ExpectedKind;
    };

    auto TripleMatches = [&](const llvm::object::OffloadBinary &Binary) {
      StringRef Triple = Binary.getTriple();
      if (Triple.empty())
        return false;
      return (!ExpectedTriple.empty() && Triple == ExpectedTriple) ||
             (!ExpectedHostTriple.empty() && Triple == ExpectedHostTriple);
    };

    auto ArchMatches = [&](const llvm::object::OffloadBinary &Binary) {
      StringRef Arch = Binary.getArch();
      return !ExpectedArch.empty() && !Arch.empty() && Arch == ExpectedArch;
    };

    auto TryEntries = [&](auto Predicate)
        -> mlir::FailureOr<mlir::OwningOpRef<mlir::ModuleOp>> {
      for (const std::unique_ptr<llvm::object::OffloadBinary> &Binary :
           *BinariesOrErr) {
        if (!Predicate(*Binary))
          continue;
        if (auto ImageOr = ParseImageFromEntry(*Binary);
            mlir::succeeded(ImageOr))
          return ImageOr;
      }
      return mlir::failure();
    };

    // Prefer exact metadata-guided matches first, then broaden as fallback.
    // Because payload metadata is often incomplete or inconsistent across toolchains,
    // so the loader uses a staged match strategy instead of one strict check.
    if (auto ImageOr = TryEntries([&](const llvm::object::OffloadBinary &B) {
          return KindMatches(B) && (TripleMatches(B) || ArchMatches(B));
        });
        mlir::succeeded(ImageOr))
      return ImageOr;

    if (auto ImageOr =
            TryEntries([&](const llvm::object::OffloadBinary &B) {
              return KindMatches(B);
            });
        mlir::succeeded(ImageOr))
      return ImageOr;

    if (auto ImageOr = TryEntries([&](const llvm::object::OffloadBinary &B) {
          return TripleMatches(B) || ArchMatches(B);
        });
        mlir::succeeded(ImageOr))
      return ImageOr;

    if (auto ImageOr = TryEntries([&](const llvm::object::OffloadBinary &) {
          return true;
        });
        mlir::succeeded(ImageOr))
      return ImageOr;
  } else {
    llvm::consumeError(BinariesOrErr.takeError());
  }

  return mlir::failure();
}

class CIRGenConsumer : public clang::ASTConsumer {

  virtual void anchor();

  CIRGenAction::OutputType Action;

  CompilerInstance &CI;

  std::unique_ptr<raw_pwrite_stream> OutputStream;

  ASTContext *Context{nullptr};
  IntrusiveRefCntPtr<llvm::vfs::FileSystem> FS;
  std::unique_ptr<CIRGenerator> Gen;
  const FrontendOptions &FEOptions;
  CodeGenOptions &CGO;

public:
  CIRGenConsumer(CIRGenAction::OutputType Action, CompilerInstance &CI,
                 CodeGenOptions &CGO, std::unique_ptr<raw_pwrite_stream> OS)
      : Action(Action), CI(CI), OutputStream(std::move(OS)),
        FS(&CI.getVirtualFileSystem()),
        Gen(std::make_unique<CIRGenerator>(CI.getDiagnostics(), std::move(FS),
                                           CI.getCodeGenOpts())),
        FEOptions(CI.getFrontendOpts()), CGO(CGO) {}

  void Initialize(ASTContext &Ctx) override {
    assert(!Context && "initialized multiple times");
    Context = &Ctx;
    Gen->Initialize(Ctx);
  }

  bool HandleTopLevelDecl(DeclGroupRef D) override {
    Gen->HandleTopLevelDecl(D);
    return true;
  }

  void HandleCXXStaticMemberVarInstantiation(clang::VarDecl *VD) override {
    Gen->HandleCXXStaticMemberVarInstantiation(VD);
  }

  void HandleOpenACCRoutineReference(const FunctionDecl *FD,
                                     const OpenACCRoutineDecl *RD) override {
    Gen->HandleOpenACCRoutineReference(FD, RD);
  }

  void HandleInlineFunctionDefinition(FunctionDecl *D) override {
    Gen->HandleInlineFunctionDefinition(D);
  }

  void HandleTranslationUnit(ASTContext &C) override {
    Gen->HandleTranslationUnit(C);

    if (!FEOptions.ClangIRDisableCIRVerifier) {
      if (!Gen->verifyModule()) {
        CI.getDiagnostics().Report(
            diag::err_cir_verification_failed_pre_passes);
        llvm::report_fatal_error(
            "CIR codegen: module verification error before running CIR passes");
        return;
      }
    }

    mlir::ModuleOp MlirModule = Gen->getModule();
    mlir::MLIRContext &MlirCtx = Gen->getMLIRContext();

    if (!FEOptions.ClangIRDisablePasses) {
      // Setup and run CIR pipeline.
      if (runCIRToCIRPasses(
              MlirModule, MlirCtx, C, !FEOptions.ClangIRDisableCIRVerifier,
              FEOptions.ClangIREnableIdiomRecognizer, CGO.OptimizationLevel > 0)
              .failed()) {
        CI.getDiagnostics().Report(diag::err_cir_to_cir_transform_failed);
        return;
      }
    }

    switch (Action) {
    case CIRGenAction::OutputType::EmitCIR:
      if (OutputStream && MlirModule) {
        mlir::OpPrintingFlags Flags;
        Flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/false);

        auto TryCombineFromPayload = [&](StringRef PayloadPath) {
          auto DeviceModuleOr =
              tryLoadEmbeddedDeviceModule(CI, PayloadPath, MlirCtx);
          if (mlir::failed(DeviceModuleOr))
            return false;
          auto CombinedOr =
              buildCombinedOffloadModule(MlirModule, **DeviceModuleOr, MlirCtx);
          if (mlir::failed(CombinedOr))
            return false;
          (**CombinedOr).print(*OutputStream, Flags);
          return true;
        };

        bool PrintedCombined = false;

        // Generic offload container inputs (e.g. RDC/multi-image flows).
        for (StringRef OffloadObjectPath : CGO.OffloadObjects) {
          if (!TryCombineFromPayload(OffloadObjectPath))
            continue;
          PrintedCombined = true;
          break;
        }

        // Legacy single-path flag used by CUDA/HIP host jobs.
        if (!PrintedCombined && !CGO.CudaGpuBinaryFileName.empty()) {
          PrintedCombined = TryCombineFromPayload(CGO.CudaGpuBinaryFileName);
        }

        if (PrintedCombined)
          break;

        if (!CGO.CudaGpuBinaryFileName.empty() || !CGO.OffloadObjects.empty()) {
          // Fallback for opaque/non-parseable payloads.
          // In this case, just print host and device module
          MlirModule->print(*OutputStream, Flags);

          StringRef PayloadPath = CGO.OffloadObjects.empty()
                                      ? StringRef(CGO.CudaGpuBinaryFileName)
                                      : StringRef(CGO.OffloadObjects.front());

          if (!emitEmbeddedOffloadPayload(*OutputStream, CI, PayloadPath))
            return;
          break;
        }

        MlirModule->print(*OutputStream, Flags);
      }
      break;
    case CIRGenAction::OutputType::EmitLLVM:
    case CIRGenAction::OutputType::EmitBC:
    case CIRGenAction::OutputType::EmitObj:
    case CIRGenAction::OutputType::EmitAssembly: {
      llvm::LLVMContext LLVMCtx;
      std::unique_ptr<llvm::Module> LLVMModule =
          lowerFromCIRToLLVMIR(MlirModule, LLVMCtx);

      BackendAction BEAction = getBackendActionFromOutputType(Action);
      emitBackendOutput(
          CI, CI.getCodeGenOpts(), C.getTargetInfo().getDataLayoutString(),
          LLVMModule.get(), BEAction, FS, std::move(OutputStream));
      break;
    }
    }
  }

  void HandleTagDeclDefinition(TagDecl *D) override {
    PrettyStackTraceDecl CrashInfo(D, SourceLocation(),
                                   Context->getSourceManager(),
                                   "CIR generation of declaration");
    Gen->HandleTagDeclDefinition(D);
  }

  void HandleTagDeclRequiredDefinition(const TagDecl *D) override {
    Gen->HandleTagDeclRequiredDefinition(D);
  }

  void CompleteTentativeDefinition(VarDecl *D) override {
    Gen->CompleteTentativeDefinition(D);
  }

  void HandleVTable(CXXRecordDecl *RD) override { Gen->HandleVTable(RD); }
};
} // namespace cir

void CIRGenConsumer::anchor() {}

CIRGenAction::CIRGenAction(OutputType Act, mlir::MLIRContext *MLIRCtx)
    : MLIRCtx(MLIRCtx ? MLIRCtx : new mlir::MLIRContext), Action(Act) {}

CIRGenAction::~CIRGenAction() { MLIRMod.release(); }

static std::unique_ptr<raw_pwrite_stream>
getOutputStream(CompilerInstance &CI, StringRef InFile,
                CIRGenAction::OutputType Action) {
  switch (Action) {
  case CIRGenAction::OutputType::EmitAssembly:
    return CI.createDefaultOutputFile(false, InFile, "s");
  case CIRGenAction::OutputType::EmitCIR:
    return CI.createDefaultOutputFile(false, InFile, "cir");
  case CIRGenAction::OutputType::EmitLLVM:
    return CI.createDefaultOutputFile(false, InFile, "ll");
  case CIRGenAction::OutputType::EmitBC:
    return CI.createDefaultOutputFile(true, InFile, "bc");
  case CIRGenAction::OutputType::EmitObj:
    return CI.createDefaultOutputFile(true, InFile, "o");
  }
  llvm_unreachable("Invalid CIRGenAction::OutputType");
}

std::unique_ptr<ASTConsumer>
CIRGenAction::CreateASTConsumer(CompilerInstance &CI, StringRef InFile) {
  std::unique_ptr<llvm::raw_pwrite_stream> Out = CI.takeOutputStream();

  if (!Out)
    Out = getOutputStream(CI, InFile, Action);

  auto Result = std::make_unique<cir::CIRGenConsumer>(
      Action, CI, CI.getCodeGenOpts(), std::move(Out));

  return Result;
}

void EmitAssemblyAction::anchor() {}
EmitAssemblyAction::EmitAssemblyAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitAssembly, MLIRCtx) {}

void EmitCIRAction::anchor() {}
EmitCIRAction::EmitCIRAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitCIR, MLIRCtx) {}

void EmitLLVMAction::anchor() {}
EmitLLVMAction::EmitLLVMAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitLLVM, MLIRCtx) {}

void EmitBCAction::anchor() {}
EmitBCAction::EmitBCAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitBC, MLIRCtx) {}

void EmitObjAction::anchor() {}
EmitObjAction::EmitObjAction(mlir::MLIRContext *MLIRCtx)
    : CIRGenAction(OutputType::EmitObj, MLIRCtx) {}
