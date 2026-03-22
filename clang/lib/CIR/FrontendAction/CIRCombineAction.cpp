#include "clang/CIR/FrontendAction/CIRCombineAction.h"

#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/CIR/FrontendAction/CIRCombineSplitUtils.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

using namespace cir;
using namespace clang;

CIRCombineAction::CIRCombineAction() : MlirContext(new mlir::MLIRContext) {}

void CIRCombineAction::ExecuteAction() {
  auto &Ci = getCompilerInstance();
  auto &Diags = Ci.getDiagnostics();
  const clang::FrontendOptions &Fo = Ci.getFrontendOpts();

  // Expect ParseFrontendArgs already validated these, but keep it defensive.
  if (Fo.ClangIRHostInput.empty() || Fo.ClangIRDeviceInput.empty()) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "-cir-combine" << "missing -cir-host-input/-cir-device-input";
    return;
  }

  if (Fo.OutputFile.empty()) {
    Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-o";
    return;
  }

  loadCIRCombineSplitDialects(*MlirContext);

  auto HostCirModuleOr =
      parseCIRModuleFromFile(Ci, Fo.ClangIRHostInput, *MlirContext);
  if (mlir::failed(HostCirModuleOr))
    return;
  mlir::OwningOpRef<mlir::ModuleOp> HostCirModule = std::move(*HostCirModuleOr);

  auto DeviceCirModuleOr =
      parseCIRModuleFromFile(Ci, Fo.ClangIRDeviceInput, *MlirContext);
  if (mlir::failed(DeviceCirModuleOr))
    return;
  mlir::OwningOpRef<mlir::ModuleOp> DeviceCirModule =
      std::move(*DeviceCirModuleOr);

  auto CombinedOr =
      buildCombinedOffloadModule(*HostCirModule, *DeviceCirModule, *MlirContext);
  if (mlir::failed(CombinedOr)) {
    Diags.Report(clang::diag::err_fe_error_reading)
        << "-cir-combine" << "failed to combine host/device CIR modules";
    return;
  }

  mlir::OwningOpRef<mlir::ModuleOp> Combined = std::move(*CombinedOr);
  auto SplitModulesOr =
      getHostDeviceModules(*Combined, Diags, "-cir-combine",
                           "missing cir.offload.container in combined CIR");
  if (mlir::failed(SplitModulesOr))
    return;

  if (!emitCIRModuleToFile(*Combined, Fo.OutputFile, Diags))
    return;
}
