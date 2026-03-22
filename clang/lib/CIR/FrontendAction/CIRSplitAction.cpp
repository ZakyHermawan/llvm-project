#include "clang/CIR/FrontendAction/CIRSplitAction.h"

#include "clang/Basic/DiagnosticDriver.h"
#include "clang/CIR/FrontendAction/CIRCombineSplitUtils.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"

using namespace cir;
using namespace clang;

CIRSplitAction::CIRSplitAction() : MlirContext(new mlir::MLIRContext) {}

void CIRSplitAction::ExecuteAction() {
  auto &Ci = getCompilerInstance();
  auto &Diags = Ci.getDiagnostics();
  const clang::FrontendOptions &Fo = Ci.getFrontendOpts();

  if (Fo.ClangIRSplitInput.empty()) {
    Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-cir-input";
    return;
  }

  bool MissingOutput = false;
  if (Fo.CIRHostOutput.empty()) {
    Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-cir-host-output";
    MissingOutput = true;
  }
  if (Fo.CIRDeviceOutput.empty()) {
    Diags.Report(clang::diag::err_drv_missing_arg_mtp) << "-cir-device-output";
    MissingOutput = true;
  }
  if (MissingOutput)
    return;

  loadCIRCombineSplitDialects(*MlirContext);

  auto CombinedModuleOr =
      parseCIRModuleFromFile(Ci, Fo.ClangIRSplitInput, *MlirContext);
  if (mlir::failed(CombinedModuleOr))
    return;

  mlir::OwningOpRef<mlir::ModuleOp> CombinedModule =
      std::move(*CombinedModuleOr);
  auto SplitModulesOr =
      getHostDeviceModules(*CombinedModule, Diags, "-cir-split",
                           "missing cir.offload.container in input CIR");
  if (mlir::failed(SplitModulesOr))
    return;
  mlir::ModuleOp HostModule = (*SplitModulesOr).first;
  mlir::ModuleOp DeviceModule = (*SplitModulesOr).second;

  if (!emitCIRModuleToFile(DeviceModule, Fo.CIRDeviceOutput, Diags))
    return;

  if (!emitCIRModuleToFile(HostModule, Fo.CIRHostOutput, Diags))
    return;
}
