// #include <ir/daphneir/Daphne.h>
// #include <ir/daphneir/Passes.h>

// #include <mlir/Dialect/SCF/IR/SCF.h>
// #include <mlir/IR/Operation.h>
// #include <mlir/Pass/Pass.h>

// #include <stdexcept>
// #include <memory>
// #include <vector>
// #include <utility>
// #include <iostream>

// using namespace mlir;


// /// An interesting analysis.
// struct MyOperationAnalysis {
//     bool symmetry;
//   // Compute this analysis with the provided operation.
//   MyOperationAnalysis(Operation *op) {
//     std::cout << "pass\n";
//     symmetry = true;
//   }
// };


// struct PropertiesAnalysisPass : public PassWrapper<PropertiesAnalysisPass, OperationPass<mlir::daphen::FillOp>>
// {
//     void runOnOperation() final;
// };

// void PropertiesAnalysisPass::runOnOperation() {
//   // Query MyOperationAnalysis for the current operation.

//     std::cout << "Ceva nu merge bine!\n";

//     MyOperationAnalysis &myAnalysis = getAnalysis<MyOperationAnalysis>();

//   // Query a cached instance of MyOperationAnalysis for the parent operation of
//   // the current operation. It will not be computed if it doesn't exist.
// //   auto optionalAnalysis = getCachedParentAnalysis<MyOperationAnalysis>();
// //   if (optionalAnalysis)
// }

// std::unique_ptr<Pass> daphne::createPropertiesAnalysisPass()
// {
//     return std::make_unique<PropertiesAnalysisPass>();
// }