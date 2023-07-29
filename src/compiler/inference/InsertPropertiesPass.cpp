/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <compiler/utils/CompilerUtils.h>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/IRMapping.h"

#include <mlir/Pass/Pass.h>

#include <iostream>

using namespace mlir;

namespace {

        class PropertiesInsertion : public RewritePattern {


            public:


                 PropertiesInsertion(MLIRContext * mctx, PatternBenefit benefit = 1)
                    : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, mctx)
                    {
                    }

        LogicalResult matchAndRewrite(Operation *op,
                        PatternRewriter &rewriter) const override
        {

            std::cout << "ceva";
            std::vector<Value> newResults;


            auto results = op->getResults();

            for (ssize_t i = 0; i < op->getNumResults(); i++) {

                auto result = op->getResult(i);
                auto resultType = result.getType();

                if (resultType.isa<mlir::daphne::MatrixType>()) {

                    auto mt = resultType.dyn_cast<daphne::MatrixType>();
                    auto properties = mt.getProperties();

                    if (properties->symmetry) {

                        rewriter.setInsertionPointAfter(op);
                        auto propertiesPointer = &properties;
                        auto constantOp = rewriter.create<mlir::daphne::ConstantOp>(op->getLoc(), reinterpret_cast<ssize_t>(properties.get()));
                        auto pointerValue = static_cast<mlir::Value>(constantOp);

                        auto newOp = rewriter.create<mlir::daphne::InsertTraitsOp>(
                            op->getLoc(),
                            resultType,
                            result,
                            pointerValue
                        );

                        newResults.push_back(newOp.getRes());

                    }
                }
            }
            return success();
        }
    };
}

/**
 * @brief Adapts an operation's input/output types such that it can be lowered to an available pre-compiled kernel.
 * 
 * While type inference propagates types through the IR, it is not guaranteed that a pre-compiled kernel
 * for each infered type combination is available. Thus, the task of this pass is to adapt input and
 * output types by casts, where necessary, to ensure that an existing pre-compiled kernel can be used.
 * 
 * At the moment, this pass is implemented in a very simple way. It merely harmonizes the value types
 * of all inputs with those of the single output of certain operations. This is because so far we mainly
 * pre-compile our kernels for homogeneous combinations of input/output types. The operations are
 * marked by traits.
 * 
 * In the future, this pass should take the kernel registry and/or extension catalog into account to find
 * out for which type combinations there are available kernels.
 */
// TODO This is not always correct for idxMin() and idxMax(): while their output always has an integer value
// type, it is not always safe to cast their input to integers.
struct InsertPropertiesPass : public PassWrapper<InsertPropertiesPass, OperationPass<func::FuncOp>>
{
    void runOnOperation() final;
};

void InsertPropertiesPass::runOnOperation()
{
       
       
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());


    // Daphne - legal
    // 
    // RewritePatternSet patterns(&getContext());
    // ConversionTarget target(getContext());
    // target.addLegalDialect<arith::ArithDialect, LLVM::LLVMDialect, scf::SCFDialect, daphne::DaphneDialect>();

    // patterns.insert<PropertiesInsertion>(&getContext());
    // if (failed(applyPartialConversion(module, target, std::move(patterns))))
    //     signalPassFailure();

    std::cout << "ceva2";
    f.getBody().front().walk([&](Operation* op) {
        
        if(!llvm::isa<daphne::InsertTraitsOp>(op)) {
            std::vector<Value> newOperands;
            std::vector<Value> newResults;

            auto results = op->getResults();

            for (ssize_t i = 0; i < op->getNumResults(); i++) {

                auto result = op->getResult(i);
                auto resultType = result.getType();

                if (resultType.isa<mlir::daphne::MatrixType>()) {

                    auto mt = resultType.dyn_cast<daphne::MatrixType>();
                    auto properties = mt.getProperties();

                    if (properties->symmetry) {

                        builder.setInsertionPointAfter(op);
                        auto propertiesPointer = &properties;
                        auto constantOp = builder.create<mlir::daphne::ConstantOp>(op->getLoc(), reinterpret_cast<ssize_t>(properties.get()));
                        auto pointerValue = static_cast<mlir::Value>(constantOp);

                        auto newOp = builder.create<mlir::daphne::InsertTraitsOp>(
                            op->getLoc(),
                            resultType,
                            result,
                            pointerValue
                        );

                        // .push_banewResultsck(newOp.getRes());
                        result.replaceAllUsesExcept(newOp.getRes(), {newOp});
                    }
                }
            }
            // auto oldValue = static_cast<mlir::Value>(op);
            // auto 

            // if (op->getResults())

            // for(ssize_t i = 0; i < op->getNumOperands (); i++) {

            //     auto operand = op->getOperand(i);
            //     auto operandType = operand.getType();

            //     if (operandType.isa<mlir::daphne::MatrixType>()) {
            //         auto mt =operandType.dyn_cast<daphne::MatrixType>();

            //         auto properties = mt.getProperties();

            //         if(properties->symmetry) {
            //             builder.setInsertionPointAfter(op);
            //             auto propertiesPointer = &properties;
            //             auto constantOp = builder.create<mlir::daphne::ConstantOp>(op->getLoc(), reinterpret_cast<ssize_t>(properties.get()));
            //             auto pointerValue = static_cast<mlir::Value>(constantOp);

            //             auto newOp = builder.create<mlir::daphne::InsertTraitsOp>(
            //                 op->getLoc(),
            //                 operandType,
            //                 operand,
            //                 pointerValue
            //             );

            //             newOperands.push_back(newOp.getRes());

            //         } 
            //     } else {
            //             newOperands.push_back(operand);
            //     }
            // }
            // op->setOperands(newOperands);
        }
    });
}

std::unique_ptr<Pass> daphne::createInsertPropertiesPass()
{
    return std::make_unique<InsertPropertiesPass>();
}
