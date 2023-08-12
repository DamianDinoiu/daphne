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

/**
 * @brief Insert the properties using the InsertTraitsOp so they can be accessed and exploit at runtime.
**/
struct InsertPropertiesPass : public PassWrapper<InsertPropertiesPass, OperationPass<func::FuncOp>>
{
    void runOnOperation() final;
};

void InsertPropertiesPass::runOnOperation()
{
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());

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

                    if (properties->symmetry || properties->minMax.size() != 0) {

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
        }
    });
}

std::unique_ptr<Pass> daphne::createInsertPropertiesPass()
{
    return std::make_unique<InsertPropertiesPass>();
}
