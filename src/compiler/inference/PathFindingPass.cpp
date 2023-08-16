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
struct PathFindingPass : public PassWrapper<PathFindingPass, OperationPass<func::FuncOp>>
{
    void runOnOperation() final;

    void generatePath(mlir::Operation * op, std::string path) {

        if (op->template hasTrait<mlir::OpTrait::PropagateProperties>()) {

            auto operands = op->getOperands();

            for (int i = 0; i < operands.size(); i++) {

                auto operand = operands[i];
                auto operandType = operand.getType();

                if(auto mt = operandType.dyn_cast<daphne::MatrixType>()) {
                
                auto parentOp = operand.getDefiningOp();
                generatePath(parentOp, path + " " + op->getName().getStringRef().data());
                    
                }
            }
        }

        if (op->template hasTrait<mlir::OpTrait::GenerateProperties>()) {

            std::cout << path << " " << op->getName().getStringRef().data() << "\n";

        }

    }

};

void PathFindingPass::runOnOperation()
{
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());

    f.getBody().front().walk([&](Operation* op) {
        
        if(!llvm::isa<daphne::InsertTraitsOp>(op)) {
            std::vector<Value> newOperands;
            std::vector<Value> newResults;

            auto results = op->getResults();

            // If the current operation is a consumer check along if the next one 
            // propagates/generates.
            if (op->template hasTrait<mlir::OpTrait::ConsumeProperties>()) {

                auto operands = op->getOperands();

                for(int i = 0; i < operands.size(); i++) {

                    auto operand = operands[i];
                    auto operandType = operand.getType();

                    if(auto mt = operandType.dyn_cast<daphne::MatrixType>()) {

                        auto parentOp = operands[0].getDefiningOp();
                        generatePath(parentOp,  op->getName().getStringRef().data());
                    
                    }

                }


            }
        }

    });
}

std::unique_ptr<Pass> daphne::createPathFindingPass()
{
    return std::make_unique<PathFindingPass>();
}
