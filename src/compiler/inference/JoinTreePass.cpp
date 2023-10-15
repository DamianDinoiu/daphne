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
#include <initializer_list>
#include <ir/daphneir/Daphne.h>
#include <ir/daphneir/Passes.h>


#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/IRMapping.h"

#include <mlir/Pass/Pass.h>

#include <iostream>
#include <utility>

using namespace mlir;

/**
 * @brief Insert the properties using the InsertTraitsOp so they can be accessed and exploit at runtime.
**/
struct JoinTreePass : public PassWrapper<JoinTreePass, OperationPass<func::FuncOp>>
{
    void runOnOperation() final;

    void generateJoinTree(mlir::Operation * operation, std::vector<mlir::Operation *> *tree) {

        auto leftOp = operation->getOperands()[0].getDefiningOp();
        auto rightOp = operation->getOperands()[1].getDefiningOp();
 
        tree->push_back(operation);

        auto rightOpResult = rightOp->getResults()[0].getType();

        if (auto ftRight = rightOpResult.dyn_cast<mlir::daphne::FrameType>()) {

            if (llvm::isa<daphne::InnerJoinOp>(leftOp)) {
                generateJoinTree(leftOp, tree);
            }
        }

    }

    static bool sortBySize(const std::pair<mlir::Value, mlir::Value> &left, const std::pair<mlir::Value, mlir::Value> &right){

        auto rFrame = left.first.getType().dyn_cast<mlir::daphne::FrameType>();
        auto lFrame = right.first.getType().dyn_cast<mlir::daphne::FrameType>();

        return (lFrame.getNumRows() > rFrame.getNumRows());

    }

};

void JoinTreePass::runOnOperation()
{
    func::FuncOp f = getOperation();
    OpBuilder builder(f.getContext());
    std::vector<mlir::Operation *> joinTree;


    f.getBody().front().walk([&](Operation* op) {

        
        if(llvm::isa<daphne::InnerJoinOp>(op)) {
            std::vector<mlir::Operation *> subTrees;
            generateJoinTree(op, &subTrees);
            joinTree = subTrees;

        }
    });


    if (joinTree.size() == 2) {
    std::vector<std::pair<mlir::Value, mlir::Value>> leafs;
    std::vector<std::pair<mlir::Value, mlir::Value>> sortedLeafs;


    for(size_t i = 0; i < joinTree.size(); i++) {

        leafs.push_back(std::make_pair(joinTree[i]->getOperand(0), joinTree[i]->getOperand(2)));
        leafs.push_back(std::make_pair(joinTree[i]->getOperand(1), joinTree[i]->getOperand(3)));

    }

    // std::sort(leafs.begin(), leafs.end(), sortBySize);
    auto rFrame = leafs[1].first.getType().dyn_cast<mlir::daphne::FrameType>();
    auto lFrame = leafs[2].first.getType().dyn_cast<mlir::daphne::FrameType>();


    std::vector<mlir::Value> op1;
    std::vector<mlir::Value> op2;
    if (lFrame.getNumRows() > rFrame.getNumRows()) {
        op1.push_back(leafs[1].first);
        op1.push_back(leafs[3].first);
        op1.push_back(leafs[1].second);
        op1.push_back(leafs[0].second);

        joinTree[1]->setOperands(mlir::ValueRange(op1));

        op2.push_back(joinTree[joinTree.size()-1]->getResult(0));
        op2.push_back(leafs[2].first);
        op2.push_back(leafs[3].second);
        op2.push_back(leafs[2].second);

        joinTree[0]->setOperands(mlir::ValueRange(op2));

        // return;
        // sortedLeafs.push_back(leafs[2]);
        // sortedLeafs.push_back(leafs[0]);
        // sortedLeafs.push_back(leafs[1]);
    } else {
        // sortedLeafs.push_back(leafs[2]);
        // sortedLeafs.push_back(leafs[1]);
        // sortedLeafs.push_back(leafs[0]);
        return;
    }


    // std::vector<mlir::Value> op1;
    // int currentLeaf = 2;
    // op1.push_back(sortedLeafs[0].first);
    // op1.push_back(sortedLeafs[1].first);
    // op1.push_back(sortedLeafs[0].second);
    // op1.push_back(sortedLeafs[1].second);
    
    // joinTree[1]->setOperands(mlir::ValueRange(op1));

    // for (int i = joinTree.size() - 2; i >= 0; i--) {

    //     std::vector<mlir::Value> op2;

    //     op2.push_back(joinTree[joinTree.size()-1]->getResult(0));
    //     op2.push_back(sortedLeafs[currentLeaf].first);
    //     op2.push_back(sortedLeafs[1].second);
    //     op2.push_back(sortedLeafs[currentLeaf].second);

    //     joinTree[i]->setOperands(mlir::ValueRange(op2));
    //     currentLeaf++;

    // }
    }
    
}

std::unique_ptr<Pass> daphne::createJoinTreePass()
{
    return std::make_unique<JoinTreePass>();
}