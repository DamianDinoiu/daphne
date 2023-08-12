/*
 * Copyright 2023 The DAPHNE Consortium
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

#include <mlir/IR/Value.h>

#include <parser/metadata/MetaDataParser.h>

#include <vector>
#include <stdexcept>
#include <utility>
#include <iostream>

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferMinMaxOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

double getMinMaxOrUnknownFromType(Value v) {
    Type t = v.getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return -1;
    else // scalar or frame
        // TODO: read scalar value (if 0 -> sparsity 0.0)
        return -1;
}

// ****************************************************************************
// MIN/MAX inference interface implementations
// ****************************************************************************
std::vector<double> daphne::FillOp::inferMinMax() {

    /*
        Get the number of rows and cols and the value.
    */
    try {
        auto numRows = CompilerUtils::constantOrThrow<int64_t>(getNumRows());
        auto numCols = CompilerUtils::constantOrThrow<int64_t>(getNumCols());
        auto value = CompilerUtils::constantOrThrow<int64_t>(getArg());
       
        return std::vector<double>(numCols, value);

    } catch(const std::runtime_error & e) {
            return {-1.0};
    }

}

std::vector<double> daphne::SliceColOp::inferMinMax() {

    Type srcTy = getSource().getType();
    
    auto srcMatTy = srcTy.dyn_cast<daphne::MatrixType>();
    auto minMax = srcMatTy.getProperties()->minMax;

    /*
        Get the number of rows and cols and the value.
    */
    auto loIn = CompilerUtils::isConstant<int64_t>(getLowerIncl());
    auto upEx = CompilerUtils::isConstant<int64_t>(getUpperExcl());

    std::vector<double> newMinMax;

    std::cout << "MinMax = " << minMax.size() << "\n";
    std::cout << "Inerval = " << (upEx.second - loIn.second) << "\n";
 
    if (minMax.size() != 0) {
    
        for (int i = loIn.second; i < upEx.second; i++) {
            newMinMax.push_back(minMax[i]);
        }

        return newMinMax;
    }

    return {};

}

std::vector<double> daphne::ExtractColOp::inferMinMax() {

    Type srcTy = getSource().getType();

    auto list = getSelectedCols().getType();

    auto test = llvm::dyn_cast<daphne::MatrixType>(list);

    if (test)
        std::cout << "WTTFFFFFF\n";

    return {};

}



// ****************************************************************************
// Symmetry inference trait implementations
// ****************************************FTY************************************
template<size_t i>
struct tryMinMaxFromIthScalar {
    static void apply(ssize_t &symmetry, Operation *op) {
        if(op->hasTrait<SymmetryFromIthScalar<i>::template Impl>())
            symmetry = CompilerUtils::constantOrDefault<ssize_t>(op->getOperand(i), -1);
    }
};

template<size_t i>
struct tryMinMaxFromIthArg {
    static void apply(ssize_t &symmetry, Operation *op) {
        if(op->hasTrait<SymmetryFromIthArg<i>::template Impl>())
            symmetry = getMinMaxOrUnknownFromType(op->getOperand(i));
    }
};

// ****************************************************************************
// Symmetry inference function
// ****************************************************************************

std::vector<double> daphne::tryInferMinMax(Operation *op) {
    if(auto inferMinMaxOp = llvm::dyn_cast<daphne::InferMinMax>(op))
        // If the operation implements the symmetry inference interface,
        // we apply that.

        return inferMinMaxOp.inferMinMax();
    else if(op->getNumResults() == 1) {
        return {};
    } else {
        // If the operation does not implement the symmetry inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<bool> symmetries;
        for(size_t i = 0; i < op->getNumResults(); i++)
            symmetries.push_back(false);
        return {};
    }
}
