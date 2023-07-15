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

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferSymmetryOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

bool getSymmetryOrUnknownFromType(Value v) {
    Type t = v.getType();
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        return mt.getProperties()->symmetry;
    else // scalar or frame
        // TODO: read scalar value (if 0 -> sparsity 0.0)
        return false;
}

// ****************************************************************************
// Symmetry inference interface implementations
// ****************************************************************************

std::vector<bool> daphne::MatrixConstantOp::inferSymmetry() {
    return {true};
}

std::vector<bool> daphne::FillOp::inferSymmetry() {

    try {
        auto numRows = CompilerUtils::constantOrThrow<int64_t>(getNumRows());
        auto numCols = CompilerUtils::constantOrThrow<int64_t>(getNumCols());

        if (numRows == numCols) {
            return {true};
        }

        return {false};

    } catch(const std::runtime_error & e) {
            return {false};
    }
    
}

std::vector<bool> daphne::DiagMatrixOp::inferSymmetry() {
    return {true};
}


// ****************************************************************************
// Symmetry inference trait implementations
// ****************************************FTY************************************
template<size_t i>
struct trySymmetryFromIthScalar {
    static void apply(ssize_t &symmetry, Operation *op) {
        if(op->hasTrait<SymmetryFromIthScalar<i>::template Impl>())
            symmetry = CompilerUtils::constantOrDefault<ssize_t>(op->getOperand(i), -1);
    }
};

template<size_t i>
struct trySymmetryFromIthArg {
    static void apply(ssize_t &symmetry, Operation *op) {
        if(op->hasTrait<SymmetryFromIthArg<i>::template Impl>())
            symmetry = getSymmetryOrUnknownFromType(op->getOperand(i));
    }
};

// ****************************************************************************
// Symmetry inference function
// ****************************************************************************

std::vector<bool> daphne::tryInferSymmetry(Operation *op) {
    if(auto inferSymmetryOp = llvm::dyn_cast<daphne::InferSymmetry>(op))
        // If the operation implements the symmetry inference interface,
        // we apply that.

        return inferSymmetryOp.inferSymmetry();
    else if(op->getNumResults() == 1) {
    
        if(op->hasTrait<EwSymmetricIfBoth>()) {
            auto symLhs = getSymmetryOrUnknownFromType(op->getOperand(0));
            auto symRhs = getSymmetryOrUnknownFromType(op->getOperand(1));
            if(symLhs && (symRhs || op->getOperand(1).getType().dyn_cast<IntegerType>()))
                return {true};
        }

        if(op->hasTrait<SymmetricIfArg>()) {
            auto symArg = getSymmetryOrUnknownFromType(op->getOperand(0));
            if(symArg)
                return {true};
        }

        if(op->hasTrait<SymmetricRes>()) {
                return {true};
        }
        

        return {false};
    } else {
        // If the operation does not implement the symmetry inference interface
        // and has zero or more than one results, we return unknown.
        std::vector<bool> symmetries;
        for(size_t i = 0; i < op->getNumResults(); i++)
            symmetries.push_back(false);
        return symmetries;
    }
}
