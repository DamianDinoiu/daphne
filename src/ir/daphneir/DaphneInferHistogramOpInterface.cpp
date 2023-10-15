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
#include <ir/daphneir/DaphneInferHistogramOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

// ****************************************************************************
// Utilities
// ****************************************************************************

std::vector<int> getHistogramOrUnknownFromType(Value v) {
    Type t = v.getType();
    ssize_t propertiesPointer = -1;
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        propertiesPointer = mt.getProperties();
    else if (auto ft = t.dyn_cast<daphne::FrameType>())
        propertiesPointer = ft.getProperties();

    if (propertiesPointer != -1) {

        Properties* properties = reinterpret_cast<Properties*>(propertiesPointer);
        return properties->histograms;

    }
    
    return {};
}

std::vector<int> daphne::EwLtOp::inferHistogram() {

    if (auto ft = getLhs().getType().dyn_cast<daphne::MatrixType>()) {
        if (ft.getProperties() != -1) {
            Properties* properties = reinterpret_cast<Properties*>(ft.getProperties());

            auto histogram = properties->histograms;
            auto min = 1;
            auto max = 10;
            double step = ((double)max - (double)min) / 5.0;

            auto rhs = getRhs().getType();
            auto compareValue = CompilerUtils::constantOrDefault<int64_t>(getRhs(), 0);
            auto numberOfValues = 0;
            
            if (histogram.size() != 0)
                for(size_t i = 0; i < histogram.size(); i++) {

                    double upper = (double)min + ((double)i + 1.0) * step;

                    if (upper <= compareValue)
                        numberOfValues += histogram[i];
                }
            return {numberOfValues};
        }
    }
    return {};
}

// ****************************************************************************
// Symmetry inference function
// ****************************************************************************

std::vector<int> daphne::tryInferHistogram(Operation *op) {
    if(auto inferHistogramOp = llvm::dyn_cast<daphne::InferHistogram>(op))
        // If the operation implements the symmetry inference interface,
        // we apply that.

        return inferHistogramOp.inferHistogram();
    else if(op->getNumResults() == 1) {

        // Propagate the histogram without alteration.
        if(op->hasTrait<PropagateHistogram>()) {

            auto propL = getHistogramOrUnknownFromType(op->getOperand(0));

            if (propL.size() != 0)
                return propL;
        }
        return {};
    } else {
        // If the operation does not implement the symmetry inference interface
        // and has zero or more than one results, we return unknown.
        return {};
    }
}