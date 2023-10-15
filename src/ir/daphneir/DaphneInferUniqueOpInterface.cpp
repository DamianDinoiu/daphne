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
#include <ir/daphneir/DaphneInferUniqueOpInterface.cpp.inc>
}

using namespace mlir;
using namespace mlir::OpTrait;

bool getUniqueOrUnknownFromType(Value v) {
    Type t = v.getType();
    ssize_t propertiesPointer = -1;
    if(auto mt = t.dyn_cast<daphne::MatrixType>())
        propertiesPointer = mt.getProperties();
    else if (auto ft = t.dyn_cast<daphne::FrameType>())
        propertiesPointer = ft.getProperties();

    if (propertiesPointer != -1) {

        Properties* properties = reinterpret_cast<Properties*>(propertiesPointer);
        return properties->unique;

    }
    return {};
}

bool daphne::SeqOp::inferUnique() {
    return true;
}

bool daphne::tryInferUnique(Operation *op) {
    if(auto inferUniqueOp = llvm::dyn_cast<daphne::InferUnique>(op))
        // If the operation implements the symmetry inference interface,
        // we apply that.

        return inferUniqueOp.inferUnique();
    else if(op->getNumResults() == 1) {
        if(op->hasTrait<UniqueIfArg>()) {
            return getUniqueOrUnknownFromType(op->getOperand(0));
        }
        return false;
    } else {
        return false;
    }
}