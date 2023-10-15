/*
 *  Copyright 2023 The DAPHNE Consortium
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERHISTOGRAMOPINTERFACE_H
#define SRC_IR_DAPHNEIR_DAPHNEINFERHISTOGRAMOPINTERFACE_H

#include <vector>
#include <utility>

// ****************************************************************************
// Sparsity inference traits
// ****************************************************************************

// All of these traits address operations with **exactly one result**.
// Supporting multiple results would complicate the traits unnecessarily, given
// the low number of DaphneIR operations with multiple results. Thus,
// operations with multiple results should simply implement the sparsity inference
// interface instead of using traits.

namespace mlir::OpTrait {

// ============================================================================
// Traits definitions
// ============================================================================

template<class ConcreteOp>
class EwHistogramIfBoth : public TraitBase<ConcreteOp, EwHistogramIfBoth> {};

/*
    Propagate the histogram without any alteration.
*/
template<class ConcreteOp>
class PropagateHistogram : public TraitBase<ConcreteOp, PropagateHistogram> {};

}

// ****************************************************************************
// Symmetry inference interfaces
// ****************************************************************************

namespace mlir::daphne {
#include <ir/daphneir/DaphneInferHistogramOpInterface.h.inc>
}

// ****************************************************************************
// Symmetry inference function
// ****************************************************************************

namespace mlir::daphne {
std::vector<int> tryInferHistogram(mlir::Operation* op);
}

#endif //SRC_IR_DAPHNEIR_DAPHNEINFERHISTOGRAMOPINTERFACE_H