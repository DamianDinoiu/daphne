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

#ifndef SRC_IR_DAPHNEIR_DAPHNEINFERSYMMETRYTRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNEINFERSYMMETRYTRAITS_TD

include "mlir/IR/OpBase.td"


// ****************************************************************************
// Traits determining symmetry for elementwise ops
// ****************************************************************************

// if both elements are true, the result is true
def EwSymmetricIfBoth : NativeOpTrait<"EwSymmetricIfBoth">;
// if either element is true, the result is true
def SymmetricIfArg : NativeOpTrait<"SymmetricIfArg">;
// the result of the oprations will always be symmetric
def SymmetricRes : NativeOpTrait<"SymmetricRes">;


#endif //SRC_IR_DAPHNEIR_DAPHNEINFERSYMMETRYTRAITS_TD