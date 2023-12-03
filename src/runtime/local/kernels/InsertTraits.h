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

#ifndef SRC_RUNTIME_LOCAL_KERNELS_EWUNARYTEST_H
#define SRC_RUNTIME_LOCAL_KERNELS_EWUNARYTEST_H

#include <memory>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/kernels/UnaryOpCode.h>
#include <runtime/local/kernels/EwUnarySca.h>
#include <runtime/local/datastructures/Traits.h>
#include <runtime/local/datastructures/Properties.h>


#include <cassert>
#include <cstddef>
#include <cmath>
#include <iostream>



// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template<class DTRes, class DTArg>
struct InsertTraits {
    static void apply(DTRes *& res, DTArg * arg, int64_t * test, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template<class DTRes, class DTArg>
void insertTraits(DTRes *& res, DTArg * arg, int64_t * test ,DCTX(ctx)) {
    InsertTraits<DTRes, DTArg>::apply(res, arg, test, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// DenseMatrix <- DenseMatrix
// ----------------------------------------------------------------------------

template<typename VT>
struct InsertTraits<DenseMatrix<VT>, DenseMatrix<VT>> {
    static void apply(DenseMatrix<VT> *& res, DenseMatrix<VT> * arg, int64_t * test, DCTX(ctx)) {
        
        auto properties = reinterpret_cast<Properties*>(test);
        auto minMax = properties->minMax;
        auto sym = properties->symmetry;

        res = const_cast<DenseMatrix<VT> *>(arg);
        res->setProperties(properties);
        res->increaseRefCounter();

       
    }
};

#endif //SRC_RUNTIME_LOCAL_KERNELS_EWUNARYTEST_H