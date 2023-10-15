[1mdiff --git a/src/ir/daphneir/DaphneDialect.cpp b/src/ir/daphneir/DaphneDialect.cpp[m
[1mindex 15ce1c06..7e784fd1 100644[m
[1m--- a/src/ir/daphneir/DaphneDialect.cpp[m
[1m+++ b/src/ir/daphneir/DaphneDialect.cpp[m
[36m@@ -156,6 +156,7 @@[m [mmlir::Type mlir::daphne::DaphneDialect::parseType(mlir::DialectAsmParser &parser[m
             return nullptr;[m
         }[m
         std::vector<mlir::Type> cts;[m
[32m+[m[32m        ssize_t properties = -1;[m
         mlir::Type type;[m
         do {[m
             if (parser.parseType(type))[m
[36m@@ -167,7 +168,7 @@[m [mmlir::Type mlir::daphne::DaphneDialect::parseType(mlir::DialectAsmParser &parser[m
             return nullptr;[m
         }[m
         return FrameType::get([m
[31m-                parser.getBuilder().getContext(), cts, numRows, numCols, nullptr[m
[32m+[m[32m                parser.getBuilder().getContext(), cts, numRows, numCols, nullptr, properties[m
         );[m
     }[m
     else if (keyword == "Handle") {[m
[36m@@ -229,6 +230,8 @@[m [mvoid mlir::daphne::DaphneDialect::printType(mlir::Type type,[m
                 os << ", ";[m
         }[m
         os << "], ";[m
[32m+[m
[32m+[m[32m        os << "properties = " << t.getProperties() << "    ";[m
         // Column labels.[m
         std::vector<std::string> * labels = t.getLabels();[m
         if(labels) {[m
[36m@@ -400,7 +403,8 @@[m [mmlir::OpFoldResult mlir::daphne::ConstantOp::fold(FoldAdaptor adaptor)[m
         ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,[m
         std::vector<Type> columnTypes,[m
         ssize_t numRows, ssize_t numCols,[m
[31m-        std::vector<std::string> * labels[m
[32m+[m[32m        std::vector<std::string> * labels,[m
[32m+[m[32m        ssize_t properties[m
 )[m
 {[m
     // TODO Verify the individual column types.[m
