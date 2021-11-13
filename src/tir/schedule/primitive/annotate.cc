/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
#include "../utils.h"

namespace tvm {
namespace tir {

void Pragma(ScheduleState self, const StmtSRef& loop_sref, const String& pragma_type,
            const PrimExpr& pragma_value, bool update) {
  AddAnn(self, loop_sref, "pragma_" + pragma_type, pragma_value, update);
}

/******** InstructionKind Registration ********/

struct PragmaTraits : public UnpackedInstTraits<PragmaTraits> {
  static constexpr const char* kName = "Pragma";
  static constexpr bool kIsPure = false;

 private:
  static constexpr size_t kNumInputs = 2;
  static constexpr size_t kNumAttrs = 1;
  static constexpr size_t kNumDecisions = 0;

  static void UnpackedApplyToSchedule(Schedule sch, LoopRV loop_rv, ExprRV pragma_value,
                                      String pragma_type) {
    return sch->Pragma(loop_rv, pragma_type, pragma_value);
  }

  static String UnpackedAsPython(Array<String> outputs, String loop_rv, String pragma_value,
                                 String pragma_type) {
    PythonAPICall py("pragma");
    py.Input("loop", loop_rv);
    py.Input("pragma_type", pragma_type);
    py.Input("pragma_value", pragma_value);
    return py.Str();
  }

  friend struct UnpackedInstTraits;
};

TVM_REGISTER_INST_KIND_TRAITS(PragmaTraits);

}  // namespace tir
}  // namespace tvm
