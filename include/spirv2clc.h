// Copyright 2020-2022 The spirv2clc authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <libspirv2clc_export.h>
#include <spirv-tools/libspirv.h>
#include <spirv/unified1/spirv.h>

namespace spvtools {
namespace opt {

class BasicBlock;
class Instruction;
class IRContext;
class Function;

namespace analysis {
class Type;
}

} // namespace opt
} // namespace spvtools

namespace spirv2clc {

struct translator {

  LIBSPIRV2CLC_EXPORT translator(spv_target_env env = SPV_ENV_OPENCL_1_2);

  LIBSPIRV2CLC_EXPORT translator(translator &&);
  LIBSPIRV2CLC_EXPORT translator &operator=(translator &&);
  LIBSPIRV2CLC_EXPORT ~translator();

  LIBSPIRV2CLC_EXPORT int translate(const std::string &assembly,
                                    std::string *srcout);
  LIBSPIRV2CLC_EXPORT int translate(const std::vector<uint32_t> &binary,
                                    std::string *srcout);

private:
  uint32_t type_id_for(uint32_t val) const;

  uint32_t type_id_for(const spvtools::opt::analysis::Type *type) const;

  spvtools::opt::analysis::Type *type_for(uint32_t tyid) const;

  spvtools::opt::analysis::Type *type_for_val(uint32_t val) const;

  uint32_t array_type_get_length(uint32_t tyid) const;

  std::string var_for(uint32_t id) const {
    if (m_literals.count(id)) {
      return m_literals.at(id);
    } else if (m_exports.count(id)) {
      return m_exports.at(id);
    } else if (m_imports.count(id)) {
      return m_imports.at(id);
    } else if (m_names.count(id)) {
      return m_names.at(id);
    } else if (m_builtin_values.count(id)) {
      switch (m_builtin_values.at(id)) {
      case SpvBuiltInWorkDim:
        return src_function_call("get_work_dim");
      default:
        return "UNIMPLEMENTED";
      }
    } else {
      return "v" + std::to_string(id);
    }
  }

  std::string src_var_decl(uint32_t tyid, const std::string &name,
                           uint32_t val = 0) const;

  std::string src_var_decl(uint32_t val) const {
    auto tyid = type_id_for(val);
    return src_var_decl(tyid, var_for(val), val);
  }

  std::string src_access_chain(const std::string &src_base,
                               const spvtools::opt::analysis::Type *ty,
                               uint32_t index) const;

  std::string src_vec_comp(uint32_t val, uint32_t comp) const {
    std::stringstream scomp;
    scomp << std::hex << comp;
    return var_for(val) + ".s" + scomp.str();
  }

  std::string src_as(uint32_t dtyid, const std::string &src) const {
    return "as_" + src_type(dtyid) + "(" + src + ")";
  }

  std::string src_as(uint32_t dtyid, uint32_t val) const {
    return src_as(dtyid, var_for(val));
  }

  std::string src_as_signed(uint32_t val) const {
    auto varty = type_id_for(val);
    return "as_" + src_type_signed(varty) + "(" + var_for(val) + ")";
  }

  std::string src_type_boolean_for_val(uint32_t val) const;

  std::string src_type(uint32_t id) const {
    if (m_types.count(id)) {
      return m_types.at(id);
    } else {
      return "UNKNOWN TYPE";
    }
  }

  std::string src_type_for_value(uint32_t idval) const {
    if (m_boolean_src_types.count(idval)) {
      return m_boolean_src_types.at(idval);
    } else {
      return src_type(type_id_for(idval));
    }
  }

  std::string src_type_signed(uint32_t id) const {
    if (m_types_signed.count(id)) {
      return m_types_signed.at(id);
    } else {
      return "UNKNOWN SIGNED TYPE";
    }
  }

  std::string src_type_memory_object_declaration(uint32_t tid, uint32_t val,
                                                 const std::string &name) const;

  std::string src_type_memory_object_declaration(uint32_t tid,
                                                 uint32_t val) const {
    return src_type_memory_object_declaration(tid, val, var_for(val));
  }

  std::string src_cast(uint32_t ty, std::string src) const {
    return "((" + src_type(ty) + ")" + src + ")";
  }

  std::string src_cast_signed(uint32_t ty, std::string src) const {
    return "((" + src_type_signed(ty) + ")" + src + ")";
  }

  std::string src_cast(uint32_t ty, uint32_t val) const {
    return src_cast(ty, var_for(val));
  }

  std::string src_cast_signed(uint32_t ty, uint32_t val) const {
    return src_cast_signed(ty, var_for(val));
  }

  std::string src_convert(uint32_t val, uint32_t ty) {
    return "convert_" + src_type(ty) + "(" + var_for(val) + ")";
  }

  std::string src_convert_signed(uint32_t val, uint32_t ty) {
    return "convert_" + src_type_signed(ty) + "(" + src_as_signed(val) + ")";
  }

  std::string src_function_call(const std::string &fn) const {
    return fn + "()";
  }

  std::string src_function_call(const std::string &fn,
                                const std::string &srcop1) const {
    return fn + "(" + srcop1 + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1) const {
    return src_function_call(fn, var_for(op1));
  }

  std::string src_function_call_signed(const std::string &fn,
                                       uint32_t op1) const {
    return src_function_call(fn, src_as_signed(op1));
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ")";
  }

  std::string src_function_call_signed(const std::string &fn, uint32_t op1,
                                       uint32_t op2) const {
    return fn + "(" + src_as_signed(op1) + ", " + src_as_signed(op2) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ")";
  }

  std::string src_function_call_signed(const std::string &fn, uint32_t op1,
                                       uint32_t op2, uint32_t op3) const {
    return fn + "(" + src_as_signed(op1) + ", " + src_as_signed(op2) + ", " +
           src_as_signed(op3) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3,
                                uint32_t op4) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ", " + var_for(op4) + ")";
  }

  std::string src_function_call(const std::string &fn, uint32_t op1,
                                uint32_t op2, uint32_t op3, uint32_t op4,
                                uint32_t op5) const {
    return fn + "(" + var_for(op1) + ", " + var_for(op2) + ", " + var_for(op3) +
           ", " + var_for(op4) + ", " + var_for(op5) + ")";
  }

  std::string src_pointer_type(uint32_t storage, uint32_t tyid, bool signedty) const;

  std::string builtin_vector_extract(uint32_t id, uint32_t idx, bool constant) const;

  bool is_valid_identifier(const std::string& name) const;
  std::string make_valid_identifier(const std::string& name) const;

  bool get_null_constant(uint32_t tyid, std::string &src) const;
  std::string
  translate_extended_unary(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_extended_binary(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_extended_ternary(const spvtools::opt::Instruction &inst) const;
  bool translate_extended_instruction(const spvtools::opt::Instruction &inst,
                                      std::string &src);
  std::string translate_binop(const spvtools::opt::Instruction &inst) const;
  std::string
  translate_binop_signed(const spvtools::opt::Instruction &inst) const;
  bool translate_instruction(const spvtools::opt::Instruction &inst,
                             std::string &src);

  bool translate_capabilities();
  bool translate_extensions() const;
  bool translate_extended_instructions_imports() const;
  bool translate_memory_model() const;
  bool translate_entry_points();
  bool translate_execution_modes();
  bool translate_debug_instructions();
  bool translate_annotations();
  bool translate_type(const spvtools::opt::Instruction &inst);
  bool translate_types_values();
  bool translate_function(spvtools::opt::Function &func);

  bool validate_module(const std::vector<uint32_t> &binary) const;
  int translate();

  void reset() {
    m_src.str("");
    m_names.clear();
    m_types.clear();
    m_types_signed.clear();
    m_literals.clear();
    m_entry_points.clear();
    m_entry_points_local_size.clear();
    m_entry_points_contraction_off.clear();
    m_builtin_variables.clear();
    m_builtin_values.clear();
    m_rounding_mode_decorations.clear();
    m_saturated_conversions.clear();
    m_exports.clear();
    m_imports.clear();
    m_restricts.clear();
    m_volatiles.clear();
    m_packed.clear();
    m_nowrite_params.clear();
    m_alignments.clear();
    m_phi_vals.clear();
    m_phi_assigns.clear();
    m_sampled_images.clear();
    m_boolean_src_types.clear();
    m_local_variable_decls.clear();
  }

  spv_target_env m_target_env;

  std::unique_ptr<spvtools::opt::IRContext> m_ir;
  std::stringstream m_src;
  std::unordered_map<uint32_t, std::string> m_names;
  std::unordered_map<uint32_t, std::string> m_types;
  std::unordered_map<uint32_t, std::string> m_types_signed;
  std::unordered_map<uint32_t, std::string> m_literals;
  std::unordered_map<uint32_t, std::string> m_entry_points;
  std::unordered_map<uint32_t, std::tuple<uint32_t, uint32_t, uint32_t>>
      m_entry_points_local_size;
  std::unordered_set<uint32_t> m_entry_points_contraction_off;
  std::unordered_map<uint32_t, SpvBuiltIn> m_builtin_variables;
  std::unordered_map<uint32_t, SpvBuiltIn> m_builtin_values;
  std::unordered_map<uint32_t, SpvFPRoundingMode> m_rounding_mode_decorations;
  std::unordered_set<uint32_t> m_saturated_conversions;
  std::unordered_map<uint32_t, std::string> m_exports;
  std::unordered_map<uint32_t, std::string> m_imports;
  std::unordered_set<uint32_t> m_restricts;
  std::unordered_set<uint32_t> m_volatiles;
  std::unordered_set<uint32_t> m_packed;
  std::unordered_set<uint32_t> m_nowrite_params;
  std::unordered_map<uint32_t, uint32_t> m_alignments;
  std::unordered_map<spvtools::opt::Function *, std::vector<uint32_t>>
      m_phi_vals;
  // phival, val pairs
  std::unordered_map<spvtools::opt::BasicBlock *,
                     std::vector<std::pair<uint32_t, uint32_t>>>
      m_phi_assigns;
  std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> m_sampled_images;
  std::unordered_map<uint32_t, std::string>
      m_boolean_src_types; // value, C type name
  std::unordered_map<uint32_t, std::string> m_local_variable_decls;
};

} // namespace spirv2clc
