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

#include "spirv2clc.h"

#define CL_TARGET_OPENCL_VERSION 120
#include "CL/cl_half.h"

#include "opt/build_module.h"
#include "opt/ir_context.h"
#include "spirv/unified1/OpenCL.std.h"

using namespace spvtools;
using namespace spvtools::opt;
using spvtools::opt::analysis::Type;

namespace {

std::string rounding_mode(SpvFPRoundingMode mode) {
  switch (mode) {
  case SpvFPRoundingModeRTE:
    return "rte";
  case SpvFPRoundingModeRTZ:
    return "rtz";
  case SpvFPRoundingModeRTP:
    return "rtp";
  case SpvFPRoundingModeRTN:
    return "rtn";
  }
  return "UNKNOWN ROUNDING MODE";
}

const spvtools::MessageConsumer spvtools_message_consumer =
    [](spv_message_level_t level, const char *, const spv_position_t &position,
       const char *message) {
      printf("spvtools says '%s' at position %zu", message, position.index);
    };

} // namespace

namespace spirv2clc {

translator::translator(spv_target_env env) : m_target_env(env) {}

translator::~translator() = default;
translator::translator(translator &&) = default;
translator &translator::operator=(translator &&) = default;

uint32_t translator::type_id_for(uint32_t val) const {
  auto defuse = m_ir->get_def_use_mgr();
  return defuse->GetDef(val)->type_id();
}

uint32_t
translator::type_id_for(const spvtools::opt::analysis::Type *type) const {
  return m_ir->get_type_mgr()->GetId(type);
}

spvtools::opt::analysis::Type *translator::type_for(uint32_t tyid) const {
  return m_ir->get_type_mgr()->GetType(tyid);
}

spvtools::opt::analysis::Type *translator::type_for_val(uint32_t val) const {
  return type_for(type_id_for(val));
}

uint32_t translator::array_type_get_length(uint32_t tyid) const {
  auto type = type_for(tyid);
  auto tarray = type->AsArray();
  auto const &length_info = tarray->length_info();
  uint64_t num_elems;
  if (length_info.words[0] !=
      spvtools::opt::analysis::Array::LengthInfo::kConstant) {
    std::cerr << "UNIMPLEMENTED array type with non-constant length"
              << std::endl;
    return 0;
  }

  if (length_info.words.size() > 2) {
    for (unsigned i = 2; i < length_info.words.size(); i++) {
      if (length_info.words[i] != 0) {
        std::cerr << "UNIMPLEMENTED array type with huge size" << std::endl;
        return 0;
      }
    }
  }

  return length_info.words[1];
}

std::string translator::src_var_decl(uint32_t tyid, const std::string &name,
                                     uint32_t val) const {
  auto ty = type_for(tyid);
  if (ty->kind() == spvtools::opt::analysis::Type::Kind::kArray) {
    auto aty = ty->AsArray();
    auto eid = type_id_for(aty->element_type());
    auto cstmgr = m_ir->get_constant_mgr();
    auto ecnt =
        cstmgr->FindDeclaredConstant(aty->LengthId())->GetSignExtendedValue();
    return src_type(eid) + " " + name + "[" + std::to_string(ecnt) + "]";
  } else {
    if (val != 0) {
      return src_type_for_value(val) + " " + name;
    } else {
      return src_type(tyid) + " " + name;
    }
  }
}

std::string
translator::src_access_chain(const std::string &src_base,
                             const spvtools::opt::analysis::Type *ty,
                             uint32_t index) const {
  std::string ret = "(" + src_base + ")";
  if (ty->kind() == spvtools::opt::analysis::Type::kStruct) {
    auto cstmgr = m_ir->get_constant_mgr();
    auto idxcst = cstmgr->FindDeclaredConstant(index);
    if (idxcst == nullptr) {
      return "UNIMPLEMENTED";
    }
    return "&(" + ret + "->m" + std::to_string(idxcst->GetZeroExtendedValue()) +
           ")";
  } else if (ty->kind() == spvtools::opt::analysis::Type::kArray) {
    return "&(" + ret + "[" + var_for(index) + "])";
  } else {
    return "UNIMPLEMENTED";
  }
}

std::string
translator::src_type_memory_object_declaration(uint32_t tid, uint32_t val,
                                               const std::string &name) const {
  std::string ret;
  if (type_for(tid)->kind() == Type::Kind::kArray) {
    auto tarray = type_for(tid)->AsArray();
    auto elemty = tarray->element_type();
    ret = src_type(type_id_for(elemty));
  } else {
    ret = src_type(tid);
  }
  if (m_restricts.count(val)) {
    ret += " restrict";
  }
  if (m_volatiles.count(val)) {
    ret += " volatile";
  }
  if (m_alignments.count(val)) {
    ret += " __attribute__((aligned(" + std::to_string(m_alignments.at(val)) +
           ")))";
  }
  ret += " " + name;
  if (type_for(tid)->kind() == Type::Kind::kArray) {
    auto len = array_type_get_length(tid);
    ret += "[" + std::to_string(len) + "]";
  }
  return ret;
}

std::string translator::src_type_boolean_for_val(uint32_t val) const {
  if (m_boolean_src_types.count(val)) {
    return m_boolean_src_types.at(val);
  } else {
    auto type = type_for_val(val);
    if (type->kind() != Type::Kind::kVector) {
      return "int";
    } else {
      auto vtype = type->AsVector();
      auto etype = vtype->element_type();
      auto ecnt = vtype->element_count();
      auto ekind = etype->kind();

      switch (ekind) {
      case Type::Kind::kInteger: {
        auto width = etype->AsInteger()->width();
        switch (width) {
        case 8:
          return "char" + std::to_string(ecnt);
        case 16:
          return "short" + std::to_string(ecnt);
        case 32:
          return "int" + std::to_string(ecnt);
        case 64:
          return "long" + std::to_string(ecnt);
        }
      }
      case Type::Kind::kFloat: {
        auto width = etype->AsFloat()->width();
        switch (width) {
        case 16:
          return "short" + std::to_string(ecnt);
        case 32:
          return "int" + std::to_string(ecnt);
        case 64:
          return "long" + std::to_string(ecnt);
        }
      }
      }
    }
  }

  std::cerr << "UNIMPLEMENTED type for translation to boolean" << std::endl;
  return "UNIMPLEMENTED TYPE FOR BOOLEAN";
}

static std::unordered_map<OpenCLLIB::Entrypoints,
                          std::pair<const std::string, bool>>
    gExtendedInstructionsTernary = {
        {OpenCLLIB::Bitselect, {"bitselect", false}},
        {OpenCLLIB::FClamp, {"clamp", false}},
        {OpenCLLIB::SClamp, {"clamp", true}},
        {OpenCLLIB::UClamp, {"clamp", false}},
        {OpenCLLIB::Fma, {"fma", false}},
        {OpenCLLIB::Mad, {"mad", false}},
        {OpenCLLIB::Mix, {"mix", false}},
        {OpenCLLIB::SMad24, {"mad24", true}},
        {OpenCLLIB::UMad24, {"mad24", false}},
        {OpenCLLIB::SMad_hi, {"mad_hi", true}},
        {OpenCLLIB::UMad_hi, {"mad_hi", false}},
        {OpenCLLIB::SMad_sat, {"mad_sat", true}},
        {OpenCLLIB::UMad_sat, {"mad_sat", false}},
        {OpenCLLIB::Select, {"select", false}},
        {OpenCLLIB::Shuffle2, {"shuffle2", false}},
        {OpenCLLIB::Smoothstep, {"smoothstep", false}},
};

std::string
translator::translate_extended_ternary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto a = inst.GetSingleWordOperand(4);
  auto b = inst.GetSingleWordOperand(5);
  auto c = inst.GetSingleWordOperand(6);
  auto fn_signed = gExtendedInstructionsTernary.at(extinst);
  if (fn_signed.second) {
    return src_as(rtype, src_function_call_signed(fn_signed.first, a, b, c));
  } else {
    return src_function_call(fn_signed.first, a, b, c);
  }
}

static std::unordered_map<OpenCLLIB::Entrypoints,
                          std::pair<const std::string, bool>>
    gExtendedInstructionsBinary = {
        {OpenCLLIB::UAbs_diff, {"abs_diff", false}},
        {OpenCLLIB::SHadd, {"hadd", true}},
        {OpenCLLIB::UHadd, {"hadd", false}},
        {OpenCLLIB::SMul_hi, {"mul_hi", true}},
        {OpenCLLIB::UMul_hi, {"mul_hi", false}},
        {OpenCLLIB::SRhadd, {"rhadd", true}},
        {OpenCLLIB::URhadd, {"rhadd", false}},
        {OpenCLLIB::Rotate, {"rotate", false}},
        {OpenCLLIB::SAdd_sat, {"add_sat", true}},
        {OpenCLLIB::UAdd_sat, {"add_sat", false}},
        {OpenCLLIB::SSub_sat, {"sub_sat", true}},
        {OpenCLLIB::USub_sat, {"sub_sat", false}},
        {OpenCLLIB::SMul24, {"mul24", true}},
        {OpenCLLIB::UMul24, {"mul24", false}},
        {OpenCLLIB::Shuffle, {"shuffle", false}},
        {OpenCLLIB::Atan2, {"atan2", false}},
        {OpenCLLIB::Atan2pi, {"atan2pi", false}},
        {OpenCLLIB::Copysign, {"copysign", false}},
        {OpenCLLIB::Fdim, {"fdim", false}},
        {OpenCLLIB::Fmax, {"fmax", false}},
        {OpenCLLIB::Fmin, {"fmin", false}},
        {OpenCLLIB::Fmod, {"fmod", false}},
        {OpenCLLIB::Hypot, {"hypot", false}},
        {OpenCLLIB::Ldexp, {"ldexp", false}},
        {OpenCLLIB::Maxmag, {"maxmag", false}},
        {OpenCLLIB::Minmag, {"minmag", false}},
        {OpenCLLIB::Modf, {"modf", false}},
        {OpenCLLIB::Nextafter, {"nextafter", false}},
        {OpenCLLIB::Pow, {"pow", false}},
        {OpenCLLIB::Pown, {"pown", false}},
        {OpenCLLIB::Powr, {"powr", false}},
        {OpenCLLIB::Remainder, {"remainder", false}},
        {OpenCLLIB::Rootn, {"rootn", false}},
        {OpenCLLIB::Sincos, {"sincos", false}},
        {OpenCLLIB::Fract, {"fract", false}},
        {OpenCLLIB::Half_divide, {"half_divide", false}},
        {OpenCLLIB::Half_powr, {"half_powr", false}},
        {OpenCLLIB::Cross, {"cross", false}},
        {OpenCLLIB::Distance, {"distance", false}},
        {OpenCLLIB::Fast_distance, {"fast_distance", false}},
        {OpenCLLIB::Step, {"step", false}},
        {OpenCLLIB::S_Upsample, {"upsample", true}},
        {OpenCLLIB::U_Upsample, {"upsample", false}},
        {OpenCLLIB::SMax, {"max", true}},
        {OpenCLLIB::UMax, {"max", false}},
        {OpenCLLIB::SMin, {"min", true}},
        {OpenCLLIB::UMin, {"min", false}},
        {OpenCLLIB::Vload_half, {"vload_half", false}},
};

std::string
translator::translate_extended_binary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto x = inst.GetSingleWordOperand(4);
  auto y = inst.GetSingleWordOperand(5);
  auto fn_signed = gExtendedInstructionsBinary.at(extinst);
  if (fn_signed.second) {
    return src_as(rtype, src_function_call_signed(fn_signed.first, x, y));
  } else {
    return src_function_call(fn_signed.first, x, y);
  }
}

static std::unordered_map<OpenCLLIB::Entrypoints, const std::string>
    gExtendedInstructionsUnary = {
        {OpenCLLIB::UAbs, "abs"},
        {OpenCLLIB::Acos, "acos"},
        {OpenCLLIB::Acosh, "acosh"},
        {OpenCLLIB::Acospi, "acospi"},
        {OpenCLLIB::Asin, "asin"},
        {OpenCLLIB::Asinh, "asinh"},
        {OpenCLLIB::Asinpi, "asinpi"},
        {OpenCLLIB::Atan, "atan"},
        {OpenCLLIB::Atanh, "atanh"},
        {OpenCLLIB::Atanpi, "atanpi"},
        {OpenCLLIB::Cbrt, "cbrt"},
        {OpenCLLIB::Ceil, "ceil"},
        {OpenCLLIB::Clz, "clz"},
        {OpenCLLIB::Cos, "cos"},
        {OpenCLLIB::Cosh, "cosh"},
        {OpenCLLIB::Cospi, "cospi"},
        {OpenCLLIB::Degrees, "degrees"},
        {OpenCLLIB::Exp, "exp"},
        {OpenCLLIB::Exp2, "exp2"},
        {OpenCLLIB::Exp10, "exp10"},
        {OpenCLLIB::Expm1, "expm1"},
        {OpenCLLIB::Fabs, "fabs"},
        {OpenCLLIB::Fast_length, "fast_length"},
        {OpenCLLIB::Fast_normalize, "fast_normalize"},
        {OpenCLLIB::Floor, "floor"},
        {OpenCLLIB::Half_cos, "half_cos"},
        {OpenCLLIB::Half_exp, "half_exp"},
        {OpenCLLIB::Half_exp2, "half_exp2"},
        {OpenCLLIB::Half_exp10, "half_exp10"},
        {OpenCLLIB::Half_log, "half_log"},
        {OpenCLLIB::Half_log2, "half_log2"},
        {OpenCLLIB::Half_log10, "half_log10"},
        {OpenCLLIB::Half_recip, "half_recip"},
        {OpenCLLIB::Half_rsqrt, "half_rsqrt"},
        {OpenCLLIB::Half_sin, "half_sin"},
        {OpenCLLIB::Half_sqrt, "half_sqrt"},
        {OpenCLLIB::Half_tan, "half_tan"},
        {OpenCLLIB::Ilogb, "ilogb"},
        {OpenCLLIB::Length, "length"},
        {OpenCLLIB::Lgamma, "lgamma"},
        {OpenCLLIB::Log, "log"},
        {OpenCLLIB::Log2, "log2"},
        {OpenCLLIB::Log10, "log10"},
        {OpenCLLIB::Log1p, "log1p"},
        {OpenCLLIB::Logb, "logb"},
        {OpenCLLIB::Nan, "nan"},
        {OpenCLLIB::Normalize, "normalize"},
        {OpenCLLIB::Radians, "radians"},
        {OpenCLLIB::Rint, "rint"},
        {OpenCLLIB::Round, "round"},
        {OpenCLLIB::Rsqrt, "rsqrt"},
        {OpenCLLIB::Sign, "sign"},
        {OpenCLLIB::Sin, "sin"},
        {OpenCLLIB::Sinh, "sinh"},
        {OpenCLLIB::Sinpi, "sinpi"},
        {OpenCLLIB::Sqrt, "sqrt"},
        {OpenCLLIB::Tan, "tan"},
        {OpenCLLIB::Tanh, "tanh"},
        {OpenCLLIB::Tanpi, "tanpi"},
        {OpenCLLIB::Trunc, "trunc"},
};

std::string
translator::translate_extended_unary(const Instruction &inst) const {
  auto rtype = inst.type_id();
  auto extinst =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));
  auto val = inst.GetSingleWordOperand(4);
  return src_function_call(gExtendedInstructionsUnary.at(extinst), val);
}

bool translator::translate_extended_instruction(const Instruction &inst,
                                                std::string &src) {
  auto rtype = inst.type_id();
  auto result = inst.result_id();
  auto set = inst.GetSingleWordOperand(2);
  auto instruction =
      static_cast<OpenCLLIB::Entrypoints>(inst.GetSingleWordOperand(3));

  std::string sval;
  bool assign_result = true;

  if (gExtendedInstructionsUnary.count(instruction)) {
    sval = translate_extended_unary(inst);
  } else if (gExtendedInstructionsBinary.count(instruction)) {
    sval = translate_extended_binary(inst);
  } else if (gExtendedInstructionsTernary.count(instruction)) {
    sval = translate_extended_ternary(inst);
  } else {
    switch (instruction) {
    case OpenCLLIB::Vloadn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vload" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vload_halfn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vload_half" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vloada_halfn: {
      auto offset = inst.GetSingleWordOperand(4);
      auto ptr = inst.GetSingleWordOperand(5);
      auto n = inst.GetSingleWordOperand(6);
      sval = src_function_call("vloada_half" + std::to_string(n), offset, ptr);
      break;
    }
    case OpenCLLIB::Vstoren: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstore" + std::to_string(n), data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_half: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      src = src_function_call("vstore_half", data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_half_r: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      auto mode = inst.GetSingleWordOperand(7);
      std::string mode_str =
          rounding_mode(static_cast<SpvFPRoundingMode>(mode));
      assign_result = false;
      src = src_function_call("vstore_half_" + mode_str, data, offset, ptr);
      break;
    }
    case OpenCLLIB::Vstore_halfn: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstore_half" + std::to_string(n), data, offset,
                              ptr);
      break;
    }
    case OpenCLLIB::Vstorea_halfn: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src = src_function_call("vstorea_half" + std::to_string(n), data, offset,
                              ptr);
      break;
    }
    case OpenCLLIB::Vstorea_halfn_r: {
      auto data = inst.GetSingleWordOperand(4);
      auto offset = inst.GetSingleWordOperand(5);
      auto ptr = inst.GetSingleWordOperand(6);
      auto mode = inst.GetSingleWordOperand(7);
      std::string mode_str =
          rounding_mode(static_cast<SpvFPRoundingMode>(mode));
      assign_result = false;
      auto n = type_for_val(data)->AsVector()->element_count();
      src =
          src_function_call("vstorea_half" + std::to_string(n) + "_" + mode_str,
                            data, offset, ptr);
      break;
    }
    case OpenCLLIB::SAbs: {
      auto val = inst.GetSingleWordOperand(4);
      sval = src_function_call_signed("abs", val);
      break;
    }
    case OpenCLLIB::SAbs_diff: {
      auto a = inst.GetSingleWordOperand(4);
      auto b = inst.GetSingleWordOperand(5);
      sval = src_function_call_signed("abs_diff", a, b);
      break;
    }
    case OpenCLLIB::Frexp: {
      auto x = inst.GetSingleWordOperand(4);
      auto exp = inst.GetSingleWordOperand(5);
      sval = src_function_call(
          "frexp", var_for(x) + ", " + src_cast_signed(type_id_for(exp), exp));
      break;
    }
    case OpenCLLIB::Lgamma_r: {
      auto x = inst.GetSingleWordOperand(4);
      auto signp = inst.GetSingleWordOperand(5);
      sval = src_function_call("lgamma_r",
                               var_for(x) + ", " +
                                   src_cast_signed(type_id_for(signp), signp));
      break;
    }
    case OpenCLLIB::Remquo: {
      auto x = inst.GetSingleWordOperand(4);
      auto y = inst.GetSingleWordOperand(5);
      auto quo = inst.GetSingleWordOperand(6);
      sval = src_function_call("remquo",
                               var_for(x) + ", " + var_for(y) + ", " +
                                   src_cast_signed(type_id_for(quo), quo));
      break;
    }
    case OpenCLLIB::Printf: {
      auto format = inst.GetSingleWordOperand(4);
      std::string src_args = var_for(format);
      for (unsigned op = 5; op < inst.NumOperands(); op++) {
        auto arg = inst.GetSingleWordOperand(op);
        src_args += ", " + var_for(arg);
      }
      sval = src_function_call("printf", src_args);
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED extended instruction " << instruction
                << std::endl;
      return false;
    }
  }

  if ((result != 0) && assign_result) {
    src = src_var_decl(result) + " = " + sval;
  }

  return true;
}

bool translator::get_null_constant(uint32_t tyid, std::string &src) const {
  auto type = type_for(tyid);
  switch (type->kind()) {
  case Type::Kind::kInteger:
    src = src_cast(tyid, "0");
    break;
  case Type::Kind::kFloat:
    src = "0.0";
    break;
  case Type::Kind::kArray:
  case Type::Kind::kStruct:
    src = "{0}";
    break;
  case Type::Kind::kBool:
    src = "false";
    break;
  case Type::Kind::kVector:
    src = "((" + src_type(tyid) + ")(0))";
    break;
  case Type::Kind::kEvent:
    src = "0";
    break;
  default:
    std::cerr << "UNIMPLEMENTED null constant type " << type->kind()
              << std::endl;
    return false;
  }

  return true;
}

std::string translator::translate_binop(const Instruction &inst) const {
  static std::unordered_map<SpvOp, const std::string> binops = {
      {SpvOpFMul, "*"},
      {SpvOpFDiv, "/"},
      {SpvOpFAdd, "+"},
      {SpvOpFSub, "-"},
      {SpvOpISub, "-"},
      {SpvOpIAdd, "+"},
      {SpvOpIMul, "*"},
      {SpvOpUDiv, "/"},
      {SpvOpUMod, "%"},
      {SpvOpULessThan, "<"},
      {SpvOpULessThanEqual, "<="},
      {SpvOpUGreaterThan, ">"},
      {SpvOpUGreaterThanEqual, ">="},
      {SpvOpLogicalEqual, "=="},
      {SpvOpLogicalNotEqual, "!="},
      {SpvOpIEqual, "=="},
      {SpvOpINotEqual, "!="},
      {SpvOpBitwiseOr, "|"},
      {SpvOpBitwiseXor, "^"},
      {SpvOpBitwiseAnd, "&"},
      {SpvOpLogicalOr, "||"},
      {SpvOpLogicalAnd, "&&"},
      {SpvOpVectorTimesScalar, "*"},
      {SpvOpShiftLeftLogical, "<<"},
      {SpvOpShiftRightLogical, ">>"},
      {SpvOpFOrdEqual, "=="},
      {SpvOpFUnordEqual, "=="},
      {SpvOpFOrdNotEqual, "!="},
      {SpvOpFUnordNotEqual, "!="},
      {SpvOpFOrdLessThan, "<"},
      {SpvOpFUnordLessThan, "<"},
      {SpvOpFOrdGreaterThan, ">"},
      {SpvOpFUnordGreaterThan, ">"},
      {SpvOpFOrdLessThanEqual, "<="},
      {SpvOpFUnordLessThanEqual, "<="},
      {SpvOpFOrdGreaterThanEqual, ">="},
      {SpvOpFUnordGreaterThanEqual, ">="},
  };

  auto v1 = inst.GetSingleWordOperand(2);
  auto v2 = inst.GetSingleWordOperand(3);

  auto &srcop = binops.at(inst.opcode());

  return var_for(v1) + " " + srcop + " " + var_for(v2);
}

std::string translator::translate_binop_signed(const Instruction &inst) const {
  static std::unordered_map<SpvOp, const std::string> binops = {
      {SpvOpSDiv, "/"},
      {SpvOpSRem, "%"},
      {SpvOpShiftRightArithmetic, ">>"},
      {SpvOpSLessThan, "<"},
      {SpvOpSLessThanEqual, "<="},
      {SpvOpSGreaterThan, ">"},
      {SpvOpSGreaterThanEqual, ">="},
  };

  auto v1 = inst.GetSingleWordOperand(2);
  auto v2 = inst.GetSingleWordOperand(3);

  auto &srcop = binops.at(inst.opcode());

  return src_as_signed(v1) + " " + srcop + " " + src_as_signed(v2);
}

std::string translator::builtin_vector_extract(uint32_t id, uint32_t idx, bool constant) const {
  std::string arg;
  if (constant) {
    arg = std::to_string(idx);
  } else {
    arg = var_for(idx);
  }

  switch (m_builtin_values.at(id)) {
  case SpvBuiltInGlobalInvocationId:
    return src_function_call("get_global_id", arg);
  case SpvBuiltInGlobalOffset:
    return src_function_call("get_global_offset", arg);
  case SpvBuiltInGlobalSize:
    return src_function_call("get_global_size", arg);
  case SpvBuiltInWorkgroupId:
    return src_function_call("get_group_id", arg);
  case SpvBuiltInWorkgroupSize:
    return src_function_call("get_local_size", arg);
  case SpvBuiltInLocalInvocationId:
    return src_function_call("get_local_id", arg);
  case SpvBuiltInNumWorkgroups:
    return src_function_call("get_num_groups", arg);
  default:
    std::cerr << "UNIMPLEMENTED built-in in builtin_vector_extract" << std::endl;
    return "UNIMPLEMENTED";
  }
}

bool translator::translate_instruction(const Instruction &inst,
                                       std::string &src) {
  auto opcode = inst.opcode();
  auto rtype = inst.type_id();
  auto result = inst.result_id();

  std::string sval;
  bool assign_result = true;
  bool boolean_result = false;
  std::string boolean_result_src_type;

  switch (opcode) {
  case SpvOpUndef: {
    if (!get_null_constant(rtype, sval)) {
      return false;
    }
    break;
  }
  case SpvOpUnreachable: // TODO trigger crash? end invocation?
    break;
  case SpvOpReturn:
    src = "return";
    break;
  case SpvOpReturnValue: {
    auto val = inst.GetSingleWordOperand(0);
    src = "return " + var_for(val);
    break;
  }
  case SpvOpFunctionCall: {
    auto func = inst.GetSingleWordOperand(2);
    sval = var_for(func) + "(";
    const char *sep = "";
    for (int i = 3; i < inst.NumOperands(); i++) {
      auto param = inst.GetSingleWordOperand(i);
      sval += sep;
      sval += var_for(param);
      sep = ", ";
    }
    sval += ")";
    if (type_for(rtype)->kind() == Type::Kind::kVoid) {
      assign_result = false;
      src = sval;
    }
    break;
  }
  case SpvOpCopyObject: {
    auto obj = inst.GetSingleWordOperand(2);
    sval = var_for(obj);
    break;
  }
  case SpvOpLifetimeStart:
  case SpvOpLifetimeStop:
    break;
  case SpvOpVariable: {
    auto storage = inst.GetSingleWordOperand(2);
    assign_result = false;
    auto varty = type_for(rtype)->AsPointer()->pointee_type();
    auto storagename = var_for(result) + "_storage";
    storagename = make_valid_identifier(storagename);
    // Declare storage
    auto tymgr = m_ir->get_type_mgr();
    src = src_type_memory_object_declaration(tymgr->GetId(varty), result,
                                             storagename);
    if (inst.NumOperands() == 4) {
      auto init = inst.GetSingleWordOperand(3);
      src += " = " + var_for(init);
    }
    src += "; ";
    // Declare pointer
    src += src_type(rtype) + " " + var_for(result) + " = &" + storagename;
    break;
  }
  case SpvOpLoad: {
    auto ptr = inst.GetSingleWordOperand(2);
    if (m_builtin_variables.count(ptr)) {
      m_builtin_values[result] = m_builtin_variables.at(ptr);
      assign_result = false;
    } else {
      sval = "*" + var_for(ptr);
    }
    break;
  }
  case SpvOpStore: {
    auto ptr = inst.GetSingleWordOperand(0);
    auto val = inst.GetSingleWordOperand(1);
    src = "*" + var_for(ptr) + " = " + var_for(val);
    break;
  }
  case SpvOpConvertPtrToU:
  case SpvOpConvertUToPtr: {
    auto src = inst.GetSingleWordOperand(2);
    sval = src_cast(rtype, src);
    break;
  }
  case SpvOpInBoundsPtrAccessChain: {
    auto base = inst.GetSingleWordOperand(2);
    auto elem = inst.GetSingleWordOperand(3);
    sval = "&" + var_for(base) + "[" + var_for(elem) + "]";
    const Type *cty = type_for_val(base)->AsPointer()->pointee_type();
    for (int i = 4; i < inst.NumOperands(); i++) {
      auto idx = inst.GetSingleWordOperand(i);
      sval = src_access_chain(sval, cty, idx);
      switch (cty->kind()) {
      case Type::Kind::kArray:
        cty = cty->AsArray()->element_type();
        break;
      case Type::Kind::kStruct:
        cty = cty->AsStruct()->element_types()[idx];
        break;
      deafult:
        std::cerr << "UNIMPLEMENTED access chain type " << cty->kind()
                  << std::endl;
        return false;
      }
    }
    break;
  }
  case SpvOpSampledImage: {
    auto image = inst.GetSingleWordOperand(2);
    auto sampler = inst.GetSingleWordOperand(3);
    m_sampled_images[result] = std::make_pair(image, sampler);
    assign_result = false;
    break;
  }
  case SpvOpImageSampleExplicitLod: {
    auto sampledimage = inst.GetSingleWordOperand(2);
    auto coord = inst.GetSingleWordOperand(3);
    auto operands = inst.GetSingleWordOperand(4);
    bool is_float = type_for(rtype)->kind() == Type::Kind::kFloat;
    bool is_float_coord = type_for_val(coord)->kind() == Type::Kind::kFloat;

    if (!is_float) {
      sval += "as_uint4(";
    }

    sval += "read_image";

    if (is_float) {
      sval += "f";
    } else {
      sval += "i"; // FIXME i vs. ui
    }

    sval += "(";
    sval += var_for(m_sampled_images.at(sampledimage).first);
    sval += ", ";
    sval += var_for(m_sampled_images.at(sampledimage).second);
    sval += ", ";
    if (!is_float_coord) {
      sval += "as_int2(";
    }
    sval += var_for(coord);
    if (!is_float_coord) {
      sval += ")";
    }
    sval += ")";
    if (!is_float) {
      sval += ")";
    }
    // TODO check Lod

    break;
  }
#if 0
    case SpvOpImageWrite: {
        auto image = inst.GetSingleWordOperand(0);
        auto coord = inst.GetSingleWordOperand(1);
        auto texel = inst.GetSingleWordOperand(2);
        auto tycoord = type_for_val(coord);
        auto tytexel = type_for_val(texel);
        bool is_float = tytexel->kind() == Type::Kind::kFloat;
        bool is_float_coord = tycoord->kind() == Type::Kind::kFloat;
        src = "write_image";
        if (is_float) {
            src += "f";
        } else {
            src += "ui"; // FIXME i vs. ui
        }
        src += "(";
        src += var_for(image);
        src += ", ";
        if (!is_float_coord) {
            src += "as_int2(";
        }
        src += var_for(coord);
        if (!is_float_coord) {
            src += ")";
        }
        src += ", ";
        src += var_for(texel);
        src += ")";
        break;
    }
#endif
  case SpvOpImageQuerySizeLod: {
    auto image = inst.GetSingleWordOperand(2);
    auto lod = inst.GetSingleWordOperand(3); // FIXME validate
    sval = "((" + src_type(rtype) + ")(";
    auto tyimg = type_for_val(image);
    sval += "get_image_width(" + var_for(image) + ")";
    auto dim = tyimg->AsImage()->dim();
    if ((dim == SpvDim2D) || (dim == SpvDim3D)) {
      sval += ", get_image_height(" + var_for(image) + ")";
    }
    if (dim == SpvDim3D) {
      sval += ", get_image_depth(" + var_for(image) + ")";
    }
    sval += "))";
    break;
  }
  case SpvOpAtomicIIncrement: {
    auto ptr = inst.GetSingleWordOperand(2);
    sval = src_function_call("atomic_inc", ptr); // FIXME exact semantics
    break;
  }
  case SpvOpAtomicIDecrement: {
    auto ptr = inst.GetSingleWordOperand(2);
    sval = src_function_call("atomic_dec", ptr); // FIXME exact semantics
    break;
  }
  case SpvOpAtomicAnd:
  case SpvOpAtomicExchange:
  case SpvOpAtomicIAdd:
  case SpvOpAtomicISub:
  case SpvOpAtomicOr:
  case SpvOpAtomicSMax:
  case SpvOpAtomicSMin:
  case SpvOpAtomicUMax:
  case SpvOpAtomicUMin:
  case SpvOpAtomicXor: {
    static std::unordered_map<SpvOp, const char *> fns{
        {SpvOpAtomicAnd, "atomic_and"},  {SpvOpAtomicExchange, "atomic_xchg"},
        {SpvOpAtomicIAdd, "atomic_add"}, {SpvOpAtomicISub, "atomic_sub"},
        {SpvOpAtomicOr, "atomic_or"},    {SpvOpAtomicSMax, "atomic_max"},
        {SpvOpAtomicSMin, "atomic_min"}, {SpvOpAtomicUMax, "atomic_max"},
        {SpvOpAtomicUMin, "atomic_min"}, {SpvOpAtomicXor, "atomic_xor"},
    };
    auto ptr = inst.GetSingleWordOperand(2);
    auto val = inst.GetSingleWordOperand(5);
    sval = src_function_call(fns.at(opcode), ptr, val); // FIXME exact semantics
    break;
  }
  case SpvOpAtomicCompareExchange: {
    auto ptr = inst.GetSingleWordOperand(2);
    auto val = inst.GetSingleWordOperand(6);
    auto cmp = inst.GetSingleWordOperand(7);
    sval = src_function_call("atomic_cmpxchg", ptr, cmp,
                             val); // FIXME exact semantics
    break;
  }
  case SpvOpCompositeExtract: {
    auto comp = inst.GetSingleWordOperand(2);
    auto idx = inst.GetSingleWordOperand(3); // FIXME support multiple indices
    if (m_builtin_values.count(comp)) {
      sval = builtin_vector_extract(comp, idx, true);
      break;
    }
    auto type = type_for_val(comp);
    switch (type->kind()) {
    case Type::Kind::kVector: {
      sval = src_vec_comp(comp, idx);
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED OpCompositeExtract, type " << type->kind()
                << std::endl;
      return false;
    }
    break;
  }
  case SpvOpCompositeInsert: {
    auto object = inst.GetSingleWordOperand(2);
    auto composite = inst.GetSingleWordOperand(3);
    auto index = inst.GetSingleWordOperand(4);

    if (inst.NumOperands() > 5) {
      std::cerr << "UNIMPLEMENTED OpCompositeInsert with multiple indices"
                << std::endl;
      return false;
    }

    assign_result = false;
    src = src_type(rtype) + " " + var_for(result) + " = " + var_for(composite) +
          "; ";
    auto type = type_for(rtype);
    switch (type->kind()) {
    case Type::Kind::kVector:
      src += src_vec_comp(result, index) + " = " + var_for(object);
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpCompositeInsert, type " << type->kind()
                << std::endl;
      return false;
    }
    break;
  }
  case SpvOpCompositeConstruct: {
    sval = "{";
    const char *sep = "";
    for (int i = 2; i < inst.NumOperands(); i++) {
      auto mem = inst.GetSingleWordOperand(i);
      sval += sep;
      sval += var_for(mem);
      sep = ", ";
    }
    sval += "}";
    break;
  }
  case SpvOpVectorExtractDynamic: {
    // ((elemtype)&vec)[elem]
    auto vec = inst.GetSingleWordOperand(2);
    auto idx = inst.GetSingleWordOperand(3);
    if (m_builtin_values.count(vec)) {
      sval = builtin_vector_extract(vec, idx, false);
    } else {
      sval = "((" + src_type(rtype) + "*)&" + var_for(vec) + ")[" + var_for(idx) +
           "]";
    }
    break;
  }
  case SpvOpVectorInsertDynamic: {
    auto vec = inst.GetSingleWordOperand(2);
    auto comp = inst.GetSingleWordOperand(3);
    auto comp_type_id = type_id_for(comp);
    auto idx = inst.GetSingleWordOperand(4);
    sval = var_for(vec);
    sval += "; ";
    sval += "((" + src_type(comp_type_id) + "*)&" + var_for(result) + ")[" +
            var_for(idx) + "] = " + var_for(comp);
    break;
  }
  case SpvOpVectorShuffle: {
    auto v1 = inst.GetSingleWordOperand(2);
    auto v2 = inst.GetSingleWordOperand(3);
    auto n1 = type_for_val(v1)->AsVector()->element_count();
    sval = "((" + src_type(rtype) + ")(";
    const char *sep = "";
    for (int i = 4; i < inst.NumOperands(); i++) {
      auto comp = inst.GetSingleWordOperand(i);
      auto srcvec = v1;
      sval += sep;
      if (comp == 0xFFFFFFFFU) {
        sval += "0";
      } else {
        if (comp >= n1) {
          srcvec = v2;
          comp -= n1;
        }
        sval += src_vec_comp(srcvec, comp);
      }
      sep = ", ";
    }
    sval += "))";
    break;
  }
  case SpvOpSDiv:
  case SpvOpSRem:
  case SpvOpShiftRightArithmetic:
    sval = src_as(rtype, translate_binop_signed(inst));
    break;
  case SpvOpVectorTimesScalar:
  case SpvOpShiftLeftLogical:
  case SpvOpShiftRightLogical:
  case SpvOpFAdd:
  case SpvOpFSub:
  case SpvOpFDiv:
  case SpvOpFMul:
  case SpvOpISub:
  case SpvOpIAdd:
  case SpvOpIMul:
  case SpvOpUDiv:
  case SpvOpUMod:
  case SpvOpBitwiseOr:
  case SpvOpBitwiseXor:
  case SpvOpBitwiseAnd:
    sval = translate_binop(inst);
    break;
  case SpvOpFMod:
  case SpvOpFRem: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    sval = src_function_call("fmod", op1, op2);
    break;
  }
  case SpvOpSNegate:
  case SpvOpFNegate: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "-" + var_for(op);
    break;
  }
  case SpvOpLogicalNot: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "!" + var_for(op);
    break;
  }
  case SpvOpNot: {
    auto op = inst.GetSingleWordOperand(2);
    sval = "~" + var_for(op);
    break;
  }
  case SpvOpLessOrGreater: {
    auto op1 = inst.GetSingleWordOperand(2);
    auto op2 = inst.GetSingleWordOperand(3);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = src_function_call("islessgreater", op1, op2);
    break;
  }
  case SpvOpFOrdEqual:
  case SpvOpFOrdNotEqual:
  case SpvOpFOrdLessThan:
  case SpvOpFOrdGreaterThan:
  case SpvOpFOrdLessThanEqual:
  case SpvOpFOrdGreaterThanEqual:
  case SpvOpFUnordEqual:
  case SpvOpFUnordNotEqual:
  case SpvOpFUnordLessThan:
  case SpvOpFUnordGreaterThan:
  case SpvOpFUnordLessThanEqual:
  case SpvOpFUnordGreaterThanEqual:
  case SpvOpLogicalOr:
  case SpvOpLogicalAnd:
  case SpvOpULessThan:
  case SpvOpULessThanEqual:
  case SpvOpUGreaterThan:
  case SpvOpUGreaterThanEqual:
  case SpvOpLogicalEqual:
  case SpvOpLogicalNotEqual:
  case SpvOpIEqual:
  case SpvOpINotEqual: {
    auto op1 = inst.GetSingleWordOperand(2);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = translate_binop(inst);
    break;
  }
  case SpvOpSLessThanEqual:
  case SpvOpSGreaterThan:
  case SpvOpSGreaterThanEqual:
  case SpvOpSLessThan: {
    auto op1 = inst.GetSingleWordOperand(2);
    boolean_result = true;
    boolean_result_src_type = src_type_boolean_for_val(op1);
    sval = translate_binop_signed(inst);
    break;
  }
  case SpvOpAny: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("any", val);
    break;
  }
  case SpvOpAll: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("all", val);
    break;
  }
  case SpvOpIsNan: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isnan", val);
    break;
  }
  case SpvOpIsInf: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isinf", val);
    break;
  }
  case SpvOpIsFinite: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isfinite", val);
    break;
  }
  case SpvOpIsNormal: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("isnormal", val);
    break;
  }
  case SpvOpSignBitSet: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("signbit", val);
    break;
  }
  case SpvOpBitCount: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call("popcount", val);
    break;
  }
  case SpvOpOrdered: {
    auto x = inst.GetSingleWordOperand(2);
    auto y = inst.GetSingleWordOperand(3);
    sval = src_function_call("isordered", x, y);
    break;
  }
  case SpvOpUnordered: {
    auto x = inst.GetSingleWordOperand(2);
    auto y = inst.GetSingleWordOperand(3);
    sval = src_function_call("isunordered", x, y);
    break;
  }
  case SpvOpConvertFToU:
  case SpvOpConvertFToS: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    sval = "convert_";
    if (opcode == SpvOpConvertFToU) {
      sval += src_type(rtype);
    } else {
      sval += src_type_signed(rtype);
    }

    if (sat) {
      sval += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      sval += "_" + rounding_mode(rmode);
    } else {
      sval += "_" + rounding_mode(SpvFPRoundingModeRTZ);
    }

    sval += "(" + var_for(op) + ")";

    // SPIR-V requires that NaNs be converted to 0 for saturating conversions
    // but OpenCL C just recommends it (§6.2.3)
    if (sat) {
      sval = src_function_call("isnan", op) + " ? 0 : " + sval;
    }

    break;
  }
  case SpvOpDot: {
    auto v1 = inst.GetSingleWordOperand(2);
    auto v2 = inst.GetSingleWordOperand(3);
    sval = src_function_call("dot", v1, v2);
    break;
  }
  case SpvOpConvertUToF:
  case SpvOpConvertSToF: {
    auto op = inst.GetSingleWordOperand(2);
    bool sat = m_saturated_conversions.count(result);
    sval = "convert_";
    sval += src_type(rtype);

    if (sat) {
      sval += "_sat";
    }

    if (m_rounding_mode_decorations.count(result)) {
      auto rmode = m_rounding_mode_decorations.at(result);
      sval += "_" + rounding_mode(rmode);
    }

    sval += "(" + var_for(op) + ")";

    break;
  }
  case SpvOpSatConvertSToU: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_as(
        rtype,
        src_function_call("convert_" + src_type_signed(rtype) + "_sat", val));
    break;
  }
  case SpvOpSatConvertUToS: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_function_call_signed("convert_" + src_type(rtype) + "_sat", val);
    break;
  }
  case SpvOpBitcast: {
    auto val = inst.GetSingleWordOperand(2);
    auto dstty = type_for(rtype);
    auto srcty = type_for_val(val);
    if ((srcty->kind() == Type::Kind::kPointer) ||
        (dstty->kind() == Type::Kind::kPointer)) {
      sval = src_cast(rtype, val);
    } else {
      sval = src_as(rtype, val);
    }
    break;
  }
  case SpvOpSConvert: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_convert_signed(val, rtype);
    break;
  }
  case SpvOpFConvert:
  case SpvOpUConvert: {
    auto val = inst.GetSingleWordOperand(2);
    sval = src_convert(val, rtype);
    break;
  }
  case SpvOpSelect: {
    auto cond = inst.GetSingleWordOperand(2);
    auto val_true = inst.GetSingleWordOperand(3);
    auto val_false = inst.GetSingleWordOperand(4);
    sval =
        var_for(cond) + " ? " + var_for(val_true) + " : " + var_for(val_false);
    break;
  }
  case SpvOpBranch: {
    auto target = inst.GetSingleWordOperand(0);
    assign_result = false;
    src = "goto " + var_for(target);
    break;
  }
  case SpvOpBranchConditional: {
    auto cond = inst.GetSingleWordOperand(0);
    auto label_true = inst.GetSingleWordOperand(1);
    auto label_false = inst.GetSingleWordOperand(2);
    assign_result = false;
    src = "if (" + var_for(cond) + ") { goto " + var_for(label_true) +
          ";} else { goto " + var_for(label_false) + ";}";
    break;
  }
  case SpvOpLoopMerge:      // Nothing to do for now TODO loop controls
  case SpvOpSelectionMerge: // TODO selection controls
    break;
  case SpvOpPhi: // Nothing to do here, phi registers are assigned elsewhere
    assign_result = false;
    break;
  case SpvOpSwitch: {
    assign_result = false;
    auto select = inst.GetSingleWordOperand(0);
    auto def = inst.GetSingleWordOperand(1);
    src = "switch (" + var_for(select) + "){";
    src += "default: goto " + var_for(def) + ";";
    for (int i = 2; i < inst.NumOperands(); i += 2) {
      auto &val = inst.GetOperand(i);
      auto &target = inst.GetOperand(i + 1);
      src += "case " + std::to_string(val.AsLiteralUint64()) + ": goto " +
             var_for(target.AsId()) + ";";
    }
    src += "}";
    break;
  }
  case SpvOpControlBarrier: {
    auto execution_scope = inst.GetSingleWordOperand(0);
    auto memory_scope = inst.GetSingleWordOperand(1);
    auto memory_semantics = inst.GetSingleWordOperand(2);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant execution scope"
          << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-workgroup execution scope"
          << std::endl;
      return false;
    }

    auto mem_scope_cst = cstmgr->FindDeclaredConstant(memory_scope);
    if (mem_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant memory scope"
          << std::endl;
      return false;
    }

    std::string flags;
    switch (mem_scope_cst->GetU32()) {
    case SpvScopeWorkgroup:
      flags = "CLK_LOCAL_MEM_FENCE";
      break;
    case SpvScopeDevice:
      flags = "CLK_GLOBAL_MEM_FENCE";
      break;
    default:
      std::cerr << "UNIMPLEMENTED memory scope in OpControlBarrier "
                << memory_scope << std::endl;
      return false;
    }

    auto mem_sem_cst = cstmgr->FindDeclaredConstant(memory_semantics);
    if (mem_sem_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpControlBarrier with non-constant memory semantics"
          << std::endl;
      return false;
    }

    auto mem_sem = mem_sem_cst->GetU32();
    if ((mem_sem != (SpvMemorySemanticsSequentiallyConsistentMask |
                     SpvMemorySemanticsWorkgroupMemoryMask)) &&
        (mem_sem != (SpvMemorySemanticsSequentiallyConsistentMask |
                     SpvMemorySemanticsCrossWorkgroupMemoryMask))) {
      std::cerr << "UNIMPLEMENTED OpControlBarrier with memory semantics "
                << mem_sem << std::endl;
      return false;
    }

    src = src_function_call("barrier", flags);
    break;
  }
  case SpvOpGroupAsyncCopy: {
    auto execution_scope = inst.GetSingleWordOperand(2);
    auto dst_ptr = inst.GetSingleWordOperand(3);
    auto src_ptr = inst.GetSingleWordOperand(4);
    auto num_elems = inst.GetSingleWordOperand(5);
    auto stride = inst.GetSingleWordOperand(6);
    auto event = inst.GetSingleWordOperand(7);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpGroupAsyncCopy with non-constant execution scope"
          << execution_scope << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr
          << "UNIMPLEMENTED OpGroupAsyncCopy with non-workgroup execution scope"
          << std::endl;
      return false;
    }

    auto stride_cst = cstmgr->FindDeclaredConstant(stride);

    if ((stride_cst != nullptr) && (stride_cst->GetZeroExtendedValue() == 1)) {
      sval = src_function_call("async_work_group_copy", dst_ptr, src_ptr,
                               num_elems, event);
    } else {
      sval = src_function_call("async_work_group_strided_copy", dst_ptr,
                               src_ptr, num_elems, stride, event);
    }

    break;
  }
  case SpvOpGroupWaitEvents: {
    auto execution_scope = inst.GetSingleWordOperand(0);
    auto num_events = inst.GetSingleWordOperand(1);
    auto event_list = inst.GetSingleWordOperand(2);

    auto cstmgr = m_ir->get_constant_mgr();

    auto exec_scope_cst = cstmgr->FindDeclaredConstant(execution_scope);
    if (exec_scope_cst == nullptr) {
      std::cerr
          << "UNIMPLEMENTED OpGroupWaitEvents with non-constant execution scope"
          << std::endl;
      return false;
    }

    if (exec_scope_cst->GetU32() != SpvScopeWorkgroup) {
      std::cerr << "UNIMPLEMENTED OpGroupWaitEvents with non-workgroup "
                   "execution scope"
                << std::endl;
      return false;
    }

    src = src_function_call("wait_group_events", num_events, event_list);
    assign_result = false;
    break;
  }
  case SpvOpExtInst: {
    assign_result = false;
    if (!translate_extended_instruction(inst, src)) {
      return false;
    }
    break;
  }
  default:
    std::cerr << "UNIMPLEMENTED instruction " << opcode << std::endl;
    return false;
  }

  if (boolean_result) {
    m_boolean_src_types[result] = boolean_result_src_type;
  }

  if ((result != 0) && assign_result) {
    src = src_var_decl(result);
    src += " = " + sval;
  }

  return true;
}

bool translator::translate_capabilities() {
  for (auto &inst : m_ir->capabilities()) {
    assert(inst.opcode() == SpvOpCapability);
    auto cap = inst.GetSingleWordOperand(0);
    switch (cap) {
    case SpvCapabilityAddresses:
    case SpvCapabilityLinkage:
    case SpvCapabilityKernel:
    case SpvCapabilityInt8:
    case SpvCapabilityInt16:
    case SpvCapabilityInt64:
    case SpvCapabilityVector16:
    case SpvCapabilityImageBasic:
    case SpvCapabilityLiteralSampler:
    case SpvCapabilityFloat16Buffer:
      break;
    case SpvCapabilityFloat16:
      m_src << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable" << std::endl;
      break;
    case SpvCapabilityFloat64:
      m_src << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable" << std::endl;
      break;
    default:
      std::cerr << "UNIMPLEMENTED capability " << cap << ".\n";
      return false;
    }
  }
  return true;
}

bool translator::translate_extensions() const {
  for (auto &inst : m_ir->module()->extensions()) {
    assert(inst.opcode() == SpvOpExtension);
    auto &op_ext = inst.GetOperand(0);
    auto ext = op_ext.AsString();
    if (ext != "SPV_KHR_no_integer_wrap_decoration") {
      std::cerr << "UNIMPLEMENTED extension " << ext << ".\n";
      return false;
    }
  }
  return true;
}

bool translator::translate_extended_instructions_imports() const {
  for (auto &inst : m_ir->ext_inst_imports()) {
    assert(inst.opcode() == SpvOpExtInstImport);
    auto name = inst.GetOperand(1).AsString();
    if (name != "OpenCL.std") {
      std::cerr << "UNIMPLEMENTED extended instruction set.\n";
      return false;
    }
  }
  return true;
}

bool translator::translate_memory_model() const {
  auto inst = m_ir->module()->GetMemoryModel();
  auto add = inst->GetSingleWordOperand(0);
  auto mem = inst->GetSingleWordOperand(1);

  if ((add != SpvAddressingModelPhysical32) &&
      (add != SpvAddressingModelPhysical64)) {
    return false;
  }
  if (mem != SpvMemoryModelOpenCL) {
    return false;
  }

  return true;
}

bool translator::translate_entry_points() {
  for (auto &ep : m_ir->module()->entry_points()) {
    auto model = ep.GetSingleWordOperand(0);
    auto func = ep.GetSingleWordOperand(1);
    auto &op_name = ep.GetOperand(2);

    if (model != SpvExecutionModelKernel) {
      return false;
    }

    m_entry_points[func] = op_name.AsString();
  }

  return true;
}

bool translator::translate_execution_modes() {
  for (auto &em : m_ir->module()->execution_modes()) {
    auto ep = em.GetSingleWordOperand(0);
    auto mode = em.GetSingleWordOperand(1);
    switch (mode) {
    case SpvExecutionModeLocalSize: {
      auto x = em.GetSingleWordOperand(2);
      auto y = em.GetSingleWordOperand(3);
      auto z = em.GetSingleWordOperand(4);
      m_entry_points_local_size[ep] = std::make_tuple(x, y, z);
      break;
    }
    case SpvExecutionModeContractionOff:
      m_entry_points_contraction_off.insert(ep);
      break;
    default:
      std::cerr << "UNIMPLEMENTED execution mode " << mode << ".\n";
      return false;
    }
  }
  return true;
}

std::unordered_set<std::string> gReservedIdentifiers = {
    // ANSI / ISO C90
    "auto",
    "break",
    "case",
    "char",
    "const",
    "continue",
    "default",
    "do",
    "double",
    "else",
    "enum",
    "extern",
    "float",
    "for",
    "goto",
    "if",
    "int",
    "long",
    "register",
    "return",
    "short",
    "signed",
    "sizeof",
    "static",
    "struct",
    "switch",
    "typedef",
    "union",
    "unsigned",
    "void",
    "volatile",
    "while",
    // C99
    "_Bool",
    "_Complex",
    "_Imaginary",
    "inline",
    "restrict",
    // OpenCL C built-in vector data types
    "char2",
    "char3",
    "char4",
    "char8",
    "char16",
    "uchar2",
    "uchar3",
    "uchar4",
    "uchar8",
    "uchar16",
    "short2",
    "short3",
    "short4",
    "short8",
    "short16",
    "ushort2",
    "ushort3",
    "ushort4",
    "ushort8",
    "ushort16",
    "int2",
    "int3",
    "int4",
    "int8",
    "int16",
    "uint2",
    "uint3",
    "uint4",
    "uint8",
    "uint16",
    "long2",
    "long3",
    "long4",
    "long8",
    "long16",
    "ulong2",
    "ulong3",
    "ulong4",
    "ulong8",
    "ulong16",
    "float2",
    "float3",
    "float4",
    "float8",
    "float16",
    "double2",
    "double3",
    "double4",
    "double8",
    "double16",
    // OpenCL C other built-in data types
    "image2d_t",
    "image3d_t",
    "image2d_array_t",
    "image1d_t",
    "image1d_buffer_t",
    "image1d_array_t",
    "image2d_depth_t",
    "image2d_array_depth_t",
    "sampler_t",
    "queue_t",
    "ndrange_t",
    "clk_event_t",
    "reserve_id_t",
    "event_t",
    "clk_mem_fence_flags",
    // OpenCL C reserved data types
    "bool2",
    "bool3",
    "bool4",
    "bool8",
    "bool16",
    "half2",
    "half3",
    "half4",
    "half8",
    "half16",
    "quad",
    "quad2",
    "quad3",
    "quad4",
    "quad8",
    "quad16",
    "complex",
    "imaginary",
    // TODO {double,float}nxm
    // OpenCL address space qualifiers
    "__global",
    "global",
    "__local",
    "local",
    "__constant",
    "constant",
    "__private",
    "private",
    "__generic",
    "generic",
    // OpenCL C function qualifiers
    "__kernel",
    "kernel",
    // OpenCL C access qualifiers
    "__read_only",
    "read_only",
    "__write_only",
    "write_only",
    "__read_write",
    "read_write",
    // OpenCL C misc
    "uniform",
    "pipe",
};

bool translator::is_valid_identifier(const std::string& name) const {
  // Check the name isn't already used
  for (auto it = m_names.begin(); it != m_names.end(); ++it) {
    if (it->second == name) {
        return false;
    }
  }

  // Check the name is not a reserved identifier
  return gReservedIdentifiers.count(name) == 0;
}

std::string translator::make_valid_identifier(const std::string& name) const {
  std::string newname = name;

  bool is_valid = is_valid_identifier(newname);
  if (!is_valid) {
    newname += "_MADE_VALID_CLC_IDENT";
  }

  is_valid = is_valid_identifier(newname);

  int name_iter = 1;
  while(!is_valid) {
    std::string candidate = newname + std::to_string(name_iter);
    is_valid = is_valid_identifier(candidate);
    if (!is_valid) {
      name_iter++;
    } else {
      newname = candidate;
      break;
    }
  }

  return newname;
}

bool translator::translate_debug_instructions() {
  // Debug 1
  for (auto &inst : m_ir->module()->debugs1()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case SpvOpSource:
    case SpvOpString:
      break;
    default:
      std::cerr << "UNIMPLEMENTED debug instructions in 7a " << opcode
                << std::endl;
      return false;
    }
  }

  // Debug 2
  for (auto &inst : m_ir->module()->debugs2()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case SpvOpName: {
      auto id = inst.GetSingleWordOperand(0);
      auto name = inst.GetOperand(1).AsString();
      std::replace(name.begin(), name.end(), '.', '_');
      m_names[id] = name;
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED debug instructions " << opcode << ".\n";
      return false;
    }
  }

  // Fixup names to avoid identifiers invalid in OpenCL C
  for (auto &id_name : m_names) {
    auto &id = id_name.first;
    auto &name = id_name.second;
    if (gReservedIdentifiers.count(name)) {
      std::string newname = make_valid_identifier(name);
      m_names[id] = newname;
    }
  }

  // Debug 3
  for (auto &inst : m_ir->module()->debugs3()) {
    std::cerr << "UNIMPLEMENTED debug instructions in 7c.\n";
    return false;
  }

  return true;
}

bool translator::translate_annotations() {
  for (auto &inst : m_ir->module()->annotations()) {
    auto opcode = inst.opcode();
    switch (opcode) {
    case SpvOpDecorate: {
      auto target = inst.GetSingleWordOperand(0);
      auto decoration = inst.GetSingleWordOperand(1);
      switch (decoration) {
      case SpvDecorationFuncParamAttr: {
        auto param_attr = inst.GetSingleWordOperand(2);
        switch (param_attr) {
        case SpvFunctionParameterAttributeNoCapture:
          break;
        case SpvFunctionParameterAttributeNoWrite:
          m_nowrite_params.insert(target);
          break;
        default:
          std::cerr << "UNIMPLEMENTED FuncParamAttr " << param_attr
                    << std::endl;
          return false;
        }
        break;
      }
      case SpvDecorationBuiltIn: {
        auto builtin = inst.GetSingleWordOperand(2);
        switch (builtin) {
        case SpvBuiltInGlobalInvocationId:
        case SpvBuiltInGlobalSize:
        case SpvBuiltInGlobalOffset:
        case SpvBuiltInWorkgroupId:
        case SpvBuiltInWorkgroupSize:
        case SpvBuiltInLocalInvocationId:
        case SpvBuiltInNumWorkgroups:
        case SpvBuiltInWorkDim:
          m_builtin_variables[target] = static_cast<SpvBuiltIn>(builtin);
          break;
        default:
          std::cerr << "UNIMPLEMENTED builtin " << builtin << std::endl;
          return false;
        }
        break;
      }
      case SpvDecorationConstant:
      case SpvDecorationAliased:
        break;
      case SpvDecorationRestrict:
        m_restricts.insert(target);
        break;
      case SpvDecorationVolatile:
        m_volatiles.insert(target);
        break;
      case SpvDecorationCoherent: // TODO anything to do?
        break;
      case SpvDecorationCPacked:
        m_packed.insert(target);
        break;
      case SpvDecorationNonReadable:
      case SpvDecorationNonWritable: // TODO const?
        break;
      case SpvDecorationAlignment: {
        auto align = inst.GetSingleWordOperand(2);
        m_alignments[target] = align;
        break;
      }
      case SpvDecorationLinkageAttributes: {
        auto name = inst.GetOperand(2).AsString();
        auto type = inst.GetSingleWordOperand(3);
        if (type == SpvLinkageTypeExport) {
          m_exports[target] = name;
        }
        if (type == SpvLinkageTypeImport) {
          m_imports[target] = name;
        }
        break;
      }
      case SpvDecorationFPFastMathMode:
        // Ignore for now as that's always correct
        // TODO add a relaxed mode where the whole program is built with fast
        // math
        break;
      case SpvDecorationFPRoundingMode: {
        auto mode = inst.GetSingleWordOperand(2);
        m_rounding_mode_decorations[target] =
            static_cast<SpvFPRoundingMode>(mode);
        break;
      }
      case SpvDecorationSaturatedConversion:
        m_saturated_conversions.insert(target);
        break;
      case SpvDecorationNoSignedWrap:
      case SpvDecorationNoUnsignedWrap:
        break;
      default:
        std::cerr << "UNIMPLEMENTED decoration " << decoration << std::endl;
        return false;
      }
      break;
    }
    case SpvOpDecorationGroup:
      break;
    case SpvOpGroupDecorate: {
      auto group = inst.GetSingleWordOperand(0);
      bool restrict = m_restricts.count(group) != 0;
      bool hasvolatile = m_volatiles.count(group) != 0;
      bool packed = m_packed.count(group) != 0;
      bool nowrite = m_nowrite_params.count(group) != 0;
      bool saturated_conversion = m_saturated_conversions.count(group) != 0;
      bool has_rounding_mode = m_rounding_mode_decorations.count(group) != 0;
      SpvFPRoundingMode rounding_mode;
      if (has_rounding_mode) {
        rounding_mode = m_rounding_mode_decorations.at(group);
      }
      bool has_alignment = m_alignments.count(group) != 0;
      uint32_t alignment;
      if (has_alignment) {
        alignment = m_alignments.at(group);
      }
      for (int i = 1; i < inst.NumOperands(); i++) {
        auto target = inst.GetSingleWordOperand(i);
        if (restrict) {
          m_restricts.insert(target);
        }
        if (hasvolatile) {
          m_volatiles.insert(target);
        }
        if (packed) {
          m_packed.insert(target);
        }
        if (nowrite) {
          m_nowrite_params.insert(target);
        }
        if (saturated_conversion) {
          m_saturated_conversions.insert(target);
        }
        if (has_rounding_mode) {
          m_rounding_mode_decorations[target] = rounding_mode;
        }
        if (has_alignment) {
          m_alignments[target] = alignment;
        }
      }
      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED annotation instruction " << opcode
                << std::endl;
      return false;
    }
  }
  return true;
}

std::string translator::src_pointer_type(uint32_t storage, uint32_t tyid, bool signedty) const {
  std::string typestr;
  if (type_for(tyid)->kind() == Type::Kind::kArray) {
    auto tarray = type_for(tyid)->AsArray();
    auto elemty = tarray->element_type();
    typestr += src_type(type_id_for(elemty));
  } else {
    if (signedty) {
        typestr += src_type_signed(tyid);
    } else {
        typestr += src_type(tyid);
    }
  }
  typestr += " ";
  switch (storage) {
  case SpvStorageClassCrossWorkgroup:
    typestr += "global";
    break;
  case SpvStorageClassUniformConstant:
    typestr += "constant";
    break;
  case SpvStorageClassWorkgroup:
    typestr += "local";
    break;
  case SpvStorageClassInput:
  case SpvStorageClassFunction:
    break;
  default:
    std::cerr << "UNIMPLEMENTED pointer storage class " << storage
              << std::endl;
    return "UNIMPLEMENTED";
  }

  typestr += "*";
  return typestr;
}

bool translator::translate_type(const Instruction &inst) {
  std::string typestr;
  std::string signedtypestr;
  auto opcode = inst.opcode();
  auto result = inst.result_id();
  switch (opcode) {
  case SpvOpTypePointer: {
    auto storage = inst.GetSingleWordOperand(1);
    auto type = inst.GetSingleWordOperand(2);
    if (m_types_signed.count(type)) {
      signedtypestr = src_pointer_type(storage, type, true);
    }
    typestr = src_pointer_type(storage, type, false);
    break;
  }
  case SpvOpTypeInt: {
    auto width = inst.GetSingleWordOperand(1);
    switch (width) {
    case 8:
      typestr = "uchar";
      signedtypestr = "char";
      break;
    case 16:
      typestr = "ushort";
      signedtypestr = "short";
      break;
    case 32:
      typestr = "uint";
      signedtypestr = "int";
      break;
    case 64:
      typestr = "ulong";
      signedtypestr = "long";
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpTypeInt width " << width << std::endl;
      return false;
    }
    break;
  }
  case SpvOpTypeFloat: {
    auto width = inst.GetSingleWordOperand(1);
    switch (width) {
    case 16:
      typestr = "half";
      break;
    case 32:
      typestr = "float";
      break;
    case 64:
      typestr = "double";
      break;
    default:
      std::cerr << "UNIMPLEMENTED OpTypeFloat width " << width << std::endl;
      return false;
    }
    break;
  }
  case SpvOpTypeVector: {
    auto ctype = inst.GetSingleWordOperand(1);
    auto cnum = inst.GetSingleWordOperand(2);
    typestr = src_type(ctype) + std::to_string(cnum);
    signedtypestr = src_type_signed(ctype) + std::to_string(cnum);
    break;
  }
  case SpvOpTypeStruct: { // TODO support volatile members
    // Declare the structure type
    m_src << "struct " + var_for(result) + " {" << std::endl;
    for (uint32_t opidx = 1; opidx < inst.NumOperands(); opidx++) {
      auto mid = inst.GetSingleWordOperand(opidx);
      m_src << "  " << src_var_decl(mid, "m" + std::to_string(opidx - 1)) << ";"
            << std::endl;
    }
    m_src << "}";
    if (m_packed.count(result)) {
      m_src << " __attribute__((packed))";
    }
    m_src << ";" << std::endl;

    // Prepare the type name
    typestr = "struct " + var_for(result);
    break;
  }
  case SpvOpTypeArray: {
    // Handled for pointers in OpTypePointer
    // Variable declarations are special-cased elsewhere
    break;
  }
  case SpvOpTypeImage: {
    auto sampledty = inst.GetSingleWordOperand(1);
    auto dim = inst.GetSingleWordOperand(2);
    auto depth = inst.GetSingleWordOperand(3);
    auto arrayed = inst.GetSingleWordOperand(4);
    auto ms = inst.GetSingleWordOperand(5);
    auto sampled = inst.GetSingleWordOperand(6);
    auto format = inst.GetSingleWordOperand(7);
    auto qual = inst.GetSingleWordOperand(8);

    if ((depth != 0) || (arrayed != 0) || (ms != 0) || (sampled != 0)) {
      std::cerr << "UNIMPLEMENTED image type (depth = " << depth
                << ", arrayed = " << arrayed << ", ms = " << ms
                << "sampled = " << sampled << ")" << std::endl;
      return false;
    }

    switch (qual) {
    case SpvAccessQualifierReadOnly:
      typestr = "read_only";
      break;
    case SpvAccessQualifierWriteOnly:
      typestr = "write_only";
      break;
    case SpvAccessQualifierReadWrite:
      typestr = "read_write";
      break;
    default:
      std::cerr << "UNIMPLEMENTED image access qualifier " << qual << std::endl;
      return false;
    }

    typestr += " ";

    switch (dim) {
    case SpvDim1D:
      typestr += "image1d_t";
      break;
    case SpvDim2D:
      typestr += "image2d_t";
      break;
    case SpvDim3D:
      typestr += "image3d_t";
      break;
    default:
      std::cerr << "UNIMPLEMENTED image dimensionality " << dim << std::endl;
      return false;
    }

    break;
  }
  case SpvOpTypeSampledImage: // TODO anything?
    break;
  case SpvOpTypeSampler:
    typestr = "sampler_t";
    break;
  case SpvOpTypeOpaque: {
    auto name = inst.GetOperand(1).AsString();
    typestr = "struct " + name;
    m_src << typestr << ";" << std::endl;
    break;
  }
  case SpvOpTypeBool:
    typestr = "bool";
    break;
  case SpvOpTypeVoid:
    typestr = "void";
    break;
  case SpvOpTypeFunction: // FIXME
    break;
  case SpvOpTypeEvent:
    typestr = "event_t";
    break;
  default:
    std::cerr << "UNIMPLEMENTED type instuction " << opcode << std::endl;
    return false;
  }

  m_types[result] = typestr;
  if (signedtypestr != "") {
    m_types_signed[result] = signedtypestr;
  }

  return true;
}

bool translator::translate_types_values() {
  for (auto &inst : m_ir->module()->types_values()) {
    auto opcode = inst.opcode();
    auto rtype = inst.type_id();
    auto result = inst.result_id();

    switch (opcode) {
    case SpvOpTypeInt:
    case SpvOpTypeVector:
    case SpvOpTypePointer:
    case SpvOpTypeVoid:
    case SpvOpTypeBool:
    case SpvOpTypeFunction:
    case SpvOpTypeFloat:
    case SpvOpTypeStruct:
    case SpvOpTypeArray:
    case SpvOpTypeOpaque:
    case SpvOpTypeImage:
    case SpvOpTypeSampler:
    case SpvOpTypeSampledImage:
    case SpvOpTypeEvent:
      if (!translate_type(inst)) {
        return false;
      }
      break;

    case SpvOpConstant: {
      auto &op_val = inst.GetOperand(2);
      auto type = type_for(rtype);
      switch (type->kind()) {
      case Type::Kind::kInteger: {
        auto tint = type->AsInteger();
        if (tint->width() <= 32) {
          m_literals[result] = src_cast(rtype, std::to_string(op_val.words[0]));
        } else if (tint->width() == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          m_literals[result] = src_cast(rtype, std::to_string(w));
        } else {
          std::cerr << "UNIMPLEMENTED integer constant width " << tint->width()
                    << std::endl;
          return false;
        }
        break;
      }
      case Type::Kind::kFloat: {
        auto tfloat = type->AsFloat();
        auto width = tfloat->width();
        std::ostringstream out;
        if (width == 16) {
          uint32_t w0 = op_val.words[0];
          cl_half h = w0 & 0xFFFF;
          float val = cl_half_to_float(h);
          out.precision(11);
          out << std::fixed << val << "h";
        } else if (width == 32) {
          uint32_t w0 = op_val.words[0];
          float val = *reinterpret_cast<float *>(&w0);
          if (std::isinf(val)) {
            if (std::signbit(val)) {
              out << "-";
            }
            out << "INFINITY";
          } else if (std::isnan(val)) {
            out << "NAN";
          } else {
            out.precision(24);
            out << std::fixed << val << "f";
          }
        } else if (width == 64) {
          uint64_t w0 = op_val.words[0];
          uint64_t w1 = op_val.words[1];
          auto w = w1 << 32 | w0;
          double val = *reinterpret_cast<double *>(&w);
          if (std::isinf(val)) {
            if (std::signbit(val)) {
              out << "-";
            }
            out << "INFINITY";
          } else if (std::isnan(val)) {
            out << "NAN";
          } else {
            out.precision(53);
            out << std::fixed << val;
          }
        } else {
          std::cerr << "UNIMPLEMENTED float constant width " << width
                    << std::endl;
          return false;
        }
        m_literals[result] = out.str();
        break;
      }
      default:
        std::cerr << "UNIMPLEMENTED OpConstant type " << type->kind()
                  << std::endl;
        return false;
      }
      break;
    }
    case SpvOpUndef:
    case SpvOpConstantNull: {
      std::string cst;
      if (!get_null_constant(rtype, cst)) {
        return false;
      }
      m_literals[result] = cst;
      break;
    }
    case SpvOpConstantTrue: {
      m_literals[result] = "true";
      break;
    }
    case SpvOpConstantFalse: {
      m_literals[result] = "false";
      break;
    }
    case SpvOpConstantSampler: {
      auto addressing_mode = inst.GetSingleWordOperand(2);
      auto normalised = inst.GetSingleWordOperand(3);
      auto filter_mode = inst.GetSingleWordOperand(4);
      m_src << "constant sampler_t " << var_for(result) << " = ";
      switch (addressing_mode) {
      case SpvSamplerAddressingModeClampToEdge:
        m_src << "CLK_ADDRESS_CLAMP_TO_EDGE";
        break;
      case SpvSamplerAddressingModeClamp:
        m_src << "CLK_ADDRESS_CLAMP";
        break;
      case SpvSamplerAddressingModeRepeat:
        m_src << "CLK_ADDRESS_REPEAT";
        break;
      case SpvSamplerAddressingModeRepeatMirrored:
        m_src << "CLK_ADDRESS_MIRRORED_REPEAT";
        break;
      case SpvSamplerAddressingModeNone:
        m_src << "CLK_ADDRESS_NONE";
        break;
      }

      m_src << " | ";

      if (normalised) {
        m_src << "CLK_NORMALIZED_COORDS_TRUE";
      } else {
        m_src << "CLK_NORMALIZED_COORDS_FALSE";
      }

      m_src << " | ";

      switch (filter_mode) {
      case SpvSamplerFilterModeNearest:
        m_src << "CLK_FILTER_NEAREST";
        break;
      case SpvSamplerFilterModeLinear:
        m_src << "CLK_FILTER_LINEAR";
        break;
      }

      m_src << ";" << std::endl;

      break;
    }
    case SpvOpConstantComposite: {
      auto type = type_for(rtype);
      std::string lit;
      switch (type->kind()) {
      case Type::Kind::kVector: {
        auto tvec = type->AsVector();
        // ((type)(c0, c1, ..., cN))
        lit = "((" + src_type(rtype) + ")(";
        const char *sep = "";
        for (uint32_t opidx = 2; opidx < tvec->element_count() + 2; opidx++) {
          auto cid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[cid];
          sep = ", ";
        }
        lit += "))";
        m_literals[result] = lit;
        break;
      }
      case Type::Kind::kStruct: {
        auto tstruct = type->AsStruct();
        // ((type){m0, m1, ..., mN})
        lit = "((" + src_type(rtype) + "){";
        const char *sep = "";
        for (uint32_t opidx = 2; opidx < tstruct->element_types().size() + 2;
             opidx++) {
          auto mid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[mid];
          sep = ", ";
        }
        lit += "})";
        m_literals[result] = lit;
        break;
      }
      case Type::Kind::kArray: {
        lit = "{";
        const char *sep = "";
        uint32_t num_elems = array_type_get_length(rtype);
        if (num_elems == 0) {
            return false;
        }

        for (uint32_t opidx = 2; opidx < num_elems + 2; opidx++) {
          auto mid = inst.GetSingleWordOperand(opidx);
          lit += sep;
          lit += m_literals[mid];
          sep = ", ";
        }
        lit += "}";
        m_literals[result] = lit;
        break;
      }
      default:
        std::cerr << "UNIMPLEMENTED OpConstantComposite type " << type->kind()
                  << std::endl;
        return false;
      }
      break;
    }
    case SpvOpVariable: {
      if (m_builtin_variables.count(result) != 0) {
        break;
      }

      auto tyvar = type_for(rtype);
      auto tykind = tyvar->kind();
      if (tykind != Type::Kind::kPointer) {
        std::cerr << "UNIMPLEMENTED global variable with type " << tykind
                  << std::endl;
        return false;
      }

      auto typtr = tyvar->AsPointer();
      auto tymgr = m_ir->get_type_mgr();
      auto typointeeid = tymgr->GetId(typtr->pointee_type());

      auto storage = inst.GetSingleWordOperand(2);

      if (storage == SpvStorageClassWorkgroup) {
        std::string local_var_decl = "local " + src_type_memory_object_declaration(typointeeid, result);
        m_local_variable_decls[result] = local_var_decl;
      } else if (storage == SpvStorageClassUniformConstant) {
        m_src << "constant "
              << src_type_memory_object_declaration(typointeeid, result);
        if (inst.NumOperands() > 3) {
          auto init = inst.GetSingleWordOperand(3);
          m_src << " = " << var_for(init);
        }
        m_src << ";" << std::endl;
      } else {
        std::cerr << "UNIMPLEMENTED global variable with storage class "
                  << storage << std::endl;
        return false;
      }

      break;
    }
    default:
      std::cerr << "UNIMPLEMENTED type/value instruction " << opcode << ".\n";
      return false;
    }
  }
  return true;
}

bool translator::translate_function(Function &func) {
  auto &dinst = func.DefInst();
  auto rtype = dinst.type_id();
  auto result = dinst.result_id();
  auto control = dinst.GetSingleWordOperand(2);

  bool decl = false;
  bool entrypoint = m_entry_points.count(result) != 0;

  if (m_entry_points_contraction_off.count(result)) {
    m_src << "#pragma OPENCL FP_CONTRACT OFF" << std::endl;
  }

  if (m_imports.count(result)) {
    m_src << "extern ";
    decl = true;
  } else if ((m_exports.count(result) == 0) && !entrypoint) {
    m_src << "static ";
  }

  if (control & SpvFunctionControlInlineMask) {
    m_src << "inline ";
  }

  m_src << src_type(rtype) + " ";
  if (entrypoint) {
    m_src << "kernel ";
    if (m_entry_points_local_size.count(result)) {
      auto &req = m_entry_points_local_size.at(result);
      m_src << "__attribute((reqd_work_group_size(";
      m_src << std::get<0>(req) << "," << std::get<1>(req) << ","
            << std::get<2>(req);
      m_src << "))) ";
    }
    m_src << m_entry_points.at(result);
  } else {
    m_src << var_for(result);
  }
  m_src << "(";
  std::string sep = "";
  func.ForEachParam([this, &sep](const Instruction *inst) {
    auto type = inst->type_id();
    auto result = inst->result_id();
    m_src << sep;
    if (m_nowrite_params.count(result)) {
      m_src << "const ";
    }
    m_src << src_type_memory_object_declaration(type, result);
    sep = ", ";
  });

  m_src << ")";
  if (decl) {
    m_src << ";" << std::endl;
    return true;
  } else {
    m_src << "{" << std::endl;
  }

  // Declare variables in the local address space used by each kernel at the
  // beginning of the kernel function. If the kernel's call tree references
  // a Workgroup variable, paste the declaration we have prepared as part of
  // translating global variables.
  if (entrypoint) {
    std::unordered_set<uint32_t> used_globals_in_local_as;
    IRContext::ProcessFunction process_fn = [this, &used_globals_in_local_as](Function* func) -> bool {
      for (auto &bb : *func) {
        for (auto &inst : bb) {
          for (auto& op : inst) {
            if (spvIsIdType(op.type)) {
              auto used_inst_id = op.AsId();
              auto defuse = m_ir->get_def_use_mgr();
              auto used_inst = defuse->GetDef(used_inst_id);
              if (used_inst->opcode() == SpvOpVariable) {
                if (used_inst->GetSingleWordOperand(2) == SpvStorageClassWorkgroup) {
                  used_globals_in_local_as.insert(used_inst_id);
                }
              }
            }
          }
        }
      }
      return false;
    };
    std::queue<uint32_t> roots;
    roots.push(result);
    m_ir->ProcessCallTreeFromRoots(process_fn, &roots);

    for (auto lvarid : used_globals_in_local_as) {
      m_src << m_local_variable_decls.at(lvarid) << ";\n";
    }
  }

  // First collect information about OpPhi's
  for (auto &bb : func) {
    for (auto &inst : bb) {
      auto rtype = inst.type_id();
      auto result = inst.result_id();
      if (inst.opcode() != SpvOpPhi) {
        continue;
      }
      m_phi_vals[&func].push_back(result);

      for (int i = 2; i < inst.NumOperands(); i += 2) {
        auto var = inst.GetSingleWordOperand(i);
        auto parent = inst.GetSingleWordOperand(i + 1);
        auto parentbb = func.FindBlock(parent);

        m_phi_assigns[&*parentbb].push_back(std::make_pair(result, var));
      }
    }
  }

  // Now translate
  bool error = false;
  if (m_phi_vals.count(&func)) {
    for (auto phival : m_phi_vals.at(&func)) {
      auto phitype = type_id_for(phival);
      m_src << "  " << src_type(phitype) << " " << var_for(phival) << ";\n";
    }
  }
  for (auto &bb : func) {
    m_src << var_for(bb.id()) + ":;" << std::endl;
    // Translate all instructions except the terminator
    for (auto &inst : bb) {
      if (&inst == bb.terminator()) {
        break;
      }
      std::string isrc;
      if (!translate_instruction(inst, isrc)) {
        error = true;
      }
      if (isrc != "") {
        m_src << "  " << isrc << ";\n";
      }
    }
    // Assign phi variables if this block can branch to other blocks with phi
    // refering to this block
    if (m_phi_assigns.count(&bb)) {
      for (auto &phival_var : m_phi_assigns.at(&bb)) {
        m_src << "  " << var_for(phival_var.first) << " = "
              << var_for(phival_var.second) << ";\n";
      }
    }

    // Translate the terminator
    std::string isrc;
    if (!translate_instruction(*bb.ctail(), isrc)) {
      error = true;
    }
    if (isrc != "") {
      m_src << "  " << isrc << ";\n";
    }
  }

  m_src << "}\n";

  if (m_entry_points_contraction_off.count(result)) {
    m_src << "#pragma OPENCL FP_CONTRACT ON" << std::endl;
  }

  return !error;
}

int translator::translate() {

  reset();

  // 1. Capabilities
  if (!translate_capabilities()) {
    return 1;
  }

  // 2. Extensions
  if (!translate_extensions()) {
    return 1;
  }

  // 3. Extended instructions imports
  if (!translate_extended_instructions_imports()) {
    return 1;
  }

  // 4. Memory model
  if (!translate_memory_model()) {
    return 1;
  }

  // 5. Entry point declarations
  if (!translate_entry_points()) {
    return 1;
  }

  // 6. Execution modes
  if (!translate_execution_modes()) {
    return 1;
  }

  // 7. Debug instructions
  if (!translate_debug_instructions()) {
    return 1;
  }

  // 8. Annotations
  if (!translate_annotations()) {
    return 1;
  }

  // 9. Type declarations, constants and global variables
  if (!translate_types_values()) {
    return 1;
  }

  // 10 & 11. Function declarations & definitions
  for (auto &func : *m_ir->module()) {
    if (!translate_function(func)) {
      return 1;
    }
  }

  return 0;
}

bool translator::validate_module(const std::vector<uint32_t> &binary) const {
  spv_diagnostic diag;
  spv_context ctx = spvContextCreate(m_target_env);
  spv_result_t res =
      spvValidateBinary(ctx, binary.data(), binary.size(), &diag);
  spvDiagnosticPrint(diag);
  spvDiagnosticDestroy(diag);
  if (res != SPV_SUCCESS) {
    return false;
  }
  // std::cout << "Module valid.\n";
  return true;
}

int translator::translate(const std::string &assembly, std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer, assembly);

  std::vector<uint32_t> module_bin;
  m_ir->module()->ToBinary(&module_bin, false);
  if (!validate_module(module_bin)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = std::move(m_src.str());
  }

  return ret;
}

int translator::translate(const std::vector<uint32_t> &binary,
                          std::string *srcout) {

  m_ir = BuildModule(m_target_env, spvtools_message_consumer,
                     binary.data(), binary.size());

  if (!validate_module(binary)) {
    return 1;
  }

  int ret = translate();

  if (ret == 0) {
    *srcout = std::move(m_src.str());
  }

  return ret;
}

} // namespace spirv2clc
