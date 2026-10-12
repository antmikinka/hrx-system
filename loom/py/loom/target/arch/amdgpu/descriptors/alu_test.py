# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Arithmetic descriptor coverage across native AMDGPU families."""

import pytest

from loom.target.arch.amdgpu.descriptors import _REG_SGPR, _REG_VGPR
from loom.target.arch.amdgpu.descriptors.api import (
    _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS,
)
from loom.target.low_descriptors import OperandRole, RegClassAlt


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
@pytest.mark.parametrize("bit_width", [32, 64])
def test_fma_residuals_use_operand_modifiers_on_every_family(target, bit_width) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    for suffix, fields in (
        ("", ("dst", "a", "b", "c")),
        (".neg_a", ("dst", "a", "b", "c")),
        (".neg_a_one", ("dst", "a", "b")),
    ):
        descriptor = overlays[f"amdgpu.v_fma_f{bit_width}{suffix}"]
        assert descriptor.instruction_name == f"V_FMA_F{bit_width}"
        assert descriptor.immediates == ()
        operands = tuple(operand.descriptor_operand for operand in descriptor.operands)
        assert tuple(operand.field_name for operand in operands) == fields
        assert all(operand.unit_count == bit_width // 32 for operand in operands)
        fixed = dict(descriptor.fixed_encoding_fields)
        assert fixed.get("NEG", 0) == (1 if suffix else 0)
        assert ("SRC2" in fixed) == (suffix == ".neg_a_one")


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
@pytest.mark.parametrize("bit_width", [32, 64])
def test_ldexp_descriptors_cover_every_family(target, bit_width: int) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    descriptor = overlays[f"amdgpu.v_ldexp_f{bit_width}"]

    assert descriptor.instruction_name == f"V_LDEXP_F{bit_width}"
    assert descriptor.semantic_tag == f"float.ldexp.f{bit_width}"
    result, input_value, exponent = (
        operand.descriptor_operand for operand in descriptor.operands
    )
    assert result.role is OperandRole.RESULT
    assert result.unit_count == bit_width // 32
    assert input_value.role is OperandRole.OPERAND
    assert input_value.unit_count == bit_width // 32
    assert input_value.reg_alts == (
        RegClassAlt(_REG_SGPR),
        RegClassAlt(_REG_VGPR),
    )
    assert exponent.role is OperandRole.OPERAND
    assert exponent.unit_count == 1
    assert exponent.reg_alts == (
        RegClassAlt(_REG_SGPR),
        RegClassAlt(_REG_VGPR),
    )


@pytest.mark.parametrize("target", _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS)
def test_unsigned_subtract_accepts_an_inline_lhs_on_every_family(target) -> None:
    overlays = {
        overlay.descriptor_key: overlay
        for overlay in _AMDGPU_CORE_DESCRIPTOR_SET_BUILDERS[target].overlay_rows()
    }
    descriptor = overlays["amdgpu.v_sub_u32.src0_inline"]

    assert descriptor.instruction_name == (
        "V_SUB_U32" if target.startswith(("cdna", "gfx9_4")) else "V_SUB_NC_U32"
    )
    assert descriptor.semantic_tag == "integer.sub.u32"
    assert tuple(
        operand.descriptor_operand.field_name for operand in descriptor.operands
    ) == ("dst", "rhs")
    assert descriptor.immediates[0].field_name == "imm32"
    assert descriptor.immediates[0].unsigned_max == 64
