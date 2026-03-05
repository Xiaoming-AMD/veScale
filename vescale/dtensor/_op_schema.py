################################################################################
# Copyright (c) Meta Platforms, Inc. and affiliates
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2023 ByteDance Ltd. and/or its affiliates.
################################################################################
from torch.distributed.tensor._op_schema import (
    OpInfo,
    OpSchema,
    OpStrategy,
    OutputSharding,
    OutputSpecType,
    OpSpec as PlacementStrategy,
    TupleStrategy,
    PlacementList,
    RuntimeSchemaInfo,
    StrategyType,
)


def _is_inplace_op(op) -> bool:
    """Compat shim: in torch>=2.8 these became methods on OpSchema; keep as standalone for veScale."""
    return op._schema.name[-1] == "_"


def _is_out_variant_op(op) -> bool:
    """Compat shim: in torch>=2.8 these became methods on OpSchema; keep as standalone for veScale."""
    return "out" in op._schema.overload_name


__all__ = [
    "OpInfo",
    "_is_inplace_op",
    "_is_out_variant_op",
    "OpSchema",
    "OpStrategy",
    "OutputSharding",
    "OutputSpecType",
    "PlacementStrategy",
    "TupleStrategy",
    "PlacementList",
    "RuntimeSchemaInfo",
    "StrategyType",
]
