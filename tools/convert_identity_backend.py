# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Convert the identity backend of a full-body SOMA NPZ animation."""

import argparse
import logging
from pathlib import Path

import torch

from soma.body import SOMALayer
from soma.units import Unit
from tools.identity_conversion import convert_soma_npz, model_kwargs
from tools.logging_utils import add_logging_args, configure_logging

logger = logging.getLogger(__name__)

BODY_BACKENDS = ("soma", "mhr", "anny", "smpl", "smplh", "smplx", "garment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert identity parameters in a full-body SOMA NPZ via bind-pose fitting."
    )
    parser.add_argument("input", type=Path, help="Input SOMA NPZ from soma.io.save_soma_npz.")
    parser.add_argument("output", type=Path, help="Output SOMA NPZ with converted identity.")
    parser.add_argument("--target-backend", required=True, choices=BODY_BACKENDS)
    parser.add_argument("--data-root", type=Path, default=Path("assets"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--lod", choices=("mid", "low", "xlo"), default="low")
    parser.add_argument("--source-model-path", default=None)
    parser.add_argument("--target-model-path", default=None)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-4)
    parser.add_argument(
        "--no-optimize-scale-params",
        action="store_true",
        help="Keep target scale parameters neutral. Native SOMA bone scales are optimized by default.",
    )
    parser.add_argument(
        "--optimize-global-scale",
        action="store_true",
        help="Optimize a target global scale. By default the input global scale is fixed.",
    )
    add_logging_args(parser)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args)

    def make_layer(backend: str, unit: str, role: str) -> SOMALayer:
        selected_path = args.source_model_path if role == "source" else args.target_model_path
        return SOMALayer(
            data_root=args.data_root,
            device=args.device,
            identity_model_type=backend,
            identity_model_kwargs=model_kwargs(selected_path),
            lod=args.lod,
            output_unit=Unit.from_name(unit),
            correctives_model_path=None,
        )

    result = convert_soma_npz(
        args.input,
        args.output,
        target_backend=args.target_backend,
        layer_factory=make_layer,
        optimize_scale_params=not args.no_optimize_scale_params,
        optimize_global_scale=args.optimize_global_scale,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
    )
    logger.info("Output: %s", args.output)
    logger.info("Mean bind-pose vertex error: %.6f", float(result.vertex_error.mean()))


if __name__ == "__main__":
    main()
