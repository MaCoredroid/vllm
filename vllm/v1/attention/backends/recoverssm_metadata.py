# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import abc
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecoverSSMPostprocessMetadata:
    """Metadata used during postprocessing for align-mode prefix caching."""

    num_spec_decodes: int
    request_indices: torch.Tensor | None
    block_table: torch.Tensor
    num_computed_tokens: torch.Tensor
    block_size: int


@dataclass(frozen=True)
class AcceptedPath:
    """Which emitted nodes survived verification, per request.

    Attributes:
        node_ids: int32 ``[num_reqs, max_depth]`` slot ids along each request's
            root-to-accepted-leaf path, right-padded with -1.
        path_lens: int32 ``[num_reqs]`` valid prefix length of ``node_ids``.
        is_linear: True when every row is ``arange(path_lens[i])``, i.e. the
            path is the contiguous window a chain always accepts. Committers
            may then take their existing fast path without gathering.
    """

    node_ids: torch.Tensor
    path_lens: torch.Tensor
    is_linear: bool

    @classmethod
    def linear(cls, num_accepted: torch.Tensor, max_depth: int) -> "AcceptedPath":
        """The path a chain accepts: the first ``num_accepted`` nodes, in order.

        Stays device-resident; ``num_accepted`` is never read on the host.
        """
        depth = torch.arange(max_depth, dtype=torch.int32, device=num_accepted.device)
        node_ids = torch.where(
            depth.unsqueeze(0) < num_accepted.unsqueeze(1), depth, -1
        )
        return cls(
            node_ids=node_ids.to(torch.int32),
            path_lens=num_accepted.to(torch.int32),
            is_linear=True,
        )


class RecoverSSMMetadata(abc.ABC):
    @abc.abstractmethod
    def commit_recoverssm_state(
        self,
        num_accepted_tokens: torch.Tensor,
        accepted_path: AcceptedPath | None = None,
    ) -> RecoverSSMPostprocessMetadata | None:
        """Commit the verified state.

        Args:
            num_accepted_tokens: Per-request accepted-token count.
            accepted_path: Which nodes were accepted. None, or a path with
                ``is_linear``, means the contiguous window implied by
                ``num_accepted_tokens`` -- the only shape a chain produces.
        """
        raise NotImplementedError
