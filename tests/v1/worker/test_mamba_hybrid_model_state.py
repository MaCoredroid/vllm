# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncsByType,
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.platforms import current_platform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.recoverssm_metadata import (
    RecoverSSMMetadata,
    RecoverSSMPostprocessMetadata,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.worker.gpu.model_states import mamba_hybrid
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import MambaSpecDecodeGPUContext


def test_prepare_attn_forwards_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    state = object.__new__(MambaHybridModelState)
    state.vllm_config = SimpleNamespace(num_speculative_tokens=0)
    state.max_model_len = 8192
    state._align_mode = False
    state.recoverssm = None

    positions = torch.tensor([1536], dtype=torch.int64)
    input_batch = SimpleNamespace(
        num_reqs=1,
        num_tokens=1,
        num_reqs_after_padding=1,
        num_tokens_after_padding=1,
        query_start_loc_np=torch.tensor([0, 1], dtype=torch.int32).numpy(),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        num_scheduled_tokens=torch.tensor([1], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([1537], dtype=torch.int32),
        seq_lens=torch.tensor([1537], dtype=torch.int32),
        is_prefilling_np=torch.tensor([False]).numpy(),
        dcp_local_seq_lens=None,
        positions=positions,
        prompt_lens=torch.tensor([1024], dtype=torch.int32),
    )
    expected_metadata = {"layer": object()}
    build_attn_metadata = Mock(return_value=expected_metadata)
    monkeypatch.setattr(mamba_hybrid, "build_attn_metadata", build_attn_metadata)

    metadata = state.prepare_attn(
        input_batch=input_batch,
        cudagraph_mode=CUDAGraphMode.NONE,
        block_tables=(),
        slot_mappings=torch.empty(0, dtype=torch.int64),
        attn_groups=[],
        kv_cache_config=Mock(),
    )

    assert metadata is expected_metadata
    assert build_attn_metadata.call_args.kwargs["positions"] is positions


def test_add_request_seeds_state_idx_in_mamba_blocks() -> None:
    """The align block table is laid out in Mamba blocks, which page unification
    can make larger than cache_config.block_size."""
    mamba_spec = MambaSpec(
        shapes=((1, 1),),
        dtypes=(torch.float32,),
        block_size=880,
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba.0"], mamba_spec)],
    )
    state = object.__new__(MambaHybridModelState)
    state.cache_config = SimpleNamespace(block_size=16)
    state._align_mode = True
    state.rope_state = None
    state.prompt_embeds_state = None
    state.num_accepted_tokens_gpu = torch.ones(2, dtype=torch.int32)
    state._mamba_state_idx_gpu = torch.zeros(2, dtype=torch.int32)
    state.set_kv_cache_config(kv_cache_config)

    state.add_request(1, SimpleNamespace(num_computed_tokens=107_360))

    assert state._mamba_state_idx_gpu[1] == 121


# ---------------------------------------------------------------------------
# Align-mode resume: admission -> preprocess -> restored state bytes
# ---------------------------------------------------------------------------

# Page unification keeps a MambaSpec at its own block size while the engine
# drops cache_config.block_size to the smallest prefix-cacheable group, so a
# seed taken in global units names a different column of the same (Mamba-unit)
# align block table.
_MAMBA_BLOCK_SIZE = 1648
_GLOBAL_BLOCK_SIZE = 816
_NUM_COMPUTED = 3 * _MAMBA_BLOCK_SIZE
# Column a global-unit seed picks here, backed by real storage so the wrong
# restore is silent rather than an illegal access.
_GLOBAL_UNIT_COL = (_NUM_COMPUTED - 1) // _GLOBAL_BLOCK_SIZE

_MAX_NUM_REQS = 4
_REQ_SLOT = 1  # nonzero request slot; batch row 0 maps onto it
_NUM_COLS = 8  # block-table columns, past the global-unit column
_CONV_WIDTH = 4
_CONV_DIM = 16
_SSM_SHAPE = (2, 8)
_PAD_ELEMS = 4  # synthetic per-state block slack, in elements

_ALIGN_COPY_FUNCS: MambaStateCopyFuncsByType = {
    MambaAttentionBackendEnum.MAMBA2: (get_conv_copy_spec, get_temporal_copy_spec),
}


@dataclass
class _StatePool:
    """One state tensor whose blocks sit on a padded stride."""

    view: torch.Tensor  # [num_blocks, *block_shape] view over `storage`
    storage: torch.Tensor  # flat backing storage, padding included
    page_elems: int
    block_elems: int

    def logical_bytes(self) -> torch.Tensor:
        """[num_blocks, block_bytes] byte view, padding excluded."""
        return self.view.reshape(self.view.shape[0], -1).contiguous().view(torch.uint8)

    def padding_bytes(self) -> torch.Tensor:
        """[num_blocks, pad_bytes] byte view of the padding only."""
        return (
            self.storage.view(-1, self.page_elems)[:, self.block_elems :]
            .contiguous()
            .view(torch.uint8)
        )


@dataclass
class _AlignResumeScenario:
    state: MambaHybridModelState
    pools: list[_StatePool]
    pre_logical: list[torch.Tensor]
    pre_padding: list[torch.Tensor]
    block_table: torch.Tensor
    seeded_col: int
    src_col: int
    dst_col: int


def _make_state_pool(
    num_blocks: int,
    block_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> _StatePool:
    """A state pool whose blocks sit a padded stride apart.

    The padding is synthetic per-state slack, not a modelled allocation: it
    exercises the real copy metadata by giving each state a block stride wider
    than its contents, which is what the kernel's "size by inner_size, not by
    block stride" rule exists for. Allocation and page unification are not
    tested here. Block ``b`` carries the distinct finite marker ``b + 1`` plus a
    quarter-step ramp (exact in bf16 and fp32); the padding carries the negated
    marker, so an over-long copy is visible.
    """
    block_elems = math.prod(block_shape)
    page_elems = block_elems + _PAD_ELEMS
    ramp = (torch.arange(page_elems, device=device) % 4).to(dtype) * 0.25
    markers = torch.arange(1, num_blocks + 1, device=device).to(dtype)[:, None]
    pages = markers + ramp[None, :]
    pages[:, block_elems:] *= -1
    storage = pages.reshape(-1)
    inner = tuple(math.prod(block_shape[i + 1 :]) for i in range(len(block_shape)))
    view = torch.as_strided(storage, (num_blocks, *block_shape), (page_elems, *inner))
    return _StatePool(
        view=view, storage=storage, page_elems=page_elems, block_elems=block_elems
    )


def _slot_buffer(fill: int, device: torch.device) -> torch.Tensor:
    """Per-request-slot int32 buffer, as MambaHybridModelState allocates them."""
    return torch.full((_MAX_NUM_REQS,), fill, dtype=torch.int32, device=device)


def _broken_precopy(mode: str) -> Callable[..., None]:
    """Negative-control stand-in for ``run_fused_precopy``: drop the copy, or
    run the real one from the column a global-unit seed would pick."""
    real = MambaSpecDecodeGPUContext.run_fused_precopy

    def hook(self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping):
        if mode == "suppressed":
            return None
        src_col_gpu[_REQ_SLOT] = _GLOBAL_UNIT_COL
        return real(
            self, num_reqs, state_idx_gpu, src_col_gpu, token_bias_gpu, idx_mapping
        )

    return hook


def _run_align_resume_scenario(
    *,
    mamba_block_size: int,
    global_block_size: int,
    precopy_hook: Callable[..., None] | None = None,
) -> _AlignResumeScenario:
    """Admit one request at ``3 * mamba_block_size`` computed tokens, schedule a
    single token, and run the real align preprocess + pre-copy.

    ``set_kv_cache_config`` / ``add_request`` / ``preprocess_state`` run as in
    production, resolving the copy metadata and launching both fused kernels
    over a model-free state pool. ``precopy_hook`` exists only for the negative
    control, which breaks the copy on purpose.
    """
    device = torch.device("cuda")
    num_computed = 3 * mamba_block_size
    num_blocks = _MAX_NUM_REQS * _NUM_COLS + 1
    conv_shape = (
        (_CONV_DIM, _CONV_WIDTH)
        if is_conv_state_dim_first()
        else (_CONV_WIDTH, _CONV_DIM)
    )

    # One layer, two states: what differs is the conv (windowed) versus temporal
    # (whole-block) copy path, not the layer count.
    conv = _make_state_pool(num_blocks, conv_shape, torch.bfloat16, device)
    ssm = _make_state_pool(num_blocks, _SSM_SHAPE, torch.float32, device)
    pools = [conv, ssm]
    forward_context = {"mamba.0": SimpleNamespace(kv_cache=[conv.view, ssm.view])}

    for pool in pools:
        assert torch.isfinite(pool.view).all(), "block markers must be finite"
        blocks = pool.logical_bytes()
        assert blocks.unique(dim=0).shape[0] == num_blocks, (
            "block markers must be distinct, else a misdirected copy is invisible"
        )

    mamba_spec = MambaSpec(
        shapes=(conv_shape, _SSM_SHAPE),
        dtypes=(torch.bfloat16, torch.float32),
        block_size=mamba_block_size,
        mamba_type=MambaAttentionBackendEnum.MAMBA2,
        mamba_cache_mode="align",
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["mamba.0"], mamba_spec)],
    )

    # Non-identity physical mapping: columns run backwards within each row, so
    # reading the wrong column (or a column as if it were a block id) lands on a
    # different block.
    block_table = (
        torch.arange(1, _MAX_NUM_REQS * _NUM_COLS + 1, dtype=torch.int32, device=device)
        .reshape(_MAX_NUM_REQS, _NUM_COLS)
        .flip(1)
        .contiguous()
    )

    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state.max_num_reqs = _MAX_NUM_REQS
    state.device = device
    state.cache_config = SimpleNamespace(
        block_size=global_block_size, mamba_cache_mode="align"
    )
    state.vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(static_forward_context=forward_context)
    )
    state.model = SimpleNamespace(
        get_mamba_state_copy_funcs=lambda _types: _ALIGN_COPY_FUNCS
    )
    state.rope_state = None
    state.prompt_embeds_state = None
    state.recoverssm = None
    # Stale acceptance count from the slot's previous occupant: admission must
    # reset it to 1 so the pre-copy runs with the neutral token bias.
    state.num_accepted_tokens_gpu = _slot_buffer(5, device)
    state._mamba_state_idx_gpu = _slot_buffer(0, device)
    state._mamba_src_col_gpu = _slot_buffer(-1, device)
    state._mamba_src_off_gpu = _slot_buffer(0, device)
    state._mamba_ctx = None
    state._mamba_group_ids = []
    state._mamba_spec = None
    state._mamba_state_copy_funcs = None

    state.set_kv_cache_config(kv_cache_config)
    state.add_request(
        _REQ_SLOT, SimpleNamespace(num_computed_tokens=num_computed, mm_features=[])
    )
    seeded_col = int(state._mamba_state_idx_gpu[_REQ_SLOT])

    input_batch = SimpleNamespace(
        num_reqs=1,
        idx_mapping=torch.tensor([_REQ_SLOT], dtype=torch.int64, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
    )
    num_computed_tokens = _slot_buffer(0, device)
    num_computed_tokens[_REQ_SLOT] = num_computed

    pre_logical = [pool.logical_bytes().clone() for pool in pools]
    pre_padding = [pool.padding_bytes().clone() for pool in pools]

    patched = (
        nullcontext()
        if precopy_hook is None
        else patch.object(MambaSpecDecodeGPUContext, "run_fused_precopy", precopy_hook)
    )
    with patched:
        state.preprocess_state(
            input_batch, (block_table,), kv_cache_config, num_computed_tokens
        )
    torch.cuda.synchronize()

    return _AlignResumeScenario(
        state=state,
        pools=pools,
        pre_logical=pre_logical,
        pre_padding=pre_padding,
        block_table=block_table,
        seeded_col=seeded_col,
        # Expected columns, derived from the invariant rather than read back
        # from the buffers under test.
        src_col=num_computed // mamba_block_size - 1,
        dst_col=cdiv(num_computed + 1, mamba_block_size) - 1,
    )


def _assert_restore_is_bit_identical(scenario: _AlignResumeScenario) -> None:
    """Each pool must equal its pre-step image with exactly the destination
    block's logical bytes replaced by the source block's.

    Byte views, not ``rtol=atol=0``: numerical equality accepts a different bit
    pattern for the same value. Padding is excluded from the content compare and
    checked separately, since the copy is sized by the block contents.
    """
    src_blk = int(scenario.block_table[0, scenario.src_col])
    dst_blk = int(scenario.block_table[0, scenario.dst_col])
    assert src_blk != dst_blk
    for idx, pool in enumerate(scenario.pools):
        pre = scenario.pre_logical[idx]
        expected = pre.clone()
        expected[dst_blk] = pre[src_blk]
        got = pool.logical_bytes()
        assert torch.equal(got, expected), (
            f"state pool {idx}: expected block {src_blk} (column "
            f"{scenario.src_col}) restored into block {dst_blk} (column "
            f"{scenario.dst_col}) with every other block untouched; blocks "
            f"differing from that image: "
            f"{(got != expected).any(dim=1).nonzero().flatten().tolist()}"
        )
        assert torch.equal(pool.padding_bytes(), scenario.pre_padding[idx]), (
            f"state pool {idx}: the pre-copy wrote past the block contents"
        )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(
    ("mamba_block_size", "global_block_size"),
    [
        (_MAMBA_BLOCK_SIZE, _GLOBAL_BLOCK_SIZE),
        (_MAMBA_BLOCK_SIZE, _MAMBA_BLOCK_SIZE),
    ],
    ids=["unequal_geometry", "equal_geometry_control"],
)
def test_align_resume_restores_committed_state_bitwise(
    mamba_block_size: int, global_block_size: int
) -> None:
    """A request admitted at ``3 * M`` computed tokens and then given one token
    must restore column 2 of its block table into column 3, byte for byte.

    The cases differ only in ``cache_config.block_size``; when it equals
    ``MambaSpec.block_size`` both divisors agree, so the equal-geometry case is
    the control that isolates the unequal one.
    """
    scenario = _run_align_resume_scenario(
        mamba_block_size=mamba_block_size, global_block_size=global_block_size
    )

    # Contents first: an index-only failure would stop the test before it ever
    # checked the state bytes.
    _assert_restore_is_bit_identical(scenario)

    assert (scenario.src_col, scenario.dst_col) == (2, 3)
    assert scenario.seeded_col == scenario.src_col
    assert int(scenario.state._mamba_src_col_gpu[_REQ_SLOT]) == scenario.src_col
    assert int(scenario.state._mamba_state_idx_gpu[_REQ_SLOT]) == scenario.dst_col
    assert int(scenario.state._mamba_src_off_gpu[_REQ_SLOT]) == 0
    assert int(scenario.state.num_accepted_tokens_gpu[_REQ_SLOT]) == 1


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize("mode", ["suppressed", "misdirected"])
def test_align_resume_restore_oracle_rejects_broken_precopy(mode: str) -> None:
    """Negative control: the byte oracle must reject a pre-copy that is dropped
    or aimed at the column a global-unit seed would pick."""
    scenario = _run_align_resume_scenario(
        mamba_block_size=_MAMBA_BLOCK_SIZE,
        global_block_size=_GLOBAL_BLOCK_SIZE,
        precopy_hook=_broken_precopy(mode),
    )

    with pytest.raises(AssertionError, match="blocks differing from that image"):
        _assert_restore_is_bit_identical(scenario)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
@pytest.mark.parametrize(("num_sampled", "expected_value"), [(0, 1), (3, 3)])
def test_postprocess_state_scalar_with_int32_mapping(
    num_sampled: int, expected_value: int
) -> None:
    state = object.__new__(MambaHybridModelState)
    state.num_accepted_tokens_gpu = torch.full(
        (4,), 9, dtype=torch.int32, device="cuda"
    )
    state._align_mode = False
    state.recoverssm = None
    state._mamba_ctx = None
    idx_mapping = torch.tensor([2, -1, 0], dtype=torch.int32, device="cuda")

    state.postprocess_state(idx_mapping, num_sampled)

    expected = torch.tensor(
        [expected_value, 9, expected_value, 9], dtype=torch.int32, device="cuda"
    )
    torch.testing.assert_close(state.num_accepted_tokens_gpu, expected)


def test_recoverssm_commits_accepted_window_after_v2_sampling() -> None:
    state = RecoverSSMState()
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = None
    num_sampled = torch.tensor([3, 1], dtype=torch.int32)
    idx_mapping = torch.tensor([0, 1], dtype=torch.int32)
    num_accepted_tokens = torch.ones(2, dtype=torch.int32)
    group = SimpleNamespace(layer_names=["layer"])

    state.record_step({"layer": metadata}, [[group]], for_capture=False)
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )
    state.commit_step(
        num_sampled,
        idx_mapping,
        state_indices=None,
        num_accepted_tokens=num_accepted_tokens,
    )

    metadata.commit_recoverssm_state.assert_called_once_with(num_sampled)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Requires CUDA")
def test_recoverssm_align_tracks_mixed_batch_state_and_neutralizes_copy_bias() -> None:
    state = object.__new__(MambaHybridModelState)
    state._align_mode = True
    state._mamba_ctx = None
    state._mamba_state_idx_gpu = torch.full((5,), -1, dtype=torch.int32, device="cuda")
    state.recoverssm = RecoverSSMState()
    state.num_accepted_tokens_gpu = torch.full(
        (5,), 9, dtype=torch.int32, device="cuda"
    )
    metadata = Mock(spec=RecoverSSMMetadata)
    metadata.commit_recoverssm_state.return_value = RecoverSSMPostprocessMetadata(
        num_spec_decodes=1,
        request_indices=torch.tensor([1], dtype=torch.int32, device="cuda"),
        num_computed_tokens=torch.tensor([6, 7], dtype=torch.int32, device="cuda"),
        block_size=8,
        block_table=torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
    )
    num_sampled = torch.tensor([2, 3], dtype=torch.int32, device="cuda")
    idx_mapping = torch.tensor([3, 1], dtype=torch.int32, device="cuda")
    group = SimpleNamespace(layer_names=["layer"])

    state.recoverssm.record_step({"layer": metadata}, [[group]], for_capture=False)

    state.postprocess_state(idx_mapping, num_sampled)

    expected_state_indices = [-1, 1, -1, -1, -1]
    assert state._mamba_state_idx_gpu.tolist() == expected_state_indices
    expected_accepted = [9, 1, 9, 2, 9]
    assert state.num_accepted_tokens_gpu.tolist() == expected_accepted
