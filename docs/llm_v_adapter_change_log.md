# LLM V-Adapter Change Log

Date: 2026-05-18

## Current Architecture

This branch keeps a single adapter on the Policy side only.

The current data flow is:

- Policy communication path: the 64-d communication vector is passed through `LLMVAdapter`, then blended back with the original vector using `LLM_ADAPTER_ALPHA`.
- Critic / Valuator path: left untouched, with no adapter inserted.

The adapter still computes a reconstruction loss:

```python
rec_loss = ((v_hat - v_old.detach()) ** 2).mean()
```

That loss is used as a Policy-side regularizer and is logged as `policy_llm_rec_loss`.

## What Was Changed In This Round

### Policy-only refactor

- Removed the critic-side adapter wiring from `src/mazebots0/model.py`.
- Kept only the Policy-side adapter path and its alpha blend.
- Kept the critic / valuator path on the original code path.
- Changed the batch payload to store only `policy_llm_rec_loss`.

### Configuration and optimizer cleanup

- Added policy-only config names in `src/mazebots0/config.py`:
  - `USE_POLICY_LLM_ADAPTER`
  - `LLM_ADAPTER_N_TOKENS`
  - `LLM_ADAPTER_OUTPUT_SCALE`
  - `LLM_ADAPTER_ALPHA`
  - `LLM_ADAPTER_LAMBDA_REC`
  - `LLM_MODEL_PATH`
- Kept backward-compatible aliases for older names so the codebase still loads.
- Updated `src/mazebots0/session.py` so the optimizer only creates a separate param group for Policy adapter parameters.
- Left the critic optimizer on its original parameter set.

### Memory fix for the frozen LLaMA bridge

- Changed `src/mazebots0/llm_v_adapter.py` so the bridge only requests hidden states when a non-final layer is explicitly needed.
- When the last layer is requested, the code now uses `last_hidden_state` directly.
- Added micro-batching inside the adapter bridge forward pass to reduce peak GPU memory during PPO updates.

## Files Updated

- `src/mazebots0/llm_v_adapter.py`
  - Bridge forward pass now avoids unnecessary hidden-state materialization.
  - Adapter forward pass now splits large batches into smaller bridge chunks.
- `src/mazebots0/model.py`
  - Policy remains adapter-enabled.
  - Valuator / critic adapter path removed.
  - Batch output keeps only `policy_llm_rec_loss`.
- `src/mazebots0/train.py`
  - Auxiliary logging now tracks only policy adapter reconstruction loss.
  - Total auxiliary penalty uses the policy adapter reconstruction weight only.
- `src/mazebots0/session.py`
  - Added policy-only CLI knobs.
  - Optimizer now creates a separate group only for Policy adapter parameters.
- `src/mazebots0/config.py`
  - Added policy-only names and backward-compatible aliases.

## PPO Impact Check

The PPO core algorithm itself was not modified.

- No changes were made to `discit.marl.MAXPPO`.
- Rollout collection, PPO clipping, value loss, entropy loss, and scheduler logic remain unchanged.
- The adapter still acts as a regularization path on the Policy side only.

## Validation

Verified that:

- Policy-only wiring is active and critic-side adapter wiring is gone.
- `policy_llm_rec_loss` remains 1D and chunkable in the training buffer.
- The adapter bridge runs successfully with micro-batching.
- A Slurm training job can start after the refactor.

## Notes

- The local LLaMA bridge is loaded from the HuggingFace cache when available.
- If the bridge cannot be loaded, the adapter still falls back to the stub bridge for shape and logic testing.
- The current priority is stability and traceability for the first Policy-only version, not maximizing throughput.
