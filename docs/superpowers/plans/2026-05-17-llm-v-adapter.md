# LLM V-Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate an online LLM V-Adapter into MazeBots0 so policy, critic, and auxiliary belief training can all run with adapted communication vectors.

**Architecture:** Add a reusable adapter module that maps 64-d communication vectors to LLaMA-sized token blocks, passes them through a frozen local LLM bridge when available, and maps them back to 64-d. Wire one adapter instance into the policy message path and one into the critic/value path so each branch can train with its own optimizer. Keep PPO logic in the external library unchanged; only expose adapter parameters and auxiliary reconstruction losses through the model/session layer.

**Tech Stack:** PyTorch, HuggingFace Transformers, MazeBots0 existing PPO stack, local HuggingFace cache.

---

### Task 1: Add reusable adapter module

**Files:**
- Create: `src/mazebots0/llm_v_adapter.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from llm_v_adapter import LLMVAdapter


def test_adapter_preserves_shape_without_llm():
    adapter = LLMVAdapter(model_path=None, bridge_mode="stub", n_tokens=8)
    v = torch.randn(2, 112, 64)
    v_hat, aux = adapter(v)
    assert v_hat.shape == (2, 112, 64)
    assert "rec_loss" in aux
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q` or `python - <<'PY' ... PY`
Expected: import or attribute error before implementation exists.

- [ ] **Step 3: Write minimal implementation**

Implement `UpMLP`, `DownMLP`, and `LLMVAdapter` with `bridge_mode="stub"` and optional local LLaMA loading.

- [ ] **Step 4: Run test to verify it passes**

Run: `python - <<'PY'
import torch
from src.mazebots0.llm_v_adapter import LLMVAdapter
adapter = LLMVAdapter(model_path=None, bridge_mode='stub', n_tokens=8)
v = torch.randn(2, 112, 64)
v_hat, aux = adapter(v)
print(v_hat.shape, sorted(aux.keys()))
PY`
Expected: `(2, 112, 64)` and `['rec_loss', ...]`.

### Task 2: Wire policy and critic branches

**Files:**
- Modify: `src/mazebots0/model.py`

- [ ] **Step 1: Write the failing test**

```python
import torch
from model import ActorCritic


def test_actorcritic_returns_mem_and_adapter_ready():
    model = ActorCritic(n_envs=1, n_bots=2, com_mode=1)
    mem = model.init_mem()
    obs = (torch.randn(2, 4, 96, 48), torch.randn(2, 108), torch.randn(1, 20, 20))
    out = model.forward(obs, mem, (torch.zeros(2, 1),))
    assert "mem" in out
```

- [ ] **Step 2: Run test to verify it fails**

Expected: missing adapter fields or shape mismatch before wiring.

- [ ] **Step 3: Write minimal implementation**

Add policy-side adapter application to the communication output, add critic-side adapter application to the communication slice of `xv`, and propagate auxiliary reconstruction losses through module outputs.

- [ ] **Step 4: Run test to verify it passes**

Run the same import-and-forward smoke test and confirm output shapes remain stable.

### Task 3: Add config knobs and session wiring

**Files:**
- Modify: `src/mazebots0/config.py`
- Modify: `src/mazebots0/session.py`
- Modify: `src/mazebots0/train.py`

- [ ] **Step 1: Write the failing test**

```python
import config as cfg


def test_llm_config_defaults_exist():
    assert cfg.USE_LLM_V_ADAPTER in (0, 1)
    assert cfg.LLM_OUTPUT_SCALE > 0
```

- [ ] **Step 2: Run test to verify it fails**

Expected: missing config names before implementation.

- [ ] **Step 3: Write minimal implementation**

Add `USE_LLM_V_ADAPTER`, `LLM_MODEL_PATH`, `LLM_N_TOKENS`, `LLM_HIDDEN_LAYER`, `LLM_OUTPUT_SCALE`, `LLM_ADAPTER_ALPHA`, and `LLM_REC_COEF`. Instantiate adapter modules in `ActorCritic` from config and pass reconstruction loss logging through `train.py`/`session.py`.

- [ ] **Step 4: Run test to verify it passes**

Run the smoke test and a targeted import of `session.py`.

### Task 4: Validate end-to-end behavior

**Files:**
- Test: `python` smoke script using the new adapter module and `model.py`

- [ ] **Step 1: Run shape smoke test**

Create a small Python snippet that instantiates the adapter in stub mode and runs policy/value forwards.

- [ ] **Step 2: Check gradient flow**

Backprop through the stub adapter and verify adapter parameters receive gradients.

- [ ] **Step 3: Document any runtime caveats**

If local LLaMA weights are unavailable or too large for the current machine, keep the stub bridge available for functional testing while preserving the real bridge path for the training machine.
