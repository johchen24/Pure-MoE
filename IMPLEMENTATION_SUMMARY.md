# Weight-Space Contrastive Expert Synthesis Implementation for GPT-OSS

## Overview
Implemented conditional weight-space contrast for GPT-OSS MoE model, creating synthetic experts by contrasting strong (top-K weighted average) and weak (rank-4) expert weights when routing entropy is high.

## Changes Made

### 1. `modeling_models/gpt-oss/modeling_gpt_oss.py`

#### Modified `GptOssMLP.forward()` (Lines 168-319)
- **Added parameter**: `use_onepass_lc` flag
- **Entropy calculation**: Shannon entropy normalized by log(num_experts)
- **Conditional activation**: Only applies contrast when:
  - `use_onepass_lc=True`
  - `sequence_length==1` (single token generation)
  - `entropy >= ENTROPY_TAU` (0.85 threshold)
  - Not training mode

#### Weight Synthesis Logic
1. **Strong experts**: Top-K experts weighted by softmax routing probabilities
2. **Weak expert**: Rank-4 expert (index 3 in sorted order)
3. **Contrast formula**: `W_synth = W_strong + BETA * (W_strong - W_weak)`
4. **Norm preservation**: Rescale to match strong expert's Frobenius norm
5. **All 4 parameters contrasted**:
   - `gate_up_proj` (fused gate/up projection)
   - `gate_up_proj_bias`
   - `down_proj`
   - `down_proj_bias`

#### Forward Pass Through Synthetic Expert
- Replicates `GptOssExperts` logic with custom weights
- Applies GPT-OSS specific activations:
  - Split fused gate_up projection
  - Clamp: gate (max=7.0), up (-7.0 to 7.0)
  - GLU activation: `gate * sigmoid(gate * 1.702)`
  - Output: `(up + 1) * glu @ down_proj`

#### Modified `GptOssDecoderLayer.forward()` (Lines 533-534)
- Extracts `use_onepass_lc` from kwargs
- Passes flag to MLP layer

#### Modified `GptOssModel.forward()` (Lines 618, 655-672)
- Added `use_onepass_lc` parameter
- **Layer-wise gating**: Only enables contrast for second half of layers (layers >= num_hidden_layers // 2)
- Rationale: Early layers learn general features, later layers handle reasoning

#### Modified `GptOssForCausalLM.forward()` (Lines 788, 828)
- Added `use_onepass_lc` parameter
- Passes flag to model

### 2. Integration with Existing Code

#### `decodingmethod/moe_utils.py`
- **Already implemented**: `onepass_greedy()` passes `use_onepass_lc=True` (line 359)
- No changes needed

#### `generate.py`
**Modified for GPT-OSS Harmony Format Support** (Lines 241-249, 342-347, 388-417):
- **Detection**: Automatically detects GPT-OSS models by checking model path for "gpt" and "oss"
- **Harmony Chat Template** (Lines 388-417):
  - Applies `tokenizer.apply_chat_template()` with reasoning level set in **system message**: `"Reasoning: medium"`
  - Required per [GPT-OSS model card](https://huggingface.co/openai/gpt-oss-20b): "must only be used with the harmony format"
  - Reasoning level is set as system message content, not as a parameter (per Hugging Face documentation)
  - Shows first formatted prompt for verification
- **Stopping Criteria** (Lines 342-347):
  - Relies on model's natural EOS tokens (`<|return|>` and `<|endoftext|>`)
  - No explicit stopping criteria needed for GPT-OSS
- **Logging** (Lines 241-249):
  - Prints notification when GPT-OSS is detected
  - Confirms harmony format will be applied with reasoning level

## Constants (Hardcoded)

```python
BETA = 0.5              # Contrast strength for weight interpolation
WEAK_RANK = 4           # 1-based rank → index 3 (rank-4 expert)
ENTROPY_TAU = 0.85      # Entropy threshold (normalized [0,1])
EPS = 1e-6              # Numerical stability
DEBUG_LC = True         # Enable debug prints
```

## Usage

```bash
python generate.py \
    --decoding_method onepass \
    --model_name_or_path <path-to-gpt-oss-model> \
    --infile <input-file> \
    --outfile <output-file> \
    --batch_size 1 \
    --max_new_tokens 256
```

**Important for GPT-OSS Models:**
- The script automatically detects GPT-OSS models (checks for "gpt" and "oss" in model path)
- Harmony chat template is applied automatically with system message `"Reasoning: medium"` (hardcoded)
- Reasoning level is set as system message content, per Hugging Face documentation
- No additional flags needed - it just works!
- First prompt will be printed to verify harmony format is applied correctly

## Debug Output

When entropy >= threshold, prints:
- Per-token entropy and routing probabilities
- Strong/weak expert selections
- Weight statistics (norms, rescaling factors)

## Key Differences from Mixtral Implementation

1. **Fused gate_up_proj**: GPT-OSS combines gate and up projections
2. **Weighted averaging**: Uses softmax routing weights for strong experts
3. **Biases**: Contrasts biases alongside weight matrices
4. **Activation**: GLU with custom alpha=1.702 instead of SiLU
5. **Clamping**: Asymmetric clamping on gate/up projections

## Testing Recommendations

1. **Verify forward pass**: Check outputs match expected shapes
2. **Compare with normal routing**: Test entropy < threshold fallback
3. **Debug prints**: Monitor entropy values and expert selections
4. **Performance**: Measure accuracy on reasoning benchmarks (GSM8K, etc.)

## Notes

- Implementation follows SCMoE paper principles (https://arxiv.org/html/2405.14507v2)
- Norm preservation prevents activation explosion
- **Layer-wise gating**: Contrast applied only to second half of layers (improves coherence)
- Designed for batch_size=1 autoregressive generation

