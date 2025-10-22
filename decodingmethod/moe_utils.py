import torch
from torch.nn import functional as F
from transformers.generation.stopping_criteria import (
    StoppingCriteriaList,
)
import numpy as np


def calculate_jsd(teacher_probs, student_probs, eps: float = 1e-9):
    """
    Calculate the Jensen–Shannon divergence between two probability
    distributions using a clamp-based stabilization (no renormalization).

    Notes
    - Inputs are assumed to be probabilities (e.g., softmaxed logits) on the
      last dimension.
    - We clamp P, Q, and their mixture M to avoid log(0) and numerical issues
      in low-probability tails, following common practice in decoding work.
    - Using natural logarithms implies JSD is bounded in [0, ln 2].

    Args:
        teacher_probs: Tensor of teacher probabilities [..., V]
        student_probs: Tensor of student probabilities [..., V]
        eps: Small floor to avoid log(0) and NaNs in KL
    """
    
    # Clamp to ensure strictly positive probabilities without changing mass
    # distribution via renormalization. This mirrors AdaCAD-style safety.
    teacher_probs = teacher_probs.clamp_min(eps)
    student_probs = student_probs.clamp_min(eps)

    # Mixture distribution M = 0.5(P + Q), also clamped for numerical safety.
    m = (0.5 * (teacher_probs + student_probs)).clamp_min(eps)

    # KL(P || M) and KL(Q || M), computed with log-target formulation.
    # F.kl_div expects first arg as log-probs when log_target=False.
    kl_pm = F.kl_div(m.log(), teacher_probs, reduction="none", log_target=False).sum(dim=-1)
    kl_qm = F.kl_div(m.log(), student_probs, reduction="none", log_target=False).sum(dim=-1)

    # JSD = 0.5 * (KL(P||M) + KL(Q||M))
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


# NOTE: Previously we mapped JSD to beta via a thresholded linear scaler.
# We now use the raw JSD (in [0, ln 2]) directly as beta for simplicity and
# to avoid extra sensitivity hyperparameters.

@torch.no_grad()
def scmoe(
    model,
    teacher_t,
    student_t,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    alpha=0.1,
    beta=0.5,
    stopping_criteria=None,
    teacher_routed_tok=[0, 1],
    teacher_num_experts_per_tok=2,
    student_routed_tok=[0],
    student_num_experts_per_tok=1,
    dynamic_beta=False,
):

    batch_size, prefix_len = input_ids.size()
    # Prepare generation kwargs with cache_position for new HF caching API
    model_kwargs = {"attention_mask": attention_mask, "use_cache": True}
    try:
        model_kwargs = model._prepare_model_kwargs_for_generation(input_ids, model_kwargs)
    except AttributeError:
        seq_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)

    model_kwargs_student = {"attention_mask": attention_mask, "use_cache": True}
    try:
        model_kwargs_student = model._prepare_model_kwargs_for_generation(input_ids, model_kwargs_student)
    except AttributeError:
        seq_len = input_ids.shape[-1]
        model_kwargs_student["cache_position"] = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=teacher_routed_tok,
            num_experts_per_tok=teacher_num_experts_per_tok,
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = model(
            **model_inputs_student,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=student_routed_tok,
            num_experts_per_tok=student_num_experts_per_tok,
        )
        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        # Dynamic beta scaling based on JSD between teacher and student distributions
        if dynamic_beta:
            # Convert logits to probabilities for JSD calculation
            teacher_probs = F.softmax(next_token_scores, dim=-1)
            student_probs = F.softmax(next_token_logits_student, dim=-1)
            
            # Calculate Jensen-Shannon Divergence
            jsd_score = calculate_jsd(teacher_probs, student_probs)
            
            # Use raw JSD as beta (natural logs → beta in [0, ln 2]) with a lower bound.
            # Rationale: avoid under-contrasting on very low-conflict yet error-prone steps.
            beta_min_threshold = 0.25  # TODO: expose via CLI if helpful
            beta_scalar = torch.maximum(
                jsd_score,
                torch.as_tensor(beta_min_threshold, device=jsd_score.device, dtype=jsd_score.dtype),
            )
            # Broadcast beta over vocab dimension once.
            beta_dynamic = beta_scalar.unsqueeze(-1)
            
            # Apply dynamic beta in contrastive formula
            diffs = (1 + beta_dynamic) * next_token_scores - beta_dynamic * next_token_logits_student
        else:
            # Use fixed beta (original SCMoE behavior)
            diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        
        # Calculate and print Shannon entropy of teacher distribution
        teacher_probs = F.softmax(next_token_scores, dim=-1)
        eps = 1e-9
        vocab_size = teacher_probs.shape[-1]
        log_vocab_size = torch.log(torch.tensor(float(vocab_size), device=teacher_probs.device))
        # Normalized entropy in [0, 1] range
        entropy = -(teacher_probs * teacher_probs.clamp_min(eps).log()).sum(dim=-1) / log_vocab_size
        
        # Print entropy for each sample in batch
        for i in range(batch_size):
            print(f"Step {step} Sample {i}: entropy={entropy[i].item():.4f}")
            
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")

        next_tokens = torch.argmax(cdlogits, dim=-1)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        
        
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        model_kwargs_student = model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
    return input_ids

@torch.no_grad()
def scmoe_with_sampling(
    model,
    teacher_t,
    student_t,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    alpha=0.1,
    beta=0.5,
    stopping_criteria=None,
    teacher_routed_tok=[0, 1],
    teacher_num_experts_per_tok=2,
    student_routed_tok=[0],
    student_num_experts_per_tok=1,
):

    batch_size, prefix_len = input_ids.size()
    model_kwargs = {}
    model_kwargs_student = {}
    model_kwargs["attention_mask"] = attention_mask
    model_kwargs_student["attention_mask"] = attention_mask
    eos_token_id = eos_token_id if eos_token_id is not None else tokenizer.eos_token_id
    eos_token_id_tensor = (
        torch.tensor([eos_token_id]).to(model.device)
        if eos_token_id is not None
        else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = (
        stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
    )
    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=teacher_routed_tok,
            num_experts_per_tok=teacher_num_experts_per_tok,
        )
        next_token_scores = outputs.logits[:, -1, :]
        next_token_scores = next_token_scores / teacher_t
        cutoff = (
            torch.log(torch.tensor(alpha, device=next_token_scores.device))
            + next_token_scores.max(dim=-1, keepdim=True).values
        )

        model_inputs_student = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs_student
        )
        outputs_student = model(
            **model_inputs_student,
            return_dict=True,
            output_hidden_states=False,
            routed_tok=student_routed_tok,
            num_experts_per_tok=student_num_experts_per_tok,
        )
        next_token_logits_student = outputs_student.logits[:, -1, :]
        next_token_logits_student = next_token_logits_student / student_t
        diffs = (1 + beta) * next_token_scores - beta * next_token_logits_student
        cdlogits = diffs.masked_fill(next_token_scores < cutoff, -float("inf"))
        if not early_stop and eos_token_id != None:
            cdlogits[:, eos_token_id] = -float("inf")
        
        cdscores = F.softmax(cdlogits, dim=-1)
        next_tokens = torch.multinomial(cdscores, num_samples=1).squeeze(-1)
        next_tokens = next_tokens * unfinished_sequences + tokenizer.pad_token_id * (
            1 - unfinished_sequences
        )
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        unfinished_sequences = unfinished_sequences & ~stopping_criteria(
            input_ids, None
        )
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            stopped = True
        else:
            stopped = False

        if stopped:
            break

        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
        model_kwargs_student = model._update_model_kwargs_for_generation(
            outputs_student,
            model_kwargs_student,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )
    return input_ids


@torch.no_grad()
def onepass_greedy(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    max_new_tokens,
    eos_token_id=None,
    early_stop=False,
    stopping_criteria=None,
    use_onepass_lc: bool = True,
):
    """
    Single-pass greedy decoding loop that calls model(...) directly each step,
    passing use_onepass_lc to enable layerwise contrast inside the model.

    Notes
    - Uses HF caching utilities to mirror generate() semantics.
    - Works for Mixtral (LC-enabled) and is a no-op flag for other models.
    """
    batch_size, prefix_len = input_ids.size()

    # Prepare model kwargs and cache positions for first step
    model_kwargs = {"attention_mask": attention_mask, "use_cache": True}
    try:
        model_kwargs = model._prepare_model_kwargs_for_generation(input_ids, model_kwargs)
    except AttributeError:
        seq_len = input_ids.shape[-1]
        model_kwargs["cache_position"] = torch.arange(seq_len, device=input_ids.device, dtype=torch.long)

    eos_token_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
    eos_token_id_tensor = (
        torch.tensor([eos_token_id], device=input_ids.device) if eos_token_id is not None else None
    )
    unfinished_sequences = input_ids.new(batch_size).fill_(1)
    stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    for step in range(max_new_tokens):
        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
        # Ensure flag persists even if prepare_inputs doesn't copy it through
        model_inputs["use_onepass_lc"] = use_onepass_lc

        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=False,
        )

        next_token_scores = outputs.logits[:, -1, :]
        # Greedy
        next_tokens = torch.argmax(next_token_scores, dim=-1)

        # Respect EOS handling
        if not early_stop and eos_token_id is not None:
            # prevent picking EOS unless criteria say stop
            pass  # greedy picking already done; rely on criteria to stop

        # Mask out finished sequences with pad
        pad_id = getattr(tokenizer, "pad_token_id", eos_token_id if eos_token_id is not None else 0)
        next_tokens = next_tokens * unfinished_sequences + pad_id * (1 - unfinished_sequences)

        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

        if eos_token_id_tensor is not None:
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1)
                .ne(eos_token_id_tensor.unsqueeze(1))
                .prod(dim=0)
            )

        # External stopping criteria
        stop_now = stopping_criteria(input_ids, None)
        unfinished_sequences = unfinished_sequences & ~stop_now
        if unfinished_sequences.max() == 0 or step == max_new_tokens - 1:
            break

        # Update caches
        model_kwargs = model._update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,
        )

    return input_ids