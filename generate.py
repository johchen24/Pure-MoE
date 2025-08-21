import sys
import transformers
print(transformers.__version__)
import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import os
import time
from tqdm import tqdm
from decodingmethod import (
    dola,
    contrastive_decoding,
    scmoe,
    scmoe_with_sampling,
)
from multiprocessing import Process
import numpy as np
import pandas as pd
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    add_start_docstrings,
)
#New added
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence
from itertools import combinations


def generate_combinations(total_length, num_to_remove):
    indices = range(total_length)
    result = list(combinations(indices, num_to_remove))
    return [list(i) for i in result]
    
#Added stopping text criteria
class StopAfterAnswerWindowStrict(StoppingCriteria):
    """
    Start checking only AFTER we see an 'answer phrase' (e.g., 'the answer is').
    Then allow up to `window_words` words; if a stop string (e.g., '\n\nQuestion:')
    appears within that window, stop immediately.
    If the phrase never appears, NEVER stop (no fallback).
    """
    def __init__(self, tokenizer, stop_strings, start_pos: int,
                 answer_phrase_regex: str = r"(?i)\b(the\s+answer\s+is|answer\s*[:=])\b",
                 warmup_tokens: int = 8,
                 window_words: int = 20,
                 check_every: int = 2):
        self.tok = tokenizer
        self.stop_strings = stop_strings
        self.start_pos = start_pos
        self.answer_rx = re.compile(answer_phrase_regex)
        self.warmup_tokens = warmup_tokens
        self.window_words = window_words
        self.check_every = check_every
        self._step = 0
        self._anchor_idx = None  # char offset in decoded tail where phrase ends

    def _count_words(self, s: str) -> int:
        return len(re.findall(r"\b\w+\b", s))

    def __call__(self, input_ids, scores, **kwargs):
        self._step += 1
        new_tokens = input_ids.shape[1] - self.start_pos
        if new_tokens < self.warmup_tokens or (self._step % self.check_every):
            return False

        tail = self.tok.decode(
            input_ids[0, self.start_pos:].tolist(),
            skip_special_tokens=False
        )

        # Anchor on "the answer is ..."
        if self._anchor_idx is None:
            m = self.answer_rx.search(tail)
            if not m:
                # STRICT: do NOT stop before the phrase appears
                return False
            self._anchor_idx = m.end()

        # After the phrase: enforce the window
        after = tail[self._anchor_idx:]

        # If a stop string appears within the window_words → stop now
        earliest_stop = None
        for s in self.stop_strings:
            i = after.find(s)
            if i != -1 and (earliest_stop is None or i < earliest_stop):
                earliest_stop = i

        if earliest_stop is not None:
            words_before_stop = self._count_words(after[:earliest_stop])
            if words_before_stop <= self.window_words:
                return True

        # Otherwise, never stop early (let max_new_tokens/EOS handle it)
        return False
#-------------------------------------------------

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list):
        self.token_id_list = token_id_list
        self.stop_tag = None

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for _ in range(len(self.token_id_list)):
            stop_states = [
                np.array_equal(
                    self.token_id_list[_],
                    input_ids[i][-len(self.token_id_list[_]) :].detach().cpu().numpy(),
                )
                for i in range(input_ids.size(0))
            ]
            if self.stop_tag is None:
                self.stop_tag = stop_states
            else:
                self.stop_tag = [
                    self.stop_tag[i] or stop_states[i]
                    for i in range(len(self.stop_tag))
                ]
            if all(self.stop_tag):
                self.stop_tag = None
                return True
        return False


def args_parse():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--infile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument(
        "--outfile", type=str, help="the data used for instructing tuning"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--decoding_method", type=str, default="topp")
    parser.add_argument("--gpus_per_model", type=int, default=2)
    parser.add_argument(
        "--model_name_or_path", default="", type=str
    )

    parser.add_argument(
        "--student_model_name_or_path",
        default="",
        type=str,
    )

    parser.add_argument("--cs_alpha", default=0.6, type=float)
    parser.add_argument("--cs_k", default=5, type=int)

    parser.add_argument(
        "--cd_alpha", default=0.1, type=float
    )  
    parser.add_argument("--cd_beta", default=0.5, type=float)
    parser.add_argument(
        "--cd_tt", default=1.0, type=float, help="teacher temperature"
    )  
    parser.add_argument("--cd_st", default=1.0, type=float, help="student temperature")

    # Adaptive beta (AdaCAD-style) for SCMoE
    parser.add_argument("--dynamic_beta", action="store_true")

    parser.add_argument("--dola_early_exit_layers", default="0,2,4,6,8,10,12,14,32", type=str)
    parser.add_argument("--dola_mature_layer", default=32, type=int)

    parser.add_argument("--num_experts_per_tok", default=2, type=int)
    parser.add_argument("--routed_tok", default=0, type=int)
    parser.add_argument("--student_num_experts_per_tok", default=1, type=int)
    parser.add_argument("--student_routed_tok", default=0, type=int)

    parser.add_argument("--dynamic_expert_routing_threshold", default=0.6, type=float)
    
    #Added stopping text criteria
    parser.add_argument(
    "--stop_text",
    type=str,
    default="",
    help="Comma-separated stop substrings (e.g., '\\n\\nQuestion:,\\nQuestion:,Q:')",
    )
    #--------------------------------------------------------------------------------------

    #Added argument to resume answering questions in the benchmark testing in-case the run gets interupted.
    parser.add_argument("--resume", action="store_true")
    #-------------------------------------------------------------------------------------
    
    args = parser.parse_args()
    return args


def generate(rank, args):
    visible_devices = [
        str(rank * args.gpus_per_model + i)
        for i in range(args.gpus_per_model)
    ]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("pad_token_id is None, set it to eos_token_id")
    elif tokenizer.eos_token_id is None and tokenizer.pad_token_id is not None:
        tokenizer.eos_token_id = tokenizer.pad_token_id
        print("eos_token_id is None, set it to pad_token_id")
    elif tokenizer.eos_token_id is None and tokenizer.pad_token_id is None:
        print("both eos_token_id and pad_token_id are None")

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if "Qwen3_30B_A3B" in args.model_name_or_path or "Mixtral" in args.model_name_or_path:
        config.num_experts_per_tok = args.num_experts_per_tok
        if args.routed_tok == 0:
            config.routed_tok = [_ for _ in range(args.num_experts_per_tok)]
        else:
            config.routed_tok = generate_combinations(8 if "Mixtral" in args.model_name_or_path else 128, args.num_experts_per_tok)[args.routed_tok]
    if args.decoding_method == "dynamic":
        config.dynamic_expert_routing_threshold = args.dynamic_expert_routing_threshold
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        config=config,
    ).eval()

    if args.decoding_method == "cd":
        config = AutoConfig.from_pretrained(
            args.student_model_name_or_path, trust_remote_code=True
        )
        student_model = AutoModelForCausalLM.from_pretrained(
            args.student_model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            config=config,
            trust_remote_code=True,
        ).eval()

    prompt_lst = []

    with open(args.infile) as f:
        idx = 0
        for line in f.readlines():
            d = json.loads(line.strip())
            d["idx"] = idx
            prompt_lst.append(d)
            idx += 1

    print(f"the total number of prompts: {len(prompt_lst)}")
    prompt_lst = prompt_lst[rank :: args.num_processes]
    print(f"the total number of prompts for rank {rank}: {len(prompt_lst)}")

    # --- Resume: filter out prompts already done (from shard files and/or a prior final file)
    if getattr(args, "resume", False):
        done = set()
    
        def _read_done(path: str):
            try:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    df = pd.read_json(path, lines=True)
                    if "idx" in df.columns:
                        return set(df["idx"].tolist())
            except Exception:
                pass
            return set()
    
        # If a prior final outfile exists, honor it
        done |= _read_done(args.outfile)
    
        # Collect finished idxs from any existing per-rank shard files (…jsonl0, …jsonl1, …)
        r = 0
        while True:
            shard = args.outfile + f"{r}"
            if not os.path.exists(shard):
                break
            done |= _read_done(shard)
            r += 1
    
        if done:
            before = len(prompt_lst)
            prompt_lst = [d for d in prompt_lst if d["idx"] not in done]
            print(
                f"resuming: will skip {len(done)} finished idxs; "
                f"remaining {len(prompt_lst)} of {before} for rank {rank}"
            )
    # ---------------------------------------------------------------------------------------------------
    
    
    s = time.time()
    
    for start in tqdm(range(0, len(prompt_lst), args.batch_size), disable=rank != 0):
        stopping_criteria = StoppingCriteriaList()
            
        if "deepseek-moe" in args.model_name_or_path:
            if "mbpp" in args.infile:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(token_id_list=[[58, 95742, 60]])
                )
            if "human_eval" in args.infile:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(
                        
                        token_id_list=[[185, 1558], [185, 351, 5589, 1531, 1442, 2318]],
                    )
                )
            else:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(token_id_list=[[185, 185]])
                )
        elif "Mixtral" in args.model_name_or_path:
            if "mbpp" in args.infile:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(
                        token_id_list=[[28792, 28757, 6349, 28793]]
                    )
                )
            if "human_eval" in args.infile:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(
                        token_id_list=[[13, 1270], [13, 335, 1848, 861, 860, 859]]
                    )
                )
            else:
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(token_id_list=[[13, 13]])
                )
        elif "Qwen3" in args.model_name_or_path or "qwen3" in args.model_name_or_path:
            if "mbpp" in args.infile:
                #   '[DONE]' -> [64920, 5225, 60]
                #   ' [DONE]' -> [58, 95742, 60]
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(
                        token_id_list=[[64920, 5225, 60], [58, 95742, 60]]
                    )
                )
            elif "human_eval" in args.infile:
                #   '\ndef' -> [198, 750]
                #   '\nif __name__ ==' -> [198, 333, 1304, 606, 563, 621]
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(
                        token_id_list=[[198, 750], [198, 333, 1304, 606, 563, 621]]
                    )
                )
            else:
                #  "\nQuestion:" -> [198, 14582, 25]
                #  "\n\nQuestion:" -> [271, 14582, 25]
                stopping_criteria.append(
                    StopAtSpecificTokenCriteria(token_id_list=[[198, 14582, 25], [271, 14582, 25]])
                )
        
        if start % 20 == 0 and rank == 0:
            print(f"rank {rank} has generated {start} prompts")
        cur_prompt_lst = prompt_lst[start : start + args.batch_size]
        prompt_text = [f"{x['instructions']}" for x in cur_prompt_lst]
        model_inputs = tokenizer(
            prompt_text, padding=True, add_special_tokens=True, return_tensors="pt"
        )
        input_ids = model_inputs["input_ids"].to(model.device)
        attention_mask = model_inputs["attention_mask"].to(model.device)
        prompt_len = input_ids.size(1)
        
        # Build substring stopper bound to this batch's prompt start
        raw_list = [s.strip() for s in (args.stop_text.split(",") if args.stop_text else [])]
        stop_strings = [s.replace("\\n", "\n") for s in raw_list if s]
        if stop_strings:
   
            # NEW (strict answer-window, no fallback):
            stopping_criteria.append(
                StopAfterAnswerWindowStrict(
                    tokenizer,
                    stop_strings=stop_strings,         # e.g., ["\n\nQuestion:"]
                    start_pos=prompt_len,
                    answer_phrase_regex=r"(?i)\b(the\s+answer\s+is|answer\s*[:=])\b",
                    warmup_tokens=8,
                    window_words=20,
                    check_every=2,
                )
            )
        #---------------------------------------------------------------------------
        
        if args.decoding_method == "greedy" or args.decoding_method == "dynamic":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "cs":
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                penalty_alpha=args.cs_alpha,
                top_k=args.cs_k,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "dola":
            early_exit_layers = [int(x) for x in args.dola_early_exit_layers.split(",")]
            outputs = dola(
                model,
                tokenizer,
                input_ids,
                attention_mask,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=1.2,
                mature_layer=args.dola_mature_layer,
                base_layer=None,
                candidate_premature_layers=early_exit_layers,
                relative_top=0.1,
                eos_token_id=None,
                stopping_criteria=stopping_criteria,
                early_stop=args.early_stop,
            )
        if args.decoding_method == "cd":
            outputs = contrastive_decoding(
                model,
                student_model,
                teacher_t=args.cd_tt,
                student_t=args.cd_st,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stop=True,
                alpha=args.cd_alpha,
                beta=args.cd_beta,
                stopping_criteria=stopping_criteria,
            )
        if args.decoding_method == "scmoe":
            if args.routed_tok == 0:
                MoE_mapping_teacher = [[_ for _ in range(args.num_experts_per_tok)]]
            else:
                MoE_mapping_teacher = generate_combinations(
                    8 if "Mixtral" in args.model_name_or_path else 128,
                    args.num_experts_per_tok,
                )
            if (args.student_routed_tok == 8 or args.student_routed_tok == 128) and args.student_num_experts_per_tok == 1:
                MoE_mapping_student = [[_] for _ in range(args.student_routed_tok + 1)]
            elif args.student_routed_tok == 0:
                MoE_mapping_student = [[_ for _ in range(args.student_num_experts_per_tok)]]
            else:
                MoE_mapping_student = generate_combinations(
                    8 if "Mixtral" in args.student_model_name_or_path else 128,
                    args.student_num_experts_per_tok,
                )
            outputs = scmoe(
                model,
                teacher_t=args.cd_tt,
                student_t=args.cd_st,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stop=args.early_stop,
                alpha=args.cd_alpha,
                beta=args.cd_beta,
                dynamic_beta=args.dynamic_beta,
                stopping_criteria=stopping_criteria,
                teacher_routed_tok=MoE_mapping_teacher[args.routed_tok],
                teacher_num_experts_per_tok=args.num_experts_per_tok,
                student_routed_tok=MoE_mapping_student[args.student_routed_tok],
                student_num_experts_per_tok=args.student_num_experts_per_tok,
            )
        if args.decoding_method == "scmoe_with_sampling":
            if args.routed_tok == 0:
                MoE_mapping_teacher = [[_ for _ in range(args.num_experts_per_tok)]]
            else:
                MoE_mapping_teacher = generate_combinations(
                    8 if "Mixtral" in args.model_name_or_path else 128,
                    args.num_experts_per_tok,
                )
            if args.student_routed_tok == 0:
                MoE_mapping_student = [[_ for _ in range(args.student_num_experts_per_tok)]]
            else:
                MoE_mapping_student = generate_combinations(
                    8 if "Mixtral" in args.student_model_name_or_path else 128,
                    args.student_num_experts_per_tok,
                )
            outputs = scmoe_with_sampling(
                model,
                teacher_t=args.cd_tt,
                student_t=args.cd_st,
                tokenizer=tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                early_stop=args.early_stop,
                alpha=args.cd_alpha,
                beta=args.cd_beta,
                stopping_criteria=stopping_criteria,
                teacher_routed_tok=MoE_mapping_teacher[args.routed_tok],
                teacher_num_experts_per_tok=args.num_experts_per_tok,
                student_routed_tok=MoE_mapping_student[args.student_routed_tok],
                student_num_experts_per_tok=args.student_num_experts_per_tok,
            )

        generation_text = tokenizer.batch_decode(
            outputs[:, prompt_len:],
            clean_up_tokenization_spaces=True,
            skip_special_tokens=True,
        )
        
        for prompt, generation in zip(cur_prompt_lst, generation_text):
            json_str = json.dumps(
                {
                    "idx": prompt["idx"],
                    "completion": generation.strip(),
                }
            )
            with open(args.outfile + f"{rank}", "a", encoding="utf-8") as f:
                f.write(json_str + "\n")

    t = time.time()
    print("time used: ", t - s)

if __name__ == "__main__":
    args = args_parse()
    print(args)
    assert args.world_size % args.gpus_per_model == 0
    args.num_processes = args.world_size // args.gpus_per_model
    process_list = []
    for i in range(args.num_processes):
        p = Process(target=generate, args=(i, args))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
    all_ret = pd.DataFrame()
    for rank in range(args.num_processes):
        with open(args.outfile + f"{rank}", "r", encoding="utf-8") as f:
            all_ret = pd.concat(
                [all_ret, pd.read_json(f, lines=True)], ignore_index=True
            )
    all_ret.sort_values(by="idx", inplace=True)
    all_ret.to_json(args.outfile, orient="records", lines=True, force_ascii=False)
    for rank in range(args.num_processes):
        os.remove(args.outfile + f"{rank}")