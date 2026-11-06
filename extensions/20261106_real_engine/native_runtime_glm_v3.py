"""Native vLLM observation and durable raw returns, never response memoization."""
from experiment_common import *
import os
import sys


def durable(path, obj):
    with Path(path).open("a") as f:
        f.write(compact(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def worker_identity(worker):
    model = worker.model_runner.model
    parameters = list(model.parameters())
    return {"pid": os.getpid(), "model_class": type(model).__name__,
            "module": type(model).__module__,
            "dtypes": sorted({str(p.dtype) for p in parameters}),
            "parameters": sum(p.numel() for p in parameters),
            "remote_model_modules": [n for n in sys.modules if n.startswith("transformers_modules.")],
            "dynamic_module_source_sha256": {n: sha(sys.modules[n].__file__) for n in sys.modules if n.startswith("transformers_modules.") and getattr(sys.modules[n], "__file__", None)}}


class NativeBackend:
    def __init__(self, llm, out, model, cache, gpu, owned):
        self.llm, self.out, self.model = llm, Path(out), model
        self.cache, self.gpu, self.owned = cache, gpu, owned
        self.stage, self.ordinal, self.active, self.observed = "unassigned", 0, None, []
        self.tokenizer = llm.get_tokenizer()
        self.responses = []
        engine = llm.llm_engine
        cfg = engine.vllm_config
        native = llm.collective_rpc(worker_identity)
        expect = {
            "qwen": ("Qwen3ForCausalLM", "vllm.model_executor.models.qwen3", 4022468096),
            "glm": ("ChatGLMForCausalLM", "vllm.model_executor.models.chatglm", 9399951360),
        }[model]
        receipt = {
            "native_workers": native, "model": model,
            "model_path": MODELS[model], "dtype": str(cfg.model_config.dtype),
            "max_model_len": cfg.model_config.max_model_len,
            "max_num_seqs": cfg.scheduler_config.max_num_seqs,
            "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
            "enforce_eager": cfg.model_config.enforce_eager,
            "prefix_cache": cfg.cache_config.enable_prefix_caching,
            "primary_eos": self.tokenizer.eos_token_id,
            "generation_config_fields": engine.processor.generation_config_fields,
            "trusted_local_rpc": "Generic serialization enabled only in this new local experiment process; reviewed worker_identity callable, not an RPC allowlist.",
        }
        write_json(self.out / "EFFECTIVE_NATIVE_RUNTIME.json", receipt)
        assert len(native) == 1
        assert (native[0]["model_class"], native[0]["module"], native[0]["parameters"]) == expect
        assert native[0]["dtypes"] == ["torch.bfloat16"]
        allowed_tokenizer = "transformers_modules.glm_hyphen_4_hyphen_9b_hyphen_chat.tokenization_chatglm"
        provenance = json.loads((ROOT / "prepared/MODEL_PROVENANCE.json").read_text())
        assert model == "glm"
        assert native[0]["remote_model_modules"] == [allowed_tokenizer]
        assert native[0]["dynamic_module_source_sha256"] == {
            allowed_tokenizer: provenance["glm"]["files"]["tokenization_chatglm.py"]
        }, "Only the exact already-verified local tokenizer module is allowed"
        self.native_reset_pid = native[0]["pid"]
        assert receipt["max_model_len"] == CONFIG["max_model_len"]
        assert receipt["max_num_seqs"] == CONFIG["max_num_seqs"]
        assert receipt["prefix_cache"] == cache and receipt["enforce_eager"] is True
        assert receipt["tensor_parallel_size"] == 1
        original = engine.processor.process_inputs

        def observed(*args, **kwargs):
            result = original(*args, **kwargs)
            request = result[1]
            p = request.sampling_params
            position = len(self.observed)
            expected = self.active[position] if self.active is not None and position < len(self.active) else None
            controls = {k: getattr(p, k) for k in
                        ("n", "temperature", "top_k", "top_p", "min_p", "max_tokens",
                         "ignore_eos", "repetition_penalty", "presence_penalty", "frequency_penalty")}
            controls.update({"guided": p.guided_decoding is not None,
                             "guided_backend": p.guided_decoding.backend if p.guided_decoding else None,
                             "guided_disable_fallback": p.guided_decoding.disable_fallback if p.guided_decoding else None,
                             "grammar_sha256": digest(p.guided_decoding.regex) if p.guided_decoding else None,
                             "all_stop_token_ids": sorted(p.all_stop_token_ids),
                             "sampling_type": p.sampling_type.name})
            row = {"logical_id": expected["logical_id"] if expected else None,
                   "engine_request_id": request.request_id,
                   "prompt_token_ids_sha256": digest(compact(list(request.prompt_token_ids))),
                   "stage": self.stage, "controls": controls, "observed_unix": time.time()}
            durable(self.out / "effective_requests.jsonl", row)
            self.observed.append(row)
            assert expected is not None
            assert row["prompt_token_ids_sha256"] == expected["prompt_token_ids_sha256"]
            assert (p.n, p.temperature, p.top_k, p.top_p, p.min_p) == (1, 0, 0, 1, 0)
            assert (p.repetition_penalty, p.presence_penalty, p.frequency_penalty) == (1, 0, 0)
            assert not p.ignore_eos and p.sampling_type.name == "GREEDY"
            assert p.max_tokens == expected["max_output"]
            assert controls["guided"] == (expected["grammar"] is not None)
            if controls["guided"]:
                assert controls["guided_backend"] == "xgrammar" and controls["guided_disable_fallback"] is True
                assert controls["grammar_sha256"] == digest(expected["grammar"])
            return result

        engine.processor.process_inputs = observed

    def reset_policy(self, case_id, policy):
        self.owned(self.gpu)
        observer_path = Path(os.environ["INTENTSAT_CACHE_RESET_LOG"])
        before = read_lines(observer_path) if observer_path.exists() else []
        reset = self.llm.reset_prefix_cache() if self.cache else None
        after = read_lines(observer_path) if observer_path.exists() else []
        observations = after[len(before):]
        durable(self.out / "cache_resets.jsonl", {"case_id": case_id, "policy": policy,
                "prefix_cache": self.cache, "outer_reset_return": reset,
                "native_reset_observations": observations, "time_unix": time.time()})
        if self.cache and (len(observations) != 1 or any(
            r["native_result"] is not True or r["cached_hash_count_after"] != 0
            or r["num_gpu_blocks"] - r["num_free_blocks_after"] != 1
            or r["pid"] != self.native_reset_pid
            or r["native_class"] != "vllm.v1.core.block_pool.BlockPool"
            for r in observations
        )):
            raise RuntimeError("Actual native prefix cache reset was not verified at policy boundary")

    def generate(self, messages_list, *, grammar=None, max_output=32):
        from vllm import SamplingParams
        from vllm.sampling_params import GuidedDecodingParams
        all_rows = []
        for offset in range(0, len(messages_list), CONFIG["science_block"]):
            self.owned(self.gpu)
            messages = messages_list[offset:offset+CONFIG["science_block"]]
            jobs = []
            for ms in messages:
                ids = self.tokenizer.apply_chat_template(ms, tokenize=True, add_generation_prompt=True)
                if len(ids) + max_output > CONFIG["max_model_len"]:
                    raise RuntimeError("Complete prompt exceeds frozen context; do not truncate or substitute")
                job = {"logical_id": f"{self.model}:{int(self.cache)}:{self.ordinal}",
                       "messages_sha256": digest(compact(ms)),
                       "prompt_token_ids_sha256": digest(compact(ids)),
                       "prompt_token_ids": ids, "grammar": grammar, "max_output": max_output}
                self.ordinal += 1
                jobs.append(job)
            params = SamplingParams(temperature=0, max_tokens=max_output,
                        guided_decoding=GuidedDecodingParams(regex=grammar, backend="xgrammar", disable_fallback=True)
                        if grammar else None)
            durable(self.out / "submissions.jsonl", {"stage": self.stage, "jobs": jobs, "time_unix": time.time()})
            self.active, self.observed = jobs, []
            started = time.perf_counter()
            values = self.llm.generate([{"prompt_token_ids": j["prompt_token_ids"]} for j in jobs],
                                       params, use_tqdm=False)
            seconds = time.perf_counter() - started
            # Persist EVERY returned item, before checking counts or parsing.
            returned = []
            for position, v in enumerate(values):
                pids = getattr(v, "prompt_token_ids", None)
                answers = []
                for a in v.outputs:
                    answers.append({"raw": a.text, "output_token_ids": list(a.token_ids),
                                    "finish_reason": a.finish_reason, "stop_reason": a.stop_reason})
                returned.append({"position": position, "engine_request_id": v.request_id,
                    "prompt_token_ids": list(pids) if pids is not None else None,
                    "num_cached_tokens": getattr(v, "num_cached_tokens", None), "outputs": answers})
            durable(self.out / "raw_returns.jsonl", {"stage": self.stage, "expected_ids": [j["logical_id"] for j in jobs],
                    "returns": returned, "seconds": seconds, "time_unix": time.time(),
                    "raw_saved_before_validation": True})
            assert len(values) == len(jobs) == len(self.observed)
            for job, value, actual in zip(jobs, values, self.observed):
                assert value.request_id == actual["engine_request_id"]
                assert digest(compact(list(value.prompt_token_ids))) == job["prompt_token_ids_sha256"]
                assert len(value.outputs) == 1
                answer = value.outputs[0]
                cached = getattr(value, "num_cached_tokens", None)
                assert cached is None or 0 <= cached <= len(value.prompt_token_ids)
                row = {"logical_id": job["logical_id"], "stage": self.stage,
                    "engine_request_id": value.request_id, "raw": answer.text,
                    "messages_sha256": job["messages_sha256"],
                    "prompt_token_ids_sha256": job["prompt_token_ids_sha256"],
                    "actual_input_tokens": len(value.prompt_token_ids),
                    "actual_output_tokens": len(answer.token_ids),
                    "cached_input_tokens": cached,
                    "observed_uncached_input_tokens": len(value.prompt_token_ids)-cached if cached is not None else None,
                    "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason}
                self.responses.append(row)
                all_rows.append(row)
                durable(self.out / "responses.jsonl", row)
            self.active = None
        return all_rows

    def usage(self, start):
        rows = self.responses[start:]
        missing = sum(r["cached_input_tokens"] is None for r in rows)
        return {"requests": len(rows), "input_tokens": sum(r["actual_input_tokens"] for r in rows),
                "output_tokens": sum(r["actual_output_tokens"] for r in rows),
                "cached_tokens_unknown_requests": missing,
                "cached_tokens_observed_lower_bound": sum(r["cached_input_tokens"] or 0 for r in rows),
                "cached_tokens": None if missing else sum(r["cached_input_tokens"] for r in rows),
                "uncached_tokens": None if missing else sum(r["observed_uncached_input_tokens"] for r in rows)}
