#Simple Llama.cpp inference

1. Using Github codespaces with min specs and CPU
2. Trying to run inference directly with Llama.cpp
3. Need to make it on Linux, however for some reason the following command is not working, although it was last time:

./main --hf-repo "microsoft/Phi-3-mini-4k-instruct-gguf" -m Phi-3-mini-4k-instruct-fp16.gguf -p "I believe the meaning of life is" -n 128

I keep getting the error: ./main file / directory not found, so using Google Colab to run with L4 GPU setting, able to use the terminal because of paid session

Just using ./llama-cli which seems to be working, so inferencing using the following as example:

./llama-cli --hf-repo "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF" -m Meta-Llama-3-8B-Instruct-IQ3_M.gguf -p "I believe the meaning of life is" -n 128

This works, don't know if anythig changed. Rationale for trying Llama before any high level library with Py, HF is to see what it does

Here's the export:

Log start
main: build = 3181 (37bef894)
main: built with cc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0 for x86_64-linux-gnu
main: seed  = 1718737909
llama_download_file: no previous model file found Meta-Llama-3-8B-Instruct-IQ3_M.gguf
llama_download_file: downloading from https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-IQ3_M.gguf to Meta-Llama-3-8B-Instruct-IQ3_M.gguf (server_etag:"2845dc88b7bac19f58efe0a7fe046cea-237", server_last_modified:Thu, 18 Apr 2024 22:40:02 GMT)...
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  1165  100  1165    0     0  15379      0 --:--:-- --:--:-- --:--:-- 15379
100 3609M  100 3609M    0     0  70.9M      0  0:00:50  0:00:50 --:--:-- 57.6M
llama_download_file: file metadata saved: Meta-Llama-3-8B-Instruct-IQ3_M.gguf.json
llama_model_loader: loaded meta data with 21 key-value pairs and 291 tensors from Meta-Llama-3-8B-Instruct-IQ3_M.gguf (version GGUF V3 (latest))
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = Meta-Llama-3-8B-Instruct-imatrix
llama_model_loader: - kv   2:                          llama.block_count u32              = 32
llama_model_loader: - kv   3:                       llama.context_length u32              = 8192
llama_model_loader: - kv   4:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 14336
llama_model_loader: - kv   6:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   7:              llama.attention.head_count_kv u32              = 8
llama_model_loader: - kv   8:                       llama.rope.freq_base f32              = 500000.000000
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000010
llama_model_loader: - kv  10:                          general.file_type u32              = 27
llama_model_loader: - kv  11:                           llama.vocab_size u32              = 128256
llama_model_loader: - kv  12:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv  13:                       tokenizer.ggml.model str              = gpt2
llama_model_loader: - kv  14:                      tokenizer.ggml.tokens arr[str,128256]  = ["!", "\"", "#", "$", "%", "&", "'", ...
llama_model_loader: - kv  15:                  tokenizer.ggml.token_type arr[i32,128256]  = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...
llama_model_loader: - kv  16:                      tokenizer.ggml.merges arr[str,280147]  = ["Ġ Ġ", "Ġ ĠĠĠ", "ĠĠ ĠĠ", "...
llama_model_loader: - kv  17:                tokenizer.ggml.bos_token_id u32              = 128000
llama_model_loader: - kv  18:                tokenizer.ggml.eos_token_id u32              = 128001
llama_model_loader: - kv  19:                    tokenizer.chat_template str              = {% set loop_messages = messages %}{% ...
llama_model_loader: - kv  20:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:   68 tensors
llama_model_loader: - type q6_K:    1 tensors
llama_model_loader: - type iq3_s:  157 tensors
llm_load_vocab: missing pre-tokenizer type, using: 'default'
llm_load_vocab:                                             
llm_load_vocab: ************************************        
llm_load_vocab: GENERATION QUALITY WILL BE DEGRADED!        
llm_load_vocab: CONSIDER REGENERATING THE MODEL             
llm_load_vocab: ************************************        
llm_load_vocab:                                             
llm_load_vocab: special tokens cache size = 256
llm_load_vocab: token to piece cache size = 0.8000 MB
llm_load_print_meta: format           = GGUF V3 (latest)
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = BPE
llm_load_print_meta: n_vocab          = 128256
llm_load_print_meta: n_merges         = 280147
llm_load_print_meta: n_ctx_train      = 8192
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 8
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 4
llm_load_print_meta: n_embd_k_gqa     = 1024
llm_load_print_meta: n_embd_v_gqa     = 1024
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-05
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: f_logit_scale    = 0.0e+00
llm_load_print_meta: n_ff             = 14336
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: causal attn      = 1
llm_load_print_meta: pooling type     = 0
llm_load_print_meta: rope type        = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 500000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_ctx_orig_yarn  = 8192
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: ssm_d_conv       = 0
llm_load_print_meta: ssm_d_inner      = 0
llm_load_print_meta: ssm_d_state      = 0
llm_load_print_meta: ssm_dt_rank      = 0
llm_load_print_meta: model type       = 8B
llm_load_print_meta: model ftype      = IQ3_S mix - 3.66 bpw
llm_load_print_meta: model params     = 8.03 B
llm_load_print_meta: model size       = 3.52 GiB (3.76 BPW) 
llm_load_print_meta: general.name     = Meta-Llama-3-8B-Instruct-imatrix
llm_load_print_meta: BOS token        = 128000 '<|begin_of_text|>'
llm_load_print_meta: EOS token        = 128001 '<|end_of_text|>'
llm_load_print_meta: LF token         = 128 'Ä'
llm_load_print_meta: EOT token        = 128009 '<|eot_id|>'
llm_load_tensors: ggml ctx size =    0.15 MiB
llm_load_tensors:        CPU buffer size =  3602.02 MiB
.....................................................................................
llama_new_context_with_model: n_ctx      = 8192
llama_new_context_with_model: n_batch    = 2048
llama_new_context_with_model: n_ubatch   = 512
llama_new_context_with_model: flash_attn = 0
llama_new_context_with_model: freq_base  = 500000.0
llama_new_context_with_model: freq_scale = 1
llama_kv_cache_init:        CPU KV buffer size =  1024.00 MiB
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_new_context_with_model:        CPU  output buffer size =     0.49 MiB
llama_new_context_with_model:        CPU compute buffer size =   560.01 MiB
llama_new_context_with_model: graph nodes  = 1030
llama_new_context_with_model: graph splits = 1

system_info: n_threads = 6 / 12 | AVX = 1 | AVX_VNNI = 0 | AVX2 = 1 | AVX512 = 1 | AVX512_VBMI = 0 | AVX512_VNNI = 1 | AVX512_BF16 = 0 | FMA = 1 | NEON = 0 | SVE = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | MATMUL_INT8 = 0 | LLAMAFILE = 1 | 
sampling: 
        repeat_last_n = 64, repeat_penalty = 1.000, frequency_penalty = 0.000, presence_penalty = 0.000
        top_k = 40, tfs_z = 1.000, top_p = 0.950, min_p = 0.050, typical_p = 1.000, temp = 0.800
        mirostat = 0, mirostat_lr = 0.100, mirostat_ent = 5.000
sampling order: 
CFG -> Penalties -> top_k -> tfs_z -> typical_p -> top_p -> min_p -> temperature 
generate: n_ctx = 8192, n_batch = 2048, n_predict = 128, n_keep = 0


I believe the meaning of life is to find happiness, but I also believe that happiness is
 subjective and can be found in many different ways. For some people, happiness may be found in their work, their family, or their friends. For others, happiness may be found in their passions, their hobbies, or their personal goals. Ultimately, I believe that the
 meaning of life is a personal and individual question that can only be answered by each
 person for themselves.} & 0.8 & 0.2 & 0.5 \\ \hline
\end{tabular}

In this table, we can see that the model is most confident when the user
llama_print_timings:        load time =    1445.76 ms
llama_print_timings:      sample time =      19.03 ms /   128 runs   (    0.15 ms per token,  6727.64 tokens per second)
llama_print_timings: prompt eval time =    1699.80 ms /     7 tokens (  242.83 ms per token,     4.12 tokens per second)
llama_print_timings:        eval time =   36553.96 ms /   127 runs   (  287.83 ms per token,     3.47 tokens per second)
llama_print_timings:       total time =   38416.27 ms /   134 tokens
Log end

Important points to remember:
1. You can use non GGUF models (best for use with Llama.cpp) but you need to convert them to GGUF - Follow ./llama-quantize ./Meta-Llama-3-8B-Instruct-IQ3_M.gguf ./models/Meta-Llama-3-8B-Instruct-IQ3_Q4_K_M.gguf Q4_K_M
2. Generally, as HF suggests with it's models, Llama.cpp can directly download GGUF models from HF
3. These models (GGUF) are aleady quantized (compressed) and so cannot be re-quantized, as I tried to do:

llama_model_loader: - type iq3_s:  157 tensors
[   1/ 291]                    token_embd.weight - [ 4096, 128256,     1,     1], type =  iq3_s, llama_model_quantize: failed to quantize: requantizing from type iq3_s is disabled
main: failed to quantize model from './Meta-Llama-3-8B-Instruct-IQ3_M.gguf'

Regular models can also be added:
https://huggingface.co/spaces/ggml-org/gguf-my-repo
and converted:
The space takes an HF repo as an input, quantizes it and creates a Public repo containing the selected quant under your HF user namespace.



