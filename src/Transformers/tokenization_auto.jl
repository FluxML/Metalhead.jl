# Ported from transformers.models.auto.tokenization_auto (transformers 5.20, Apache-2.0, The
# Hugging Face Team): the tokenizer class that `AutoTokenizer.from_pretrained` builds for a
# repository, whose `__init__` sets the defaults of its settings (`TOKENIZER_CLASSES`).

# `TOKENIZER_MAPPING_NAMES`: the tokenizer class of each model type of config.json.
const MODEL_TYPE_TOKENIZERS = Dict(String(m) => class for (class, models) in [
    "AlbertTokenizer" => "albert yoso",
    "BartTokenizer" => "florence2",
    "BarthezTokenizer" => "barthez",
    "BartphoTokenizer" => "bartpho",
    "BertGenerationTokenizer" => "bert-generation",
    "BertJapaneseTokenizer" => "bert-japanese",
    "BertTokenizer" => "align bark bert blip bros chinese_clip convbert distilbert " *
        "electra ernie flava git grounding-dino layoutlm megatron-bert mm-grounding-dino " *
        "nezha nomic_bert qdqbert realm retribert squeezebert tvp vilt visual_bert",
    "BertweetTokenizer" => "bertweet",
    "BigBirdTokenizer" => "big_bird",
    "BioGptTokenizer" => "biogpt",
    "BlenderbotSmallTokenizer" => "blenderbot-small",
    "BlenderbotTokenizer" => "blenderbot",
    "ByT5Tokenizer" => "byt5",
    "CLIPTokenizer" => "aimv2 clip clipseg groupvit omdet-turbo oneformer owlv2 owlvit " *
        "sam3 sam3_video xclip",
    "CTRLTokenizer" => "ctrl",
    "CamembertTokenizer" => "camembert",
    "CanineTokenizer" => "canine",
    "ClvpTokenizer" => "clvp",
    "CodeLlamaTokenizer" => "code_llama",
    "CohereTokenizer" => "aya_vision cohere cohere2",
    "CpmAntTokenizer" => "cpmant",
    "CpmTokenizer" => "cpm",
    "DPRQuestionEncoderTokenizer" => "dpr",
    "DebertaTokenizer" => "deberta",
    "DebertaV2Tokenizer" => "deberta-v2",
    "DiaTokenizer" => "dia",
    "EsmTokenizer" => "esm",
    "EsmcTokenizer" => "esmc esmfold2",
    "FNetTokenizer" => "fnet",
    "FSMTTokenizer" => "fsmt",
    "FastSpeech2ConformerTokenizer" => "fastspeech2_conformer",
    "FlaubertTokenizer" => "flaubert",
    "FunnelTokenizer" => "funnel",
    "GPT2Tokenizer" => "blip-2 codegen dbrx emu3 gpt2 gpt_neo gptj hyperclovax_vision_v2 " *
        "instructblip instructblipvideo jais2 opt phi starcoder2",
    "GPTNeoXJapaneseTokenizer" => "gpt_neox_japanese",
    "GPTNeoXTokenizer" => "falcon_mamba gpt_neox mamba mamba2 mpt olmo olmoe rwkv xlstm",
    "GPTSw3Tokenizer" => "gpt-sw3",
    "GemmaTokenizer" => "diffusion_gemma gemma gemma2 gemma3 gemma3_text gemma3n " *
        "gemma3n_text recurrent_gemma shieldgemma2 t5gemma zaya",
    "HerbertTokenizer" => "herbert",
    "LEDTokenizer" => "led",
    "LasrTokenizer" => "lasr_ctc lasr_encoder",
    "LayoutLMv2Tokenizer" => "layoutlmv2",
    "LayoutLMv3Tokenizer" => "layoutlmv3",
    "LayoutXLMTokenizer" => "layoutxlm",
    "LlamaTokenizer" => "idefics idefics2",
    "LukeTokenizer" => "luke",
    "LxmertTokenizer" => "lxmert",
    "M2M100Tokenizer" => "m2m_100",
    "MBart50Tokenizer" => "mbart50",
    "MBartTokenizer" => "mbart",
    "MLukeTokenizer" => "mluke",
    "MPNetTokenizer" => "mpnet",
    "MarianTokenizer" => "marian",
    "MarkupLMTokenizer" => "markuplm",
    "MgpstrTokenizer" => "mgp-str",
    "MobileBertTokenizer" => "mobilebert",
    "MvpTokenizer" => "mvp",
    "MyT5Tokenizer" => "myt5",
    "NllbTokenizer" => "nllb nllb-moe",
    "NougatTokenizer" => "nougat pp_formulanet",
    "OpenAIGPTTokenizer" => "openai-gpt",
    "PLBartTokenizer" => "plbart",
    "ParakeetTokenizer" => "granite_speech5_ctc nemotron3_5_asr nemotron_asr_streaming " *
        "parakeet_ctc parakeet_rnnt parakeet_tdt",
    "PegasusTokenizer" => "pegasus",
    "PerceiverTokenizer" => "perceiver",
    "PhobertTokenizer" => "phobert",
    "ProphetNetTokenizer" => "prophetnet",
    "Qwen2Tokenizer" => "audioflamingo3 colqwen2 cosmos3_omni fun_asr_nano hunyuan_vl " *
        "internvl moss_transcribe_diarize ovis2 qianfan_ocr qwen2 qwen2_5_omni " *
        "qwen2_5_vl qwen2_audio qwen2_moe qwen2_vl qwen3 qwen3_asr qwen3_moe qwen3_next " *
        "qwen3_omni_moe qwen3_vl qwen3_vl_moe",
    "Qwen2TokenizerFast" => "lighton_ocr vibevoice",
    "Qwen3_5Tokenizer" => "qwen3_5 qwen3_5_moe qwen3_5_moe_text qwen3_5_text qwen4_exp",
    "RagTokenizer" => "rag",
    "ReformerTokenizer" => "reformer",
    "RemBertTokenizer" => "rembert",
    "RoCBertTokenizer" => "roc_bert",
    "RoFormerTokenizer" => "roformer",
    "RobertaTokenizer" => "bart bridgetower clap data2vec-text ibert lilt longformer " *
        "mega mra roberta roberta-prelayernorm",
    "SeamlessM4TTokenizer" => "seamless_m4t seamless_m4t_v2",
    "Siglip2Tokenizer" => "siglip2",
    "SiglipTokenizer" => "siglip",
    "Speech2TextTokenizer" => "speech_to_text",
    "SpeechT5Tokenizer" => "speecht5",
    "SplinterTokenizer" => "splinter",
    "T5Tokenizer" => "mt5 musicgen musicgen_melody switch_transformers t5",
    "TapasTokenizer" => "tapas",
    "Tipsv2Tokenizer" => "tipsv2",
    "TokenizersBackend" => "EvollaModel arctic aria beam bigbird_pegasus " *
        "camembertv2-base canary chameleon chatlm cohere_asr deepseek_ocr deepseek_ocr2 " *
        "deepseek_v2 deepseek_v3 deepseek_v32 deepseek_v4 deepseek_vl deepseek_vl_hybrid " *
        "deepseek_vl_v2 ernie4_5 ernie4_5_moe flex_olmo fuyu glm glm4 glm4_moe " *
        "glm4_moe_lite glm4v glm4v_moe glm_image glmasr got_ocr2 gpt_bigcode granite " *
        "granitemoe granitemoehybrid granitemoeshared h2ovl_chat hyperclovax " *
        "hyperclovax_vlm internlm2 jamba janus kimi_k25 kimi_linear kosmos-2 llava " *
        "llava_next mimo_v2_flash minicpm3 minicpmv minicpmv4_6 minicpmv4_7 minimax_m2 " *
        "ministral ministral3 mistral mistral3 mixtral modernbert modernbert-decoder " *
        "molmo molmo2 nemotron nvfp4 nystromformer olmo2 olmo3 olmo_hybrid opencua " *
        "openvla paddleocr_vl pegasus_x persimmon phi3 phi3_v phimoe pix2struct pixtral " *
        "smolvlm stablelm step3_vl step3p5 umt5 unlimited_ocr vipllava voxtral " *
        "voxtral_realtime xlm-roberta-xl",
    "UdopTokenizer" => "udop",
    "VideoPrismTokenizer" => "videoprism",
    "VitsTokenizer" => "vits",
    "Wav2Vec2CTCTokenizer" => "data2vec-audio hubert unispeech unispeech-sat wav2vec2 " *
        "wav2vec2-bert wav2vec2-conformer",
    "Wav2Vec2PhonemeCTCTokenizer" => "wav2vec2_phoneme",
    "WhisperTokenizer" => "whisper",
    "XGLMTokenizer" => "xglm",
    "XLMRobertaTokenizer" => "gte jina_embeddings_v3 metaclip_2 trocr xlm-roberta xmod",
    "XLMTokenizer" => "xlm",
    "XLNetTokenizer" => "xlnet"
] for m in split(models))

# The model types of `MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS` of a class that isn't the
# generic one: it replaces the class that their files name.
const INCORRECT_HUB_TOKENIZER_CLASS = Set(["hyperclovax_vision_v2", "qwen2", "qwen3_5",
    "qwen3_5_moe", "qwen3_5_moe_text", "qwen4_exp", "siglip2"])

strip_fast(name) = replace(name, r"Fast$" => "")

# The names of the generic class, which AutoTokenizer also builds for a class it doesn't
# know.
const GENERIC_TOKENIZERS = ("TokenizersBackend", "PreTrainedTokenizer",
    "PreTrainedTokenizerFast", "PythonBackend", "MistralCommonBackend", "BloomTokenizer")
const KNOWN_TOKENIZERS = Set(strip_fast(c) for c in values(MODEL_TYPE_TOKENIZERS))

# `tokenizer_class_from_name`, as AutoTokenizer uses it.
function known_tokenizer(name)
    name in GENERIC_TOKENIZERS && return "TokenizersBackend"
    class = strip_fast(name)
    return class in KNOWN_TOKENIZERS ? class : "TokenizersBackend"
end

# The tokenizer class that `AutoTokenizer.from_pretrained` builds, and the file that decides
# it (`nothing` for none): `hub` is the `tokenizer_class` of tokenizer_config.json,
# `config_class` that of config.json and `model_type` its model type, each `nothing` if
# absent. Where the class that the files name differs from that of the model type, a
# generic class of the model type wins, as does the class of a model type whose files are
# known to name a wrong one.
function tokenizer_class(hub, config_class, model_type)
    named = something(hub, config_class, Some(nothing))
    registered = get(MODEL_TYPE_TOKENIZERS, model_type, nothing)
    by_model_type = "config.json (`model_type`)"
    if named !== nothing && registered !== nothing &&
       strip_fast(registered) != strip_fast(named)
        known_tokenizer(registered) == "TokenizersBackend" &&
            return "TokenizersBackend", by_model_type
        model_type in INCORRECT_HUB_TOKENIZER_CLASS &&
            return known_tokenizer(registered), by_model_type
    end
    hub === nothing || return known_tokenizer(hub), "tokenizer_config.json"
    config_class === nothing ||
        return known_tokenizer(config_class), "config.json (`tokenizer_class`)"
    registered === nothing && return "TokenizersBackend", nothing
    return known_tokenizer(registered), by_model_type
end

# The defaults that the `__init__` of transformers' tokenizer classes and their class
# attributes set, for the classes that Metalhead validates: those of the validated
# repositories, whose tokenizer.json tokenizes strings as the class, which rebuilds its
# pipeline, does. LlamaTokenizer isn't one: it pre-tokenizes Llama 2's tokenizer.json
# otherwise.
const TOKENIZER_CLASSES = Dict(
    "TokenizersBackend" => (;),
    "GPT2Tokenizer" => (; unk_token = "<|endoftext|>", bos_token = "<|endoftext|>",
        eos_token = "<|endoftext|>"),
    "Qwen2Tokenizer" => (; unk_token = "<|endoftext|>", eos_token = "<|endoftext|>",
        pad_token = "<|endoftext|>"),
    "Qwen3_5Tokenizer" => (; unk_token = "<|endoftext|>", eos_token = "<|endoftext|>",
        pad_token = "<|endoftext|>"),
    "GemmaTokenizer" => (; unk_token = "<unk>", bos_token = "<bos>", eos_token = "<eos>",
        pad_token = "<pad>", mask_token = "<mask>", padding_side = "left"),
    "CLIPTokenizer" => (; unk_token = "<|endoftext|>", bos_token = "<|startoftext|>",
        eos_token = "<|endoftext|>", pad_token = "<|endoftext|>"))

# Classes that rebuild the pre-tokenizer of tokenizer.json with their `add_prefix_space`.
const PREFIX_SPACE_CLASSES = ("GPT2Tokenizer", "Qwen2Tokenizer", "Qwen3_5Tokenizer")
