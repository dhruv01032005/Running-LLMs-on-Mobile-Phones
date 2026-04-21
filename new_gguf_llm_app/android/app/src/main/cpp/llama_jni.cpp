#include <jni.h>

#include <algorithm>
#include <cstdio>
#include <limits>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "llama.h"

namespace {

std::string g_model_path;
bool g_model_initialized = false;
std::mutex g_llama_mutex;
llama_model * g_model = nullptr;
llama_context * g_ctx = nullptr;
bool g_backend_initialized = false;

int g_context_size = 8192;
int g_max_gen_tokens = 2048;
int g_last_generated_tokens = 0;

void unload_model() {
    if (g_ctx != nullptr) {
        llama_free(g_ctx);
        g_ctx = nullptr;
    }
    if (g_model != nullptr) {
        llama_model_free(g_model);
        g_model = nullptr;
    }
    g_model_initialized = false;
}

std::string jstring_to_string(JNIEnv * env, jstring value) {
    if (value == nullptr) {
        return "";
    }

    const char * chars = env->GetStringUTFChars(value, 0);
    std::string out(chars == nullptr ? "" : chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

bool file_exists(const std::string & path) {
    FILE * fp = std::fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        return false;
    }
    std::fclose(fp);
    return true;
}

std::string load_model_internal(const std::string & path, int n_ctx, int n_predict) {
    if (path.empty()) {
        return "Model path is empty";
    }

    if (!file_exists(path)) {
        return "Model file not found: " + path;
    }

    unload_model();

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }

    g_context_size = std::max(1024, n_ctx);
    g_max_gen_tokens = std::clamp(n_predict, 16, 2048);

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(path.c_str(), model_params);
    if (g_model == nullptr) {
        return "Failed to load GGUF model: " + path;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = static_cast<uint32_t>(g_context_size);
    ctx_params.n_batch = 256;
    ctx_params.n_threads = std::max(1u, std::thread::hardware_concurrency());

    g_ctx = llama_init_from_model(g_model, ctx_params);
    if (g_ctx == nullptr) {
        unload_model();
        return "Failed to create llama context";
    }

    g_model_path = path;
    g_model_initialized = true;
    return "Model initialized: " + g_model_path;
}

std::string token_to_text_piece(llama_token token) {
    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (vocab == nullptr) {
        return "";
    }

    char piece[256];
    const int n = llama_token_to_piece(
        vocab,
        token,
        piece,
        static_cast<int>(sizeof(piece)),
        0,
        true);

    if (n <= 0) {
        return "";
    }

    return std::string(piece, piece + n);
}

llama_token sample_next_token(
    const float * logits,
    int n_vocab,
    const llama_vocab * vocab,
    const std::vector<llama_token> & generated_tokens) {
    if (logits == nullptr || n_vocab <= 0) {
        return LLAMA_TOKEN_NULL;
    }

    constexpr int kMinTokensBeforeEos = 128;

    llama_token best_token = LLAMA_TOKEN_NULL;
    float best_logit = -std::numeric_limits<float>::infinity();
    const llama_token eos = vocab == nullptr ? LLAMA_TOKEN_NULL : llama_vocab_eos(vocab);

    for (int tok = 0; tok < n_vocab; ++tok) {
        if (tok == eos && static_cast<int>(generated_tokens.size()) < kMinTokensBeforeEos) {
            continue;
        }

        const float logit = logits[tok];
        if (logit > best_logit) {
            best_logit = logit;
            best_token = static_cast<llama_token>(tok);
        }
    }

    return best_token;
}

}  // namespace

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_new_1gguf_1llm_1app_MainActivity_initializeBackend(
    JNIEnv * env,
    jobject /* this */,
    jstring /* nativeLibDir */) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);

    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
        return env->NewStringUTF("Backend initialized");
    }

    return env->NewStringUTF("Backend already initialized");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_new_1gguf_1llm_1app_MainActivity_initializeModelNative(
    JNIEnv * env,
    jobject /* this */,
    jstring modelPath,
    jint nCtx,
    jint nPredict) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);

    const std::string path = jstring_to_string(env, modelPath);
    const std::string response = load_model_internal(path, static_cast<int>(nCtx), static_cast<int>(nPredict));
    return env->NewStringUTF(response.c_str());
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_new_1gguf_1llm_1app_MainActivity_disposeModelNative(
    JNIEnv * env,
    jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);

    const bool had_model = (g_ctx != nullptr || g_model != nullptr);
    unload_model();
    g_model_path.clear();

    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = false;
    }

    return env->NewStringUTF(had_model ? "Model disposed" : "Model already disposed");
}

extern "C"
JNIEXPORT jstring JNICALL
Java_com_example_new_1gguf_1llm_1app_MainActivity_runInferenceNative(
    JNIEnv * env,
    jobject /* this */,
    jstring prompt) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);

    if (!g_model_initialized || g_ctx == nullptr || g_model == nullptr) {
        g_last_generated_tokens = 0;
        return env->NewStringUTF("Model is not initialized");
    }

    const llama_vocab * vocab = llama_model_get_vocab(g_model);
    if (vocab == nullptr) {
        g_last_generated_tokens = 0;
        return env->NewStringUTF("Model vocabulary is unavailable");
    }

    llama_memory_t memory = llama_get_memory(g_ctx);
    if (memory != nullptr) {
        llama_memory_clear(memory, true);
    }

    if (prompt == nullptr) {
        g_last_generated_tokens = 0;
        return env->NewStringUTF("Prompt is null");
    }

    const std::string prompt_text = jstring_to_string(env, prompt);
    if (prompt_text.empty()) {
        g_last_generated_tokens = 0;
        return env->NewStringUTF("Prompt is empty");
    }

    std::vector<llama_token> prompt_tokens(
        std::max(32, static_cast<int>(prompt_text.size()) + 32));

    int n_prompt = llama_tokenize(
        vocab,
        prompt_text.c_str(),
        static_cast<int32_t>(prompt_text.size()),
        prompt_tokens.data(),
        static_cast<int32_t>(prompt_tokens.size()),
        true,
        true);

    if (n_prompt < 0) {
        prompt_tokens.resize(static_cast<size_t>(-n_prompt));
        n_prompt = llama_tokenize(
            vocab,
            prompt_text.c_str(),
            static_cast<int32_t>(prompt_text.size()),
            prompt_tokens.data(),
            static_cast<int32_t>(prompt_tokens.size()),
            true,
            true);
    }

    if (n_prompt <= 0) {
        g_last_generated_tokens = 0;
        return env->NewStringUTF("Tokenization failed");
    }

    prompt_tokens.resize(static_cast<size_t>(n_prompt));

    const int n_ctx = static_cast<int>(llama_n_ctx(g_ctx));
    const int reserved_for_generation = std::min(g_max_gen_tokens, std::max(256, n_ctx / 3));
    const int max_prompt_tokens = std::max(1, n_ctx - reserved_for_generation - 8);
    if (n_prompt > max_prompt_tokens) {
        prompt_tokens.erase(
            prompt_tokens.begin(),
            prompt_tokens.end() - static_cast<std::vector<llama_token>::difference_type>(max_prompt_tokens));
        n_prompt = max_prompt_tokens;
    }

    std::vector<llama_token> token_history = prompt_tokens;
    std::vector<llama_token> generated_tokens;
    generated_tokens.reserve(static_cast<size_t>(g_max_gen_tokens));

    int n_past = 0;
    for (int i = 0; i < n_prompt; ++i) {
        llama_token tok = prompt_tokens[static_cast<size_t>(i)];
        llama_pos pos = n_past;
        llama_seq_id seq_id_0 = 0;
        llama_seq_id * seq_id_ptr = &seq_id_0;
        int32_t n_seq_id = 1;
        int8_t logits = (i == n_prompt - 1) ? 1 : 0;

        llama_batch batch{};
        batch.n_tokens = 1;
        batch.token = &tok;
        batch.pos = &pos;
        batch.n_seq_id = &n_seq_id;
        batch.seq_id = &seq_id_ptr;
        batch.logits = &logits;

        if (llama_decode(g_ctx, batch) != 0) {
            g_last_generated_tokens = 0;
            return env->NewStringUTF("llama_decode failed while processing prompt");
        }
        n_past++;
    }

    std::string generated;
    generated.reserve(1024);

    for (int step = 0; step < g_max_gen_tokens; ++step) {
        const float * logits = llama_get_logits(g_ctx);
        if (logits == nullptr) {
            g_last_generated_tokens = static_cast<int>(generated_tokens.size());
            return env->NewStringUTF("Failed to access logits");
        }

        const int n_vocab = llama_vocab_n_tokens(vocab);
        if (n_vocab <= 0) {
            g_last_generated_tokens = static_cast<int>(generated_tokens.size());
            return env->NewStringUTF("Invalid vocabulary size");
        }

        const llama_token next_token = sample_next_token(logits, n_vocab, vocab, generated_tokens);
        if (next_token == LLAMA_TOKEN_NULL) {
            g_last_generated_tokens = static_cast<int>(generated_tokens.size());
            return env->NewStringUTF("Sampling failed");
        }

        const llama_token eos = llama_vocab_eos(vocab);
        if (next_token == eos) {
            break;
        }

        generated += token_to_text_piece(next_token);
        const std::string end_marker = "<|im_end|>";
        const size_t marker_pos = generated.find(end_marker);
        if (marker_pos != std::string::npos) {
            generated.erase(marker_pos);
            generated_tokens.push_back(next_token);
            break;
        }
        generated_tokens.push_back(next_token);
        token_history.push_back(next_token);

        llama_token sampled = next_token;
        llama_pos pos = n_past;
        llama_seq_id seq_id_0 = 0;
        llama_seq_id * seq_id_ptr = &seq_id_0;
        int32_t n_seq_id = 1;
        int8_t request_logits = 1;

        llama_batch next_batch{};
        next_batch.n_tokens = 1;
        next_batch.token = &sampled;
        next_batch.pos = &pos;
        next_batch.n_seq_id = &n_seq_id;
        next_batch.seq_id = &seq_id_ptr;
        next_batch.logits = &request_logits;

        if (llama_decode(g_ctx, next_batch) != 0) {
            break;
        }
        n_past++;
    }

    if (generated.empty()) {
        generated = "No output generated.";
    }

    g_last_generated_tokens = static_cast<int>(generated_tokens.size());

    return env->NewStringUTF(generated.c_str());
}

extern "C"
JNIEXPORT jint JNICALL
Java_com_example_new_1gguf_1llm_1app_MainActivity_getLastGeneratedTokensNative(
    JNIEnv * /* env */,
    jobject /* this */) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);
    return static_cast<jint>(g_last_generated_tokens);
}

extern "C"
JNIEXPORT void JNICALL
JNI_OnUnload(JavaVM * /* vm */, void * /* reserved */) {
    std::lock_guard<std::mutex> lock(g_llama_mutex);
    unload_model();
    if (g_backend_initialized) {
        llama_backend_free();
        g_backend_initialized = false;
    }
}
