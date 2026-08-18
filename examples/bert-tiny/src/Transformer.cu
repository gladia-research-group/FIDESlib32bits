#include "Transformer.cuh"

namespace FIDESlib::CKKS {

std::vector<std::vector<lbcrypto::Ciphertext<DCRTPoly>>> ct_tokens;
lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys_;
bool TIMING = false;

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> encoder(
    PtWeights_GPU& weights_layer, MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
    TransposePrecomputations_GPU& Tprecomp_gpu, std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& tokens,
    PtMasks_GPU& masks, EncoderConfiguration& conf, int layerNo) {
    constexpr bool PRINT = false;
    constexpr bool TIMING = true;
    std::chrono::time_point<std::chrono::system_clock> start_gpu, end_gpu;

    if (TIMING) {
        cudaDeviceSynchronize();
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    Context& cc = tokens[0][0].cc_;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> K, Q, V;
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> GPUResult_QKT, GPUResult_Sm_V, GPUResult_Output, GPUResult_Up,
        GPUResult_Down;

    dropMatrixLevel(tokens, conf.level_matmul);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(tokens, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2, "tokens",
                    false);

    PCMM_GPU(tokens, weights_layer.Wk, conf.blockSize, K, precomp_gpu, weights_layer.bk, masks.row_masks[conf.token_length]);
    PCMM_GPU(tokens, weights_layer.Wq, conf.blockSize, Q, precomp_gpu, weights_layer.bq, masks.row_masks[conf.token_length]);
    PCMM_GPU(tokens, weights_layer.Wv, conf.blockSize, V, precomp_gpu, weights_layer.bv, masks.row_masks[conf.token_length]);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    ////////////////////////////// Multi Head Attention /////////////////////////////////
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> Sm_V, Sm_V2;

    MatrixBootstrap(Q, conf.numSlots, conf.prescale);
    MatrixBootstrap(K, conf.numSlots, conf.prescale);

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> QKT1, QKT2;

    auto Q1 = MatrixMask(Q, masks.head_masks[0]);
    auto Q2 = MatrixMask(Q, masks.head_masks[1]);

    auto K1 = MatrixMask(K, masks.head_masks[0]);
    auto K2 = MatrixMask(K, masks.head_masks[1]);

    auto K1_T = MatrixTranspose_GPU(std::move(K1), conf.blockSize, Tprecomp_gpu);
    auto K2_T = MatrixTranspose_GPU(std::move(K2), conf.blockSize, Tprecomp_gpu);

    CCMM_GPU(Q1, K1_T, conf.blockSize, QKT1, precomp_gpu);
    CCMM_GPU(Q2, K2_T, conf.blockSize, QKT2, precomp_gpu);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "CCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    FIDESlib::CKKS::Plaintext double_mask(QKT1[0][0].cc_);
    double_mask.copy(masks.row_masks[conf.token_length]);
    double_mask.multPt(double_mask, masks.head_masks[0], true);

    QKT1 = MatrixMask(QKT1, double_mask);
    QKT2 = MatrixMask(QKT2, double_mask);

    // offset for sst2 
    if (layerNo == 1){ 
        MatrixAddScalar(QKT1, -0.25); 
        MatrixAddScalar(QKT2, -0.25); 
    }

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT2: ", false);

    MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);
    EvalSoftmax_Matrix(QKT1, ct_tokens[0][0], keys_.secretKey, masks.mask_tokens[conf.token_length],
                       masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize,
                       conf.bStepAcc, conf.token_length, true);
    MatrixBootstrap(QKT1, conf.numSlots, conf.prescale);

    MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);
    EvalSoftmax_Matrix(QKT2, ct_tokens[0][0], keys_.secretKey, masks.mask_tokens[conf.token_length],
                       masks.mask_broadcast, masks.mask_layernorm[0], masks.mask_max, conf.numSlots, conf.blockSize,
                       conf.bStepAcc, conf.token_length, true);
    MatrixBootstrap(QKT2, conf.numSlots, conf.prescale);

    QKT1 = MatrixMask(QKT1, double_mask);
    QKT2 = MatrixMask(QKT2, double_mask);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Softmax took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT1, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(QKT2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "QKT2: ", false);

    auto V1 = MatrixMask(V, masks.head_masks[0]);
    auto V2 = MatrixMask(V, masks.head_masks[1]);
    MatrixRotate(V2, conf.blockSize / conf.num_heads);

    CCMM_GPU(QKT1, V1, conf.blockSize, Sm_V, precomp_gpu);
    CCMM_GPU(QKT2, V2, conf.blockSize, Sm_V2, precomp_gpu);

    Sm_V = MatrixMask(Sm_V, masks.head_masks[0]);
    Sm_V2 = MatrixMask(Sm_V2, masks.head_masks[0]);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V1: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V2, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V2: ", false);
    MatrixRotate(Sm_V2, -conf.blockSize / conf.num_heads);

    Sm_V = MatrixAdd(Sm_V, Sm_V2);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "CCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    //////////////////////////////////////////////////////////////////////////////////////

    K.clear(); Q.clear(); V.clear();
    QKT1.clear(); QKT2.clear();
    MatrixBootstrap(Sm_V, conf.numSlots);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(Sm_V, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Sm_V: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Boot took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Output CCMM
    // dropMatrixLevel(Sm_V, conf.level_matmul);
    PCMM_GPU(Sm_V, weights_layer.Wo, conf.blockSize, GPUResult_Output, precomp_gpu, weights_layer.bo, masks.row_masks[conf.token_length]);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_output: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Layer Norm
    GPUResult_Output = MatrixAdd(GPUResult_Output, tokens);
    tokens.clear();
    
    EvalLayerNorm_Matrix(GPUResult_Output, ct_tokens[0][0], keys_.secretKey, masks.mask_layernorm,
                         masks.row_masks[conf.token_length], weights_layer.Wln1, weights_layer.bln1, conf.numSlots,
                         conf.blockSize, conf.bStepAcc, true);

    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Output, conf.numSlots);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Output, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "LN took: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count())
                  << " ms." << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Up PCMM
    dropMatrixLevel(GPUResult_Output, conf.level_matmul - 3);
    PCMM_GPU(GPUResult_Output, weights_layer.Wu, conf.blockSize, GPUResult_Up, precomp_gpu, weights_layer.bu, masks.row_masks[conf.token_length]);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "RELU input: ", false);

    // dropMatrixLevel(GPUResult_Up, conf.level_matmul);

    // ReLU
    MatrixBootstrap(GPUResult_Up, conf.numSlots);
    EvalGelu_Matrix(GPUResult_Up, conf.numSlots);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Up, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "RELU: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "Gelu took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Down PCMM
    // MatrixBootstrap(GPUResult_Up, conf.numSlots);
    // dropMatrixLevel(GPUResult_Up, conf.level_matmul - 2);
    PCMM_GPU(GPUResult_Up, weights_layer.Wd, conf.blockSize, GPUResult_Down, precomp_gpu, weights_layer.bd, masks.row_masks[conf.token_length]);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", false);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "Result_Down: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "PCMM took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    // Layer Norm
    GPUResult_Down = MatrixAdd(GPUResult_Down, GPUResult_Output);
    // MatrixBootstrap(GPUResult_Down, conf.numSlots);

    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN input: ", false);

    EvalLayerNorm_Matrix(GPUResult_Down, ct_tokens[0][0], keys_.secretKey, masks.mask_layernorm,
                         masks.row_masks[conf.token_length], weights_layer.Wln2, weights_layer.bln2, conf.numSlots,
                         conf.blockSize, conf.bStepAcc, true);

    if constexpr (PRINT)
        std::cout << "# ------- bts ------- " << std::endl;
    MatrixBootstrap(GPUResult_Down, conf.numSlots);
    if constexpr (PRINT)
        printMatrix(decryptGPUMatrix(GPUResult_Down, keys_.secretKey, ct_tokens, conf.numSlots, conf.blockSize), 2, 2,
                    "LN: ", false);

    if (TIMING) {
        cudaDeviceSynchronize();
        end_gpu = std::chrono::high_resolution_clock::now();
        std::cout << "LN took: "
                  << (std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()) << " ms."
                  << std::endl;
        start_gpu = std::chrono::high_resolution_clock::now();
    }
    return GPUResult_Down;
}

// csv file read and tokenization
struct ColMap {int s1=-1, s2=-1, text=-1, label=-1, idx=-1;};
static std::string lower(std::string s){ std::transform(s.begin(),s.end(),s.begin(),::tolower); return s; }
static inline bool is_integer(const std::string& s);
static inline bool is_int_01(const std::string& s);
enum class SampleKind { SingleSentence, PairSentence };
static SampleKind dataset_kind(std::string dataset);
static ColMap build_colmap(const std::vector<std::string>& header);
static void normalize_tokenized_punct(std::string& s);
static bool parse_csv_row(const std::string& line, std::vector<std::string>& outCols);
static inline void strip_inplace(std::string& s); 
static inline void rstrip_inplace(std::string& s); 


// -------------------- unified processor --------------------
void process_sentences_from_csv(std::string& file_path,
                                std::string& output_file,
                                std::string& model_name,
                                std::string& model_path,
                                std::string& output_path,
                                EncoderConfiguration& base_conf,
                                lbcrypto::PublicKey<lbcrypto::DCRTPoly>& publicKey,
                                FIDESlib::CKKS::Context& GPUcc,
                                std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                                PtWeights_GPU& weights_layer0,
                                PtWeights_GPU& weights_layer1,
                                PtMasks_GPU& masks,
                                MatrixMatrixProductPrecomputations_GPU& precomp_gpu,
                                TransposePrecomputations_GPU& Tprecomp_gpu,
                                lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& cc,
                                lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& sk,
                                const std::string& dataset,
                                int test_case, double num_sigma)
{
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Could not open file: " << file_path << std::endl;
        return;
    }

    const SampleKind kind = dataset_kind(dataset);

    size_t total_counter = 0, correct_counter = 0;
    std::string line;
    std::vector<std::string> cols;

    // ---------- Header detection + column map ----------
    ColMap cmap;               // {s1,s2,text,label,idx} all -1 by default
    bool have_map = false;

    {
        std::streampos pos = file.tellg();
        std::string first;
        if (std::getline(file, first)) {
            std::vector<std::string> header;
            parse_csv_row(first, header);
            // header-like if it contains any of these tokens
            bool headerish = false;
            for (auto h : header) {
                h = lower(h);
                if (h == "sentence1" || h == "sentence2" || h == "sentence" ||
                    h == "premise"   || h == "hypothesis" ||
                    h == "label"     || h == "idx" || h == "id" || h == "text") {
                    headerish = true; break;
                }
            }
            if (headerish) {
                cmap = build_colmap(header);
                have_map = (cmap.label >= 0) || (cmap.text >= 0) || (cmap.s1 >= 0);

                if (dataset_kind(dataset) == SampleKind::SingleSentence) {
                    if (cmap.text < 0 || cmap.label < 0) {
                        std::cerr << "ERROR: Header must have 'sentence' and 'label' columns.\n";
                        return;
                    }
                }

            } else {
                file.clear();
                file.seekg(pos);
            }
        } else {
            file.clear();
            file.seekg(pos);
        }
    }

    // ---------- Rows ----------
    while (std::getline(file, line)) {
        try {
            rstrip_inplace(line);
            if (line.empty()) continue;

            if (!parse_csv_row(line, cols)) continue;
            for (auto& c : cols) strip_inplace(c);

            int label = -1;
            std::string idx_str;
            std::string s1, s2, sentence;

            if (have_map) {
                // ---- Header-driven parsing ----
                if (cmap.label >= 0 && cmap.label < (int)cols.size()) {
                    try { label = std::stoi(cols[cmap.label]); } catch (...) { label = -1; }
                }
                if (cmap.idx >= 0 && cmap.idx < (int)cols.size())
                    idx_str = cols[cmap.idx];

                if (kind == SampleKind::PairSentence) {
                    if (cmap.s1 >= 0 && cmap.s1 < (int)cols.size()) s1 = cols[cmap.s1];
                    if (cmap.s2 >= 0 && cmap.s2 < (int)cols.size()) s2 = cols[cmap.s2];
                    if (s1.empty() || s2.empty() || label < 0) continue;
                    normalize_tokenized_punct(s1);
                    normalize_tokenized_punct(s2);
                } else {
                    if (cmap.text >= 0 && cmap.text < (int)cols.size())
                        sentence = cols[cmap.text];
                    else if (!cols.empty())
                        sentence = cols.front();
                    if (sentence.empty() || label < 0) continue;
                    normalize_tokenized_punct(sentence);
                }
            } else {
                // ---- Fallback: heuristic (no header present) ----
                // 1) label is last clean 0/1
                int label_idx = -1;
                for (int i = (int)cols.size() - 1; i >= 0; --i) {
                    if (is_int_01(cols[i])) { label = std::stoi(cols[i]); label_idx = i; break; }
                }
                if (label < 0) continue;

                // 2) idx = nearest previous pure integer
                for (int i = label_idx - 1; i >= 0; --i) {
                    if (is_integer(cols[i])) { idx_str = cols[i]; break; }
                }
                if (idx_str.empty()) idx_str = std::to_string(total_counter);

                // 3) gather text fields
                std::vector<std::string> texts;
                for (int i = 0; i < (int)cols.size(); ++i) {
                    if (i == label_idx) continue;
                    if (i != label_idx && is_integer(cols[i]) && (i > label_idx - 2)) continue;
                    texts.push_back(cols[i]);
                }
                if (texts.empty()) continue;

                if (kind == SampleKind::SingleSentence) {
                    sentence = texts.front();
                    normalize_tokenized_punct(sentence);
                } else {
                    s1 = texts.size() >= 1 ? texts[0] : std::string();
                    s2 = texts.size() >= 2 ? texts[1] : std::string();
                    if (s1.empty() || s2.empty()) continue;
                    normalize_tokenized_punct(s1);
                    normalize_tokenized_punct(s2);
                }
            }

            if (idx_str.empty()) idx_str = std::to_string(total_counter);

            // ---------- Tokenize ----------
            EncoderConfiguration conf = base_conf;
            const std::string iter_out_fname = output_file; 
            const std::string token_path = (std::filesystem::path(model_path) / iter_out_fname).string();

            if (kind == SampleKind::SingleSentence) {
                conf.token_length = tokenizer(sentence, dataset, model_name, model_path, iter_out_fname);
            } else {
                conf.token_length = tokenizer_pair(s1, s2, model_name, model_path, iter_out_fname);
            }
            if (conf.token_length <= 0 || conf.token_length > 128) {
                // std::cerr << "[WARN] token_length==" << conf.token_length << " skip idx=" << idx_str << "\n";
                continue;
            }


            // ---------- Clone ct_tokens ----------
            std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ct_tokens_clone;
            ct_tokens_clone.resize(ct_tokens.size());
            for (size_t i = 0; i < ct_tokens.size(); ++i) {
                ct_tokens_clone[i].resize(ct_tokens[i].size());
                for (size_t j = 0; j < ct_tokens[i].size(); ++j) {
                    ct_tokens_clone[i][j] = ct_tokens[i][j]->Clone();
                }
            }

            // ---------- Terminal output ----------
            std::cout << "\n///////////////////////////////////////\n";
            if (kind == SampleKind::SingleSentence) {
                std::cout << "Input: '" << sentence << "'" << ", " << conf.token_length << "\n";
            } else {
                std::cout << "Sentence1: \"" << s1 << "\"\n";
                std::cout << "Sentence2: \"" << s2 << "\"\n";
                std::cout << "Length: " << conf.token_length << "\n";
            }
            std::cout << "Label: " << label << "\n";

            // ---------- Log ----------
            {
                std::ofstream outFile(output_path, std::ios::app);
                outFile << "\n///////////////////////////////////////\n";
                if (kind == SampleKind::SingleSentence) {
                    outFile << "Input: '" << sentence << "'" << ", " << conf.token_length << "\n";
                } else {
                    outFile << "Sentence1: \"" << s1 << "\"\n";
                    outFile << "Sentence2: \"" << s2 << "\"\n";
                    outFile << "Length: " << conf.token_length << "\n";
                }
                outFile << "Label: " << label << "\n";
            }

            // ---------- Encrypt to GPU ----------
            std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> tokens_gpu;
            encryptMatrixtoGPU(token_path, tokens_gpu, publicKey, GPUcc,
                               conf.numSlots, conf.blockSize,
                               conf.rows, conf.cols,
                               conf.level_matmul);

            // ---------- Run ----------
            cudaDeviceSynchronize();
            auto start_gpu = std::chrono::high_resolution_clock::now();

            tokens_gpu = encoder(weights_layer0, precomp_gpu, Tprecomp_gpu,
                                        tokens_gpu, masks, conf, 0);
            tokens_gpu = encoder(weights_layer1, precomp_gpu, Tprecomp_gpu,
                                        tokens_gpu, masks, conf, 1);

            uint32_t class_pred = classifier(cc, tokens_gpu, sk, ct_tokens_clone,
                                             precomp_gpu, weights_layer1, masks,
                                             conf.numSlots, conf.blockSize, conf.token_length,
                                             true, output_path);

            cudaDeviceSynchronize();
            auto end_gpu = std::chrono::high_resolution_clock::now();

            tokens_gpu.clear();

            ++total_counter;
            if (class_pred == static_cast<uint32_t>(label)) ++correct_counter;

            {
                std::ofstream outFile2(output_path, std::ios::app);
                outFile2 << "took: "
                         << std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count()
                         << " ms.\n";
                outFile2 << "Accuracy: " << correct_counter << "/" << total_counter << "\n";
            }
            std::cout << "Accuracy: " << correct_counter << "/" << total_counter << std::endl;
        }
        catch (const std::exception& e) {
            std::cerr << "[EXCEPTION] " << e.what() << " — skipping row.\n";
        }
        catch (...) {
            std::cerr << "[EXCEPTION] unknown — skipping row.\n";
        }
    }
}


int32_t classifier(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                   std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& input,
                   lbcrypto::PrivateKey<lbcrypto::DCRTPoly>& privateKey,
                   std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ct_tokens,
                   const MatrixMatrixProductPrecomputations_GPU& precomp, PtWeights_GPU& weights_layer,
                   PtMasks_GPU& masks, int numSlots, int blockSize, int token_length, bool bts,
                   const std::string& output_path) {

    bool constexpr PRINT = false;

    FIDESlib::CKKS::Context& GPUcc = input[0][0].cc_;

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> result, result_f;
    // PCMM_2(input, weights_layer.Wp, blockSize, result, precomp, weights_layer.bp, masks.row_masks[token_length]);
    PCMM_GPU(input, weights_layer.Wp, blockSize, result, precomp, weights_layer.bp, masks.row_masks[token_length]);

    if (PRINT)
        printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "PCMM", false);

    evalTanh(result[0][0], numSlots, -40, 40, true);  // -10, 10 -> -20, 20

    if (PRINT)
        printMatrix(decryptGPUMatrix(result, keys_.secretKey, ct_tokens, numSlots, blockSize), 2, 2, "Tanh", false);

    FIDESlib::CKKS::Ciphertext result_0(GPUcc), result_1(GPUcc);
    result_0.copy(result[0][0]);
    result_0.multPt(weights_layer.Wc[0][0], false);
    Accumulate(result_0, 4, 1, blockSize);
    result_0.addPt(weights_layer.bc[0][0]);

    FIDESlib::CKKS::RawCipherText raw_res;
    result_0.store(raw_res);
    auto result_gpu0(ct_tokens[0][0]->Clone());
    GetOpenFHECipherText(result_gpu0, raw_res);

    Plaintext weights_rotated(GPUcc), bias_rotated(GPUcc);
    weights_rotated.copy(weights_layer.Wc[0][0]);
    weights_rotated.automorph(blockSize);
    bias_rotated.copy(weights_layer.bc[0][0]);
    bias_rotated.automorph(blockSize);

    result_1.copy(result[0][0]);
    result_1.multPt(weights_rotated, false);
    Accumulate(result_1, 4, 1, blockSize);
    result_1.addPt(bias_rotated);

    FIDESlib::CKKS::RawCipherText raw_res1;
    result_1.store(raw_res1);
    auto result_gpu1(ct_tokens[0][0]->Clone());
    GetOpenFHECipherText(result_gpu1, raw_res1);

    try {
        lbcrypto::Plaintext pt_result_gpu0;
        context->Decrypt(privateKey, result_gpu0, &pt_result_gpu0);
        double result0 = pt_result_gpu0->GetRealPackedValue()[0];

        lbcrypto::Plaintext pt_result_gpu1;
        context->Decrypt(privateKey, result_gpu1, &pt_result_gpu1);
        double result1 = pt_result_gpu1->GetRealPackedValue()[0];

        int yhat = 1;
        if (result0 > result1) {
            yhat = 0;
        }

        std::ofstream outFile(output_path, std::ios::app);
        outFile << "logits: " << result0 << ", " << result1 << std::endl;
        outFile << "Class: " << yhat << std::endl;
        outFile.close();

        // terminal output
        std::cout << "logits: " << result0 << ", " << result1 << std::endl;
        std::cout << "Class: " << yhat << std::endl;
        return yhat;

    } catch (const std::exception& e) {
        // std::cerr << "none. Decryption failed: " << e.what() << std::endl;
        return -1;
    } catch (...) {
        // std::cerr << "none. Unknown error occurred during decryption." << std::endl;
        return -1;
    }
    std::cout << std::endl;
}

std::vector<int> GenerateRotationIndices_GPU(int blockSize, int bStep, int bStepAcc, int colSize) {

    if (colSize == 0) { colSize = blockSize; }
    // JKLS MatMul rotation indices
    std::vector<int32_t> rotation_indices_MM = GenerateMatMulRotationIndices_GPU(blockSize, bStep, colSize);
    // Multi-head Attention rotation indices
    std::vector<int32_t> rotation_indices_MHA = GenerateMatMulRotationIndices_GPU(64, bStep, colSize);   // d_k = 64

    // Transpose rotation indices
    std::vector<int> rotation_indices_T = GenerateTransposeRotationIndices_GPU(blockSize, bStep);

    std::vector<int> rotsum_indices = {
        1,  2,  3,  4,   8,   16,  32,  64,  8192, 0,   -1,  -2,
        -3, -4, -8, -16, -32, -64, 127, -15, -31,  -47, -63, -127};  // 127 is for pooling, -blockSize for Concat

    std::vector<int> accum_indices = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, 1, blockSize);
    std::vector<int> accum_indices2 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize);
    std::vector<int> accum_indices3 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 2);
    std::vector<int> accum_indices4 = FIDESlib::CKKS::GetAccumulateRotationIndices(bStepAcc, blockSize, blockSize / 4);
    std::vector<int> broad_indices = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize);
    std::vector<int> broad_indices2 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 2);
    std::vector<int> broad_indices3 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 4);
    std::vector<int> broad_indices4 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 8);
    std::vector<int> broad_indices5 = FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, 1, blockSize / 16);  // 4
    std::vector<int> broad_indices6 =
        FIDESlib::CKKS::GetbroadcastRotationIndices(bStepAcc, blockSize, blockSize * blockSize);

    // if (blockSize == 128) rotsum_indices = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

    // Merge the rotation indices and remove duplicates

    std::set<int32_t> merged_set(rotsum_indices.begin(), rotsum_indices.end());
    merged_set.insert(rotation_indices_MM.begin(), rotation_indices_MM.end());
    merged_set.insert(rotation_indices_MHA.begin(), rotation_indices_MHA.end());
    merged_set.insert(rotation_indices_T.begin(), rotation_indices_T.end());
    merged_set.insert(accum_indices.begin(), accum_indices.end());
    merged_set.insert(accum_indices2.begin(), accum_indices2.end());
    merged_set.insert(accum_indices3.begin(), accum_indices3.end());
    merged_set.insert(accum_indices4.begin(), accum_indices4.end());
    merged_set.insert(broad_indices.begin(), broad_indices.end());
    merged_set.insert(broad_indices2.begin(), broad_indices2.end());
    merged_set.insert(broad_indices3.begin(), broad_indices3.end());
    merged_set.insert(broad_indices4.begin(), broad_indices4.end());
    merged_set.insert(broad_indices5.begin(), broad_indices5.end());
    merged_set.insert(broad_indices6.begin(), broad_indices6.end());
    std::vector<int32_t> rotation_indices(merged_set.begin(), merged_set.end());

    return rotation_indices;
}

void MatrixBootstrap(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int numSlots, bool input_prescaled) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            Bootstrap(matrix[i][j], numSlots, input_prescaled);
        }
    }
}

void MatrixAddScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].addScalar(value);
        }
    }
}

void MatrixMultScalar(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, double value) {
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].multScalar(value);
        }
    }
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixAdd(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix,
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix2) {

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
    masked_matrix.reserve(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc_);
            masked_ct.copy(matrix[i][j]);
            masked_ct.add(matrix2[i][j]);
            row.emplace_back(std::move(masked_ct));
        }
        masked_matrix.emplace_back(std::move(row));
    }
    return masked_matrix;
}

void MatrixRotate(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, int index) {
    FIDESlib::CKKS::ContextData& cc = matrix[0][0].cc;
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            matrix[i][j].rotate(index);
        }
    }
}

std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> MatrixMask(
    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& matrix, FIDESlib::CKKS::Plaintext& mask) {

    std::vector<std::vector<FIDESlib::CKKS::Ciphertext>> masked_matrix;
    masked_matrix.reserve(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        std::vector<FIDESlib::CKKS::Ciphertext> row;
        row.reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            FIDESlib::CKKS::Ciphertext masked_ct(matrix[i][j].cc_);
            masked_ct.copy(matrix[i][j]);
            masked_ct.multPt(mask);
            // masked_ct.rescale();
            row.emplace_back(std::move(masked_ct));
        }
        masked_matrix.emplace_back(std::move(row));
    }
    return masked_matrix;
}

void dropMatrixLevel(std::vector<std::vector<FIDESlib::CKKS::Ciphertext>>& in, int level) {
    for (auto& row : in)
        for (auto& ct : row) {
            if (ct.NoiseLevel == 2)
                ct.rescale();
            if (ct.getLevel() > level) {
                ct.dropToLevel(level);
                assert(ct.getLevel() == level);
            }
        }
}

static std::string sh_quote(const std::string& s);
static std::string sh_single_quote(const std::string& s);

int tokenizer(const std::string& sentence,
              const std::string& dataset,         
              const std::string& model_name,
              const std::string& model_path,
              const std::string& output_filename)
{
    std::filesystem::path dir(model_path);
    std::filesystem::path file(output_filename);
    std::filesystem::path full_path = dir / file;
    std::string file_mode;
    if (std::filesystem::exists(full_path)) {
        file_mode = "update"; 
    } else {
        file_mode = "create";
    }

    const std::string script_path = "../src/python/ExtractEmbeddings.py"; 

    // Build: python3 ExtractEmbeddings.py <sentence> <dataset> <model_name> <out_dir> <out_fname>
    // NOTE: model_path is a directory (out_dir).
    std::string cmd = "python3 "
        + sh_quote(script_path) + " "
        + sh_quote(sentence)    + " "
        + sh_quote(dataset)     + " "
        + sh_quote(model_name)  + " "
        + sh_quote(model_path)  + " "
        + sh_quote(output_filename) + " "
        + sh_quote(file_mode);

    std::array<char, 256> buffer{};
    std::string result;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen() failed to run tokenizer");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    int exitCode = pclose(pipe);
    if (exitCode != 0) {
        throw std::runtime_error("Tokenizer script failed, exit code: " + std::to_string(exitCode)
                                 + ", stdout: " + result);
    }

    // Parse last integer from stdout (seq length)
    // Trim spaces/newlines:
    auto trim = [](std::string& s){
        size_t a = s.find_first_not_of(" \t\r\n");
        size_t b = s.find_last_not_of(" \t\r\n");
        if (a == std::string::npos) { s.clear(); return; }
        s = s.substr(a, b - a + 1);
    };
    trim(result);

    try {
        return std::stoi(result);
    } catch (...) {
        std::cerr << "[WARNING] Could not parse token count: \"" << result << "\"\n";
        return 0;
    }
}


int tokenizer_pair(const std::string& sentence1, const std::string& sentence2, const std::string& model_name,
                   const std::string& output_path, const std::string& output_filename) {
    const std::string script_path = "../src/python/";
    const std::string script = "ExtractEmbeddings_pair.py";  

    const std::string s1 = sh_single_quote(sentence1);
    const std::string s2 = sh_single_quote(sentence2);
    const std::string m  = sh_single_quote(model_name);
    const std::string op = sh_single_quote(output_path);
    const std::string of = sh_single_quote(output_filename);

    const std::string cmd = "python3 " + script_path + script + " "
                          + s1 + " " + s2 + " " + m + " " + op + " " + of;

    std::array<char, 128> buffer{};
    std::string result;
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe)
        throw std::runtime_error("popen() failed to run the tokenizer_pair script");

    while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
        result += buffer.data();
    }
    int exitCode = pclose(pipe);
    if (exitCode != 0) {
        throw std::runtime_error("tokenizer_pair script failed, exit code: " + std::to_string(exitCode));
    }

    try {
        return std::stoi(result);  // returns seq_len (including special tokens)
    } catch (...) {
        std::cerr << "[WARNING] Failed to parse token count from script output: \"" << result << "\"\n";
        return 0;
    }
}
size_t CountNumTokens(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << file_path << std::endl;
        return 0;
    }

    size_t count = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Trim leading/trailing whitespace
        line.erase(0, line.find_first_not_of(" \t\n\r\f\v"));
        line.erase(line.find_last_not_of(" \t\n\r\f\v") + 1);

        if (!line.empty()) {
            ++count;
        }
    }

    file.close();
    return count;
}


// -------------------- tokenization --------------------

static inline void ltrim_inplace(std::string& s) {
    size_t i = 0;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
    if (i) s.erase(0, i);
}
static inline void rtrim_inplace(std::string& s) {
    size_t i = s.size();
    while (i > 0 && std::isspace(static_cast<unsigned char>(s[i - 1]))) --i;
    if (i < s.size()) s.erase(i);
}
static inline void strip_inplace(std::string& s) { ltrim_inplace(s); rtrim_inplace(s); }
static inline void rstrip_inplace(std::string& s) { rtrim_inplace(s); }

static bool parse_csv_row(const std::string& line, std::vector<std::string>& outCols) {
    outCols.clear();
    std::string field;
    bool inQuotes = false;
    char quote = '"';

    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (inQuotes) {
            if (c == quote) {
                if (i + 1 < line.size() && line[i + 1] == quote) { // doubled quote -> literal
                    field.push_back(quote);
                    ++i;
                } else {
                    inQuotes = false;
                }
            } else {
                field.push_back(c);
            }
        } else {
            if (c == ',') {
                outCols.push_back(field);
                field.clear();
            } else if (c == '"' || c == '\'') {
                if (field.empty()) { inQuotes = true; quote = c; }
                else { field.push_back(c); } // treat embedded as literal
            } else {
                field.push_back(c);
            }
        }
    }
    outCols.push_back(field);
    return true;
}

static void normalize_tokenized_punct(std::string& s) {
    // normalize whitespace kinds to spaces
    for (char& c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';

    // collapse multiple spaces
    {
        std::string t; t.reserve(s.size());
        bool prevSpace = false;
        for (char c : s) {
            bool sp = std::isspace(static_cast<unsigned char>(c));
            if (!(sp && prevSpace)) t.push_back(sp ? ' ' : c);
            prevSpace = sp;
        }
        s.swap(t);
    }

    // trim edges
    strip_inplace(s);

    // drop leading/trailing quote/comma noise
    while (!s.empty() && (s.front() == '"' || s.front() == '\'' || s.front() == ',' || std::isspace(static_cast<unsigned char>(s.front()))))
        s.erase(s.begin());
    while (!s.empty() && (s.back()  == '"' || s.back()  == '\'' || s.back()  == ',' || std::isspace(static_cast<unsigned char>(s.back()))))
        s.pop_back();

    // remove spaces BEFORE punctuation: "word ,"
    {
        std::string t; t.reserve(s.size());
        for (size_t i = 0; i < s.size(); ++i) {
            if (s[i]==' ' && i+1<s.size()) {
                char p = s[i+1];
                if (p==','||p=='.'||p=='!'||p=='?'||p==';'||p==':') continue;
            }
            t.push_back(s[i]);
        }
        s.swap(t);
    }

    // tighten around apostrophes: doesn ' t -> doesn't
    {
        std::string t; t.reserve(s.size());
        for (size_t i=0;i<s.size();++i) {
            if (s[i]==' ' && ((i+1<s.size() && s[i+1]=='\'') || (i>0 && s[i-1]=='\''))) continue;
            t.push_back(s[i]);
        }
        s.swap(t);
    }

    // tighten spaces inside quotes
    {
        // after opening quote
        std::string t; t.reserve(s.size());
        for (size_t i=0;i<s.size();++i) {
            t.push_back(s[i]);
            if ((s[i]=='"'||s[i]=='\'') && i+1<s.size() && s[i+1]==' ') {
                size_t j = i+1;
                while (j<s.size() && s[j]==' ') ++j;
                if (j<s.size()) t.push_back(s[j]);
                i = j;
            }
        }
        s.swap(t);
        // before closing quote
        std::string u; u.reserve(s.size());
        for (size_t i=0;i<s.size();++i) {
            if (s[i]==' ' && i+1<s.size() && (s[i+1]=='"'||s[i+1]=='\'')) continue;
            u.push_back(s[i]);
        }
        s.swap(u);
    }

    strip_inplace(s);
}

static ColMap build_colmap(const std::vector<std::string>& header) {
    ColMap m;
    for (int i=0;i<(int)header.size();++i) {
        const std::string h = lower(header[i]);
        if (h == "sentence1" || h == "premise"   || h == "question1" || h == "s1") m.s1 = i;
        else if (h == "sentence2"|| h == "hypothesis"|| h == "question2" || h == "s2") m.s2 = i;
        else if (h == "label"    || h == "gold_label" || h == "y") m.label = i;
        else if (h == "idx"      || h == "id" || h == "index") m.idx = i;
        else if (h == "sentence" || h == "text") m.text = i; // single-sentence datasets
    }
    return m;
}

// -------------------- dataset routing --------------------
static SampleKind dataset_kind(std::string dataset) {
    std::transform(dataset.begin(), dataset.end(), dataset.begin(), ::tolower);
    if (dataset=="rte" || dataset=="mrpc" || dataset=="qqp" || dataset=="stsb" || dataset=="qnli" || dataset=="mnli")
        return SampleKind::PairSentence;
    return SampleKind::SingleSentence; // sst2, cola, imdb, ...
}

static inline bool is_int_01(const std::string& s) {
    std::string t = s; strip_inplace(t);
    if (t=="0"||t=="1") return true;
    if (!t.empty() && (t[0]=='+'||t[0]=='-')) {
        std::string r = t.substr(1);
        return (r=="0"||r=="1");
    }
    return false;
}

static inline bool is_integer(const std::string& s) {
    std::string t = s; strip_inplace(t);
    if (t.empty()) return false;
    size_t i=0; if (t[0]=='+'||t[0]=='-') ++i;
    if (i==t.size()) return false;
    for (; i<t.size(); ++i) if (!std::isdigit(static_cast<unsigned char>(t[i]))) return false;
    return true;
}

static std::string sh_single_quote(const std::string& s) {
    // Returns the shell-safe single-quoted version of s
    // 'foo'bar' -> 'foo'"'"'bar'
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out.append("'\"'\"'");  // end ', insert "', start '
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

static std::string sh_quote(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') out += "'\"'\"'";  // close, insert ', reopen
        else out.push_back(c);
    }
    out.push_back('\'');
    return out;
}
}  // namespace FIDESlib::CKKS
