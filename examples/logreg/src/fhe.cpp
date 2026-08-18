#include "fhe.hpp"
#include "Definitions.hpp"
#include <iostream>
#include <vector>

fideslib::CryptoContext<fideslib::DCRTPoly> cc = nullptr;
fideslib::KeyPair<fideslib::DCRTPoly> keys;
uint32_t depth = 20;

static fideslib::Plaintext mask_0 = nullptr;
static fideslib::Plaintext mask_1 = nullptr;
static fideslib::Plaintext mask_3 = nullptr;

static fideslib::PrivateKey<fideslib::DCRTPoly> sk;

fideslib::Ciphertext<fideslib::DCRTPoly> encrypt_data(const std::vector<double>& data, const fideslib::PublicKey<fideslib::DCRTPoly>& pk, int scale_deg, int level) {
	auto pt = cc->MakeCKKSPackedPlaintext(data, scale_deg, level);
	return cc->Encrypt(pk, pt);
}

std::vector<double> decrypt_data(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, const fideslib::PrivateKey<fideslib::DCRTPoly>& sk, size_t num_slots) {
	fideslib::Plaintext pt;
	cc->Decrypt(ct, sk, &pt);
	std::vector<double> result;
	for (const auto& v : pt->GetCKKSPackedValue())
		result.push_back(v.real());
	return result;
}

uint32_t create_context(bool inference) {
	// Boot every 2 iterations parameters.
	uint32_t scale_mod = 50;
	uint32_t first_mod = 55;
	uint32_t digits	   = sparse_encaps ? 3 : 5;
	depth			   = sparse_encaps ? (slow ? 23 : 22) : 26;

	fideslib::CCParams<fideslib::CryptoContextCKKSRNS> params;
	params.SetScalingModSize(scale_mod);
	params.SetFirstModSize(first_mod);
	params.SetRingDim(ringDim);
	params.SetBatchSize(numSlots);
	params.SetSecurityLevel(fideslib::HEStd_NotSet);
	params.SetScalingTechnique(fideslib::FLEXIBLEAUTO);
	params.SetKeySwitchTechnique(fideslib::HYBRID);
	params.SetSecretKeyDist(sparse_encaps ? fideslib::SPARSE_TERNARY : fideslib::UNIFORM_TERNARY);
	params.SetNumLargeDigits(digits);
	params.SetDevices(std::vector<int>(devices));
	params.SetMultiplicativeDepth(depth);

	cc = GenCryptoContext(params);
	cc->Enable(fideslib::FHE);
	cc->Enable(fideslib::PKE);
	cc->Enable(fideslib::LEVELEDSHE);
	cc->Enable(fideslib::KEYSWITCH);
	cc->Enable(fideslib::ADVANCEDSHE);

	return depth;
}

void prepare_context(const fideslib::KeyPair<fideslib::DCRTPoly>& k, size_t cols, size_t rows) {
	cc->EvalMultKeyGen(k.secretKey);

	std::vector<int> rot_idx;
	for (size_t j = 1; j < cols; j <<= 1) {
		rot_idx.push_back(j);
		rot_idx.push_back(-j);
	}
	for (size_t i = cols; i < cols * rows; i <<= 1)
		rot_idx.push_back(i);
	cc->EvalRotateKeyGen(k.secretKey, rot_idx);

	cc->EvalBootstrapSetup(levelBudget, bStep, cols, 0);
	cc->EvalBootstrapKeyGen(k.secretKey, cols);
	cc->LoadContext(k.publicKey);

	// Create masks for activation function
	std::vector<double> m0(cols, 0), m1(cols, 0), m3(cols, 0);
	m0[0]  = 0.5;
	m1[0]  = 0.15;
	m3[0]  = -0.0015;
	mask_0 = cc->MakeCKKSPackedPlaintext(m0, 1, 0, nullptr, cols);
	mask_1 = cc->MakeCKKSPackedPlaintext(m1, 1, 0, nullptr, cols);
	mask_3 = cc->MakeCKKSPackedPlaintext(m3, 1, 0, nullptr, cols);
}

static void activation(fideslib::Ciphertext<fideslib::DCRTPoly>& ct) {
	auto ct3 = cc->EvalSquare(ct);
	auto aux = cc->EvalMult(ct, mask_3);
	ct3		 = cc->EvalMult(ct3, aux);
	cc->EvalMultInPlace(ct, mask_1);
	cc->EvalAddInPlace(ct, ct3);
	cc->EvalAddInPlace(ct, mask_0);
}

// Cascading row accumulation - each rotation uses the updated ct
static void row_accumulate(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, size_t cols) {
	for (size_t j = 1; j < cols; j <<= 1)
		cc->EvalAddInPlace(ct, cc->EvalRotate(ct, j));
}

// Cascading row propagation (negative direction)
static void row_propagate(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, size_t cols) {
	for (size_t j = 1; j < cols; j <<= 1)
		cc->EvalAddInPlace(ct, cc->EvalRotate(ct, -static_cast<int>(j)));
}

// Cascading column accumulation
static void column_accumulate(fideslib::Ciphertext<fideslib::DCRTPoly>& ct, size_t rows, size_t cols) {
	for (size_t j = cols; j < rows * cols; j <<= 1)
		cc->EvalAddInPlace(ct, cc->EvalRotate(ct, j));
}

static void train_iteration(fideslib::Ciphertext<fideslib::DCRTPoly>& data,
  const fideslib::Ciphertext<fideslib::DCRTPoly>& results,
  fideslib::Ciphertext<fideslib::DCRTPoly>& weights,
  size_t rows,
  size_t cols,
  size_t batch_size,
  double lr) {
	static bool do_boot = false;

	auto ct = cc->EvalMult(data, weights);

	row_accumulate(ct, cols);
	activation(ct);

	cc->EvalSubInPlace(ct, results);
	row_propagate(ct, cols);
	auto scale = (lr / batch_size);
	if (do_boot && (!sparse_encaps || !slow)) {
		scale *= cc->GetPreScaleFactor(cols);
	}
	cc->EvalMultInPlace(data, scale);

	ct = cc->EvalMult(ct, data);

	column_accumulate(ct, rows, cols);

	if (do_boot && (!sparse_encaps || !slow)) {
		cc->EvalMultInPlace(weights, cc->GetPreScaleFactor(cols));
	}
	cc->EvalSubInPlace(weights, ct);

	if (do_boot) {
		weights->SetSlots(cols);
		cc->EvalBootstrapInPlace(weights, 1, 0, prescale);
		weights->SetSlots(numSlots);
	}
	do_boot = !do_boot;
}

static void inference_iteration(fideslib::Ciphertext<fideslib::DCRTPoly>& data, const fideslib::Ciphertext<fideslib::DCRTPoly>& weights, size_t cols) {
	auto ct = cc->EvalMult(data, weights);
	row_accumulate(ct, cols);
	activation(ct);
	data = ct;
}

std::vector<iteration_time_t> logistic_regression_train(const std::vector<std::vector<double>>& data,
  const std::vector<std::vector<double>>& results,
  fideslib::Ciphertext<fideslib::DCRTPoly>& weights,
  size_t rows,
  size_t cols,
  size_t last_rows,
  size_t iterations,
  const fideslib::PublicKey<fideslib::DCRTPoly>& pk) {

	std::vector<fideslib::Ciphertext<fideslib::DCRTPoly>> enc_data(data.size());
	std::vector<fideslib::Ciphertext<fideslib::DCRTPoly>> enc_results(results.size());

	for (size_t i = 0; i < iterations; ++i) {
		enc_data[i]	 = encrypt_data(data[i], pk, 2, depth - 9);
		enc_results[i] = encrypt_data(results[i], pk, 2, depth - 7);
	}

	cc->Synchronize();
	auto start_total = std::chrono::high_resolution_clock::now();
	for (size_t it = 0; it < iterations; ++it) {
		size_t idx	 = it % data.size();
		size_t batch = (idx == data.size() - 1) ? last_rows : rows;
		double lr	 = std::max(10.0 / (it + 1), 0.005);

		train_iteration(enc_data[idx], enc_results[idx], weights, rows, cols, batch, lr);
	}
	cc->Synchronize();
	auto end_total = std::chrono::high_resolution_clock::now();

	return { { std::chrono::duration_cast<time_unit_t>(end_total - start_total), time_unit_t::zero()} };
}

std::vector<iteration_time_t> logistic_regression_inference(std::vector<std::vector<double>>& data,
  const fideslib::Ciphertext<fideslib::DCRTPoly>& weights,
  size_t cols,
  const fideslib::KeyPair<fideslib::DCRTPoly>& k) {

	std::vector<fideslib::Ciphertext<fideslib::DCRTPoly>> enc_data(data.size());
	for (size_t i = 0; i < data.size(); ++i)
		enc_data[i] = encrypt_data(data[i], k.publicKey, 1, depth - 4);


	cc->Synchronize();
	auto start_total = std::chrono::high_resolution_clock::now();
	for (size_t it = 0; it < data.size(); ++it) {
		inference_iteration(enc_data[it], weights, cols);
	}
	cc->Synchronize();
	auto end_total = std::chrono::high_resolution_clock::now();

	for (size_t i = 0; i < data.size(); ++i) {
		data[i] = decrypt_data(enc_data[i], k.secretKey, numSlots);
	}
	return { {std::chrono::duration_cast<time_unit_t>(end_total - start_total), time_unit_t::zero()} };
}

std::vector<iteration_time_t>
fideslib_training(const std::vector<std::vector<double>>& data, const std::vector<double>& results, std::vector<double>& weights, size_t iterations) {

	std::vector<std::vector<double>> data_fhe, results_fhe;
	std::vector<double> weights_fhe;

	auto [cols, rows, last] = pack_data_fhe(data, results, weights, data_fhe, results_fhe, weights_fhe, numSlots);

	uint32_t depth = create_context(false);
	keys		   = cc->KeyGen();
	sk			   = keys.secretKey;
	prepare_context(keys, cols, rows);

	weights_fhe.resize(cols);
	auto enc_weights = encrypt_data(weights_fhe, keys.publicKey, 2, depth - 9);

	auto times = logistic_regression_train(data_fhe, results_fhe, enc_weights, rows, cols, last, iterations, keys.publicKey);

	auto dec = decrypt_data(enc_weights, keys.secretKey, numSlots);
	unpack_weights(dec, weights, weights.size());

	return times;
}

std::pair<std::vector<iteration_time_t>, double>
fideslib_inference(const std::vector<std::vector<double>>& data, const std::vector<double>& results, const std::vector<double>& weights) {

	std::vector<std::vector<double>> data_fhe, results_fhe;
	std::vector<double> weights_fhe;

	auto [cols, rows, last] = pack_data_fhe(data, results, weights, data_fhe, results_fhe, weights_fhe, numSlots);

	create_context(true);
	keys = cc->KeyGen();
	prepare_context(keys, cols, rows);

	auto enc_weights = encrypt_data(weights_fhe, keys.publicKey, 1, depth - 4);
	auto times		 = logistic_regression_inference(data_fhe, enc_weights, cols, keys);

	std::vector<std::vector<double>> unpacked;
	unpack_data(data_fhe, unpacked, rows, cols, last, weights.size());

	// Calculate accuracy (disabled for timing)
	size_t correct = 0;

	for (size_t i = 0; i < unpacked.size(); ++i) {
		bool pred	= unpacked[i][0] >= 0.5;
		bool actual = results[i] == 1.0;
		if (pred == actual)
			++correct;
	}

	double acc = 100.0 * correct / unpacked.size();

	return { times, acc };
}
