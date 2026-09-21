#include "data.hpp"
#include "fhe.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

std::vector<int> devices		  = {};
bool prescale					  = true;
bool sparse_encaps				  = true;
bool slow						  = false;
std::vector<uint32_t> bStep		  = { 16, 16 };
std::vector<uint32_t> levelBudget = { 2, 2 };
uint32_t ringDim				  = 1 << 16;
uint32_t numSlots				  = ringDim / 2;

static void print_usage(const char* name) {
	std::cerr << "Usage:\n"
			  << "  " << name << " train <dataset> <iterations> <devices> <sparse> <slow>\n"
			  << "  " << name << " inference <dataset> <devices> <sparse> <slow>\n"
			  << "  " << name << " perf <dataset> <iterations> <devices> <sparse> <slow>\n"
			  << "\nDatasets: random, mnist\n"
			  << "Sparse: 0 = UNIFORM_TERNARY, 1 = SPARSE_TERNARY\n"
			  << "Slow: 0 = fast mode, 1 = slow mode\n";
	exit(EXIT_FAILURE);
}

static dataset_t parse_dataset(const std::string& s) {
	if (s == "random")
		return RANDOM;
	if (s == "mnist")
		return MNIST;
	std::cerr << "Unknown dataset: " << s << std::endl;
	exit(EXIT_FAILURE);
}

static void setup_devices(int count) {
	devices.clear();
	for (int i = 0; i < count; ++i)
		devices.push_back(i);
}

int main(int argc, char* argv[]) {
	if (argc < 4)
		print_usage(argv[0]);

	std::string mode  = argv[1];
	dataset_t dataset = parse_dataset(argv[2]);

	if (mode == "train") {
		if (argc != 7)
			print_usage(argv[0]);
		size_t iterations = std::stoul(argv[3]);
		setup_devices(std::stoi(argv[4]));
		sparse_encaps = std::stoi(argv[5]) != 0;
		slow = std::stoi(argv[6]) != 0;

		std::vector<std::vector<double>> data;
		std::vector<double> results, weights;
		size_t features = prepare_data_csv(dataset, TRAIN, data, results);
		generate_weights(features, weights);

		auto times = fideslib_training(data, results, weights, iterations);
		print_times(times, "TRAIN", !devices.empty(), data.size());

	} else if (mode == "inference") {
		if (argc != 6)
			print_usage(argv[0]);
		setup_devices(std::stoi(argv[3]));
		sparse_encaps = std::stoi(argv[4]) != 0;
		slow = std::stoi(argv[5]) != 0;

		std::vector<std::vector<double>> data;
		std::vector<double> results, weights;
		size_t features = prepare_data_csv(dataset, VALIDATION, data, results);
		load_weights("../weights/weights.csv", features, weights);

		auto [times, accuracy] = fideslib_inference(data, results, weights);
		print_times(times, "INFERENCE", !devices.empty(), data.size());
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	} else if (mode == "perf") {
		if (argc != 7)
			print_usage(argv[0]);
		size_t iterations = std::stoul(argv[3]);
		setup_devices(std::stoi(argv[4]));
		sparse_encaps = std::stoi(argv[5]) != 0;
		slow = std::stoi(argv[6]) != 0;

		std::vector<std::vector<double>> train_data, val_data;
		std::vector<double> train_results, val_results;
		prepare_data_csv(dataset, TRAIN, train_data, train_results);
		size_t features = prepare_data_csv(dataset, VALIDATION, val_data, val_results);

		std::vector<double> weights;
		generate_weights(features, weights);

		auto train_times = fideslib_training(train_data, train_results, weights, iterations);
		print_times(train_times, "TRAIN", !devices.empty(), train_data.size());

		auto [val_times, accuracy] = fideslib_inference(val_data, val_results, weights);
		print_times(val_times, "INFERENCE", !devices.empty(), val_data.size());
		std::cout << "Accuracy: " << accuracy << "%" << std::endl;

	} else {
		print_usage(argv[0]);
	}

	return EXIT_SUCCESS;
}