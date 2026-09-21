#ifndef UTILS_HPP
#define UTILS_HPP

#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <fideslib/fideslib.hpp>

#include "stb_image.h"

static inline std::vector<double> read_values_from_file(const std::string& filename, double scale = 1) {
	std::vector<double> values;
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Can not open " << filename << std::endl;
		return values;
	}

	std::string row;
	while (std::getline(file, row)) {
		std::istringstream stream(row);
		std::string value;
		while (std::getline(stream, value, ',')) {
			try {
				double num = std::stod(value);
				values.push_back(num * scale);
			} catch (const std::invalid_argument& e) {
				std::cerr << "Can not convert: " << value << std::endl;
				values.push_back(0.0);
			}
		}
	}

	file.close();

	return values;
}

static inline std::string get_class(size_t max_index) {
	switch (max_index) {
		//I know, I should use a dict
		case 0:
			return "Airplane";
		case 1:
			return "Automobile";
		case 2:
			return "Bird";
		case 3:
			return "Cat";
		case 4:
			return "Deer";
		case 5:
			return "Dog";
		case 6:
			return "Frog";
		case 7:
			return "Horse";
		case 8:
			return "Ship";
		case 9:
			return "Truck";
		default:
			return "?";
	}
}

static inline std::vector<double> read_fc_weight(const std::string &filename) {
		std::vector<double> weight = read_values_from_file(filename);
		std::vector<double> weight_corrected;

		for (size_t i = 0; i < 64; i++) {
			for (size_t j = 0; j < 10; j++) {
				weight_corrected.push_back(weight[(10 * i) + j]);
			}
			for (size_t j = 0; j < 64 - 10; j++) {
				weight_corrected.push_back(0);
			}
		}

		return weight_corrected;
	}


static inline void write_to_file(const std::string& filename, const std::string& content) {
	std::ofstream file;
	file.open(filename);
	file << content.c_str();
	file.close();
}

static inline std::string read_from_file(const std::string& filename) {
	std::string line;
	std::ifstream myfile(filename);
	if (myfile.is_open()) {
		if (getline(myfile, line)) {
			myfile.close();
			return line;
		} else {
			std::cerr << "Could not open " << filename << "." << std::endl;
			exit(1);
		}
	} else {
		std::cerr << "Could not open " << filename << "." << std::endl;
		exit(1);
	}
}

inline std::vector<double> read_image(const char* filename) {
	int width				  = 32;
	int height				  = 32;
	int channels			  = 3;
	unsigned char* image_data = stbi_load(filename, &width, &height, &channels, 0);

	if (!image_data) {
		std::cerr << "Could not load the image in " << filename << std::endl;
		return {};
	}

	std::vector<double> imageVector;
	imageVector.reserve(static_cast<size_t>(width) * static_cast<size_t>(height) * static_cast<size_t>(channels));

	for (int i = 0; i < width * height; ++i) {
		// Channel R
		imageVector.push_back(static_cast<double>(image_data[3 * i]) / 255.0f);
	}
	for (int i = 0; i < width * height; ++i) {
		// Channel G
		imageVector.push_back(static_cast<double>(image_data[1 + 3 * i]) / 255.0f);
	}
	for (int i = 0; i < width * height; ++i) {
		// Channel B
		imageVector.push_back(static_cast<double>(image_data[2 + 3 * i]) / 255.0f);
	}

	stbi_image_free(image_data);

	return imageVector;
}

static inline double compute_approx_error(fideslib::Plaintext& expected, fideslib::Plaintext& bootstrapped) {
	std::vector<std::complex<double>> result;
	std::vector<std::complex<double>> expectedResult;

	result		   = bootstrapped->GetCKKSPackedValue();
	expectedResult = expected->GetCKKSPackedValue();

	if (result.size() != expectedResult.size()) {
		std::cerr << "Warning: cannot compute approximation error, different sizes." << std::endl;
	}

	double maxError = 0;
	for (size_t i = 0; i < result.size(); ++i) {
		double error = std::abs(result[i].real() - expectedResult[i].real());
		if (maxError < error)
			maxError = error;
	}

	return std::abs(std::log2(maxError));
}

#endif // UTILS_HPP