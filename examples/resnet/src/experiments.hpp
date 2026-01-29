#ifndef EXPERIMENTS_HPP
#define EXPERIMENTS_HPP

#include <cstdint>
#include <iostream>

#include <fideslib/fideslib.hpp>

typedef struct experiment_settings {
	uint32_t log_ring_dim{};
	uint32_t log_scale_factor{};
	uint32_t log_primes{};
	uint32_t digits_hks{};
	uint32_t cts_levels{};
	uint32_t stc_levels{};
	uint32_t relu_degree{};
	bool serialize = false;
	std::string parameters_folder{};
	bool verbose							= false;
	uint32_t depth							= 0;
	bool prescale							= false;
	bool autoload							= false;
	bool by_layer_loading					= false;
	fideslib::SecretKeyDist secret_key_dist = fideslib::SPARSE_TERNARY;

} experiment_settings;

const experiment_settings experiment1 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp1",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = false,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment2 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp2",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = false,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

const experiment_settings experiment3 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp3",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = false,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment4 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp4",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = false,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

const experiment_settings experiment5 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp5",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = true,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment6 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp6",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = true,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

const experiment_settings experiment7 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp7",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = true,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment8 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp8",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = true,
	.by_layer_loading  = false,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

const experiment_settings experiment9 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp9",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = true,
	.by_layer_loading  = true,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment10 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp10",
	.depth			   = 23,
	.prescale		   = false,
	.autoload		   = true,
	.by_layer_loading  = true,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

const experiment_settings experiment11 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp11",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = true,
	.by_layer_loading  = true,
	.secret_key_dist   = fideslib::SPARSE_TERNARY,
};

const experiment_settings experiment12 = {
	.log_ring_dim	   = 16,
	.log_scale_factor  = 56,
	.log_primes		   = 52,
	.digits_hks		   = 3,
	.cts_levels		   = 3,
	.stc_levels		   = 3,
	.relu_degree	   = 27,
	.serialize		   = true,
	.parameters_folder = "keys_exp12",
	.depth			   = 23,
	.prescale		   = true,
	.autoload		   = true,
	.by_layer_loading  = true,
	.secret_key_dist   = fideslib::SPARSE_ENCAPSULATED,
};

inline experiment_settings get_experiment_settings(int experiment_number) {
	switch (experiment_number) {
	case 1: return experiment1;
	case 2: return experiment2;
	case 3: return experiment3;
	case 4: return experiment4;
	case 5: return experiment5;
	case 6: return experiment6;
	case 7: return experiment7;
	case 8: return experiment8;
	case 9: return experiment9;
	case 10: return experiment10;
	case 11: return experiment11;
	case 12: return experiment12;
	default: std::cout << "Experiment number not valid. Using experiment 1." << std::endl; return experiment1;
	}
}

#endif // EXPERIMENTS_HPP