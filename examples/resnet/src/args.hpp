#ifndef ARGS_HPP
#define ARGS_HPP

#include <iostream>
#include <string>
#include <sys/stat.h>

typedef struct args {
	bool generate_context{};
	int context_index{};
	std::string input_filename{};
	bool verbose{};
	std::string program_name{};

} args;

inline args check_arguments(int argc, char* argv[]) {

	args program_args;

	program_args.program_name	  = std::string(argv[0]);
	program_args.generate_context = false;
	program_args.verbose		  = false;

	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "verbose") {
			if (i + 1 < argc) {
				program_args.verbose = true;
			}
		}
	}

	for (int i = 1; i < argc; ++i) {
		if (std::string(argv[i]) == "load_keys") {
			if (i + 1 < argc) {
				program_args.context_index	   = atoi(argv[i + 1]);
				program_args.generate_context = false;
			}
		}

		if (std::string(argv[i]) == "generate_keys") {
			program_args.generate_context = true;
			if (i + 1 < argc) {
				std::string folder;
				
				// Extract next argument as integer.
				int index = atoi(argv[i + 1]);
				program_args.context_index	   = index;
			}
		}
		if (std::string(argv[i]) == "input") {
			if (i + 1 < argc) {
				program_args.input_filename = "../" + std::string(argv[i + 1]);
				if (program_args.verbose)
					std::cout << "Input image set to: \"" << program_args.input_filename << "\"." << std::endl;
			}
		}
	}

	if (program_args.context_index <= 0) {
		std::cerr << "You either have to use the argument \"generate_keys\" or "
					 "\"load_keys\"!\nIf it is your first time, you could try "
					 "with \""
				  << program_args.program_name << " generate_keys 1\"\nCheck the README.md.\nAborting. :-(" << std::endl;
		exit(1);
	}

	if (program_args.context_index > 12) {
		std::cerr << "Context index not valid. Please, set a a number from '1' to "
					 "'8'. Check the README.md"
				  << std::endl;
		exit(1);
	}

	return program_args;
}

#endif // ARGS_HPP