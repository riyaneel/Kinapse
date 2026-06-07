#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <SnnTrainer.hpp>

using namespace engine;

__attribute__((noinline)) static void print_usage(const char *name) noexcept {
	std::cerr << "Usage: " << name
			  << " --dataset <path_to.bin> --out <path_to_weights.bin> [--epochs num] [--profile "
				 "<scalper|momentum|defensive>]\n"
			  << "Options:\n"
			  << "  --dataset   Path to the compiled historical binary dataset.\n"
			  << "  --out       Path where the trained weights will be saved.\n"
			  << "  --epochs    Number of passes over the dataset (default: 1).\n"
			  << "  --profile   Market Maker profile (default: scalper).\n";
}

int main(const int argc, char **argv) {
	std::string					  dataset_path;
	std::string					  output_path;
	snn_train::MarketMakerProfile profile = snn_train::MarketMakerProfile::SCALPER;
	int							  epochs  = 1;

	for (int i = 1; i < argc; ++i) {
		if (std::strcmp(argv[i], "--dataset") == 0 && i + 1 < argc) {
			dataset_path = argv[++i];
		} else if (std::strcmp(argv[i], "--out") == 0 && i + 1 < argc) {
			output_path = argv[++i];
		} else if (std::strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
			epochs = std::atoi(argv[++i]);
			if (epochs < 1) {
				std::cerr << "[ERROR] epochs must be >= 1\n";
				return EXIT_FAILURE;
			}
		} else if (std::strcmp(argv[i], "--profile") == 0 && i + 1 < argc) {
			if (std::string p = argv[++i]; p == "momentum")
				profile = snn_train::MarketMakerProfile::MOMENTUM;
			else if (p == "defensive")
				profile = snn_train::MarketMakerProfile::DEFENSIVE;
			else if (p == "scalper")
				profile = snn_train::MarketMakerProfile::SCALPER;
			else {
				std::cerr << "[ERROR] Unknown profile: " << p << "\n";
				return EXIT_FAILURE;
			}
		}
	}

	if (dataset_path.empty() || output_path.empty()) {
		print_usage(argv[0]);
		return EXIT_FAILURE;
	}

	int fd = open(dataset_path.c_str(), O_RDONLY);
	if (fd < 0) {
		std::cerr << "[ERROR] Failed to open dataset file: " << dataset_path << "\n";
		return EXIT_FAILURE;
	}

	struct stat sb{};
	if (fstat(fd, &sb) < 0) {
		std::cerr << "[ERROR] Failed to read file stats.\n";
		close(fd);
		return EXIT_FAILURE;
	}

	const auto st_size = static_cast<size_t>(sb.st_size);
	if (st_size % sizeof(matching::WireOrder) != 0) {
		std::cerr << "[ERROR] Dataset size is not a multiple of WireOrder (64 bytes). Corrupted file?\n";
		close(fd);
		return EXIT_FAILURE;
	}

	const size_t num_orders	 = st_size / sizeof(matching::WireOrder);
	void		*mapped_data = mmap(nullptr, st_size, PROT_READ, MAP_PRIVATE, fd, 0);
	if (mapped_data == MAP_FAILED) {
		std::cerr << "[ERROR] Memory mapping failed.\n";
		close(fd);
		return EXIT_FAILURE;
	}

	if (madvise(mapped_data, st_size, MADV_SEQUENTIAL | MADV_WILLNEED) == -1) {
		std::cerr << "[WARN] madvise failed, continuing without prefetch hint.\n";
	}

	const auto *wire_orders = static_cast<const matching::WireOrder *>(mapped_data);
	std::cout << "[INIT] Initializing OrderPool...\n";
	snn_train::SnnTrainerConfig config;
	config.profile = profile;
	snn_train::SnnTrainer trainer(config);

	auto start_time = std::chrono::high_resolution_clock::now();

	for (int epoch = 0; epoch < epochs; ++epoch) {
		if (epoch > 0) {
			std::cout << "[TRAIN] Epoch " << epoch + 1 << " / " << epochs << ", resetting episode state.\n";
			trainer.reset_episode();
		}

		trainer.train(wire_orders, num_orders);
	}

	auto end_time	 = std::chrono::high_resolution_clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

	std::cout << "[STATS] Training completed in " << duration_ms << " ms.\n";
	std::cout << "[STATS] Throughput: "
			  << (num_orders / (duration_ms > 0 ? static_cast<size_t>(duration_ms) : 1)) * 1000 << " orders/sec.\n";

	trainer.export_weights(output_path);

	munmap(mapped_data, st_size);
	close(fd);

	return 0;
}
