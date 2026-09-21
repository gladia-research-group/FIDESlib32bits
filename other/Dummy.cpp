#include <omp.h>

#include <iostream>

int main() {
    #pragma omp parallel for
    for (int i = 0; i < 8; i++) {
        int thread_id = omp_get_thread_num();
        std::cout << "Hello from thread " << thread_id << std::endl;
    }
    return 0;
}