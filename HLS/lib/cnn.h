#ifndef CNN_H_
#define CNN_H_

#include <stdexcept>
#include <string>

const int kNum = 256;
const int kKernel = 5;
const int kImSize = 224;
const int kInImSize = 228;
const int kOutImSize = 112;

// Utility function declarations.
void LoadData(
    const std::string& data_dir,
    float input[kNum][kInImSize][kInImSize],
    float weight[kNum][kNum][kKernel][kKernel],
    float bias[kNum]
);
int Verify(
    const std::string& data_dir,
    const float output[kNum][kOutImSize][kOutImSize]
);
void CnnKernel(
    const float input[kNum][kInImSize][kInImSize],
    const float weight[kNum][kNum][kKernel][kKernel],
    const float bias[kNum],
    float output[kNum][kOutImSize][kOutImSize]
);
void CnnSequential(
    const float input[kNum][kInImSize][kInImSize],
    const float weight[kNum][kNum][kKernel][kKernel],
    const float bias[kNum],
    float output[kNum][kOutImSize][kOutImSize]
);
#endif
