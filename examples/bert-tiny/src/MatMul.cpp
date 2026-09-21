// @author TPOC: contact@palisade-crypto.org
//
// @copyright Copyright (c) 2021, Duality Technologies Inc.
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. THIS SOFTWARE IS
// PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
// IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
// MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
// EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THISvector<
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#define PROFILE

#include "MatMul.h"

namespace FIDESlib::CKKS {

std::vector<std::vector<double>> generateRandomMatrix(size_t numRows, size_t numCols, unsigned int seed) {
    std::vector<std::vector<double>> matrix;
    std::random_device rd;
    std::mt19937 gen;

    if (seed != 0) {
        gen.seed(seed);
    } else {
        gen.seed(rd());
    }

    std::uniform_real_distribution<> dis(-1.0, 1.0);
    for (size_t i = 0; i < numRows; i++) {
        std::vector<double> x;
        for (size_t k = 0; k < numCols; k++) {
            x.emplace_back(dis(gen));
        }
        matrix.emplace_back(std::move(x));
        x.clear();
    }

    return matrix;
}

std::vector<int32_t> GenerateMatMulRotationIndices(uint32_t rowSize) {
    std::set<int32_t> indices;
    for (size_t i = 1; i < rowSize; i++) {
        indices.insert(i);
        indices.insert(-i);
        indices.insert(i * rowSize);
        indices.insert(i - rowSize);
    }

    std::vector<int32_t> indicesList(indices.begin(), indices.end());
    return indicesList;
}

// // Helper methods to get permutation matrices for matrix multiplication
// template <typename Element>
// std::vector<std::vector<Element>> getDiagonals(std::vector<std::vector<Element>> matrix) {
//     size_t diagonalLength = matrix.size();
//     if (diagonalLength == 0) {
//         return std::vector<std::vector<Element>>();
//     }
//     size_t numDiagonals = matrix[0].size();

//     std::vector<std::vector<Element>> diagonals;
//     for (size_t j = 0; j < numDiagonals; j++) {
//         std::vector<Element> diagonal;
//         for (size_t i = 0; i < diagonalLength; i++) {
//             diagonal.emplace_back(matrix[i][(i + j) % numDiagonals]);
//         }
//         diagonals.emplace_back(diagonal);
//     }
//     return diagonals;
// }

// std::vector<std::vector<double>> getSigmaPermutationMatrix(size_t rowSize, size_t colSize) {
//     std::vector<std::vector<double>> sigma(rowSize * colSize, std::vector<double>(rowSize * colSize, 0));

//     for (size_t i = 0; i < rowSize; i++) {
//         for (size_t j = 0; j < colSize; j++) {
//             size_t rowIndex = colSize * i + j;                     
//             size_t colIndex = colSize * i + ((i + j) % colSize); 
//             sigma[rowIndex][colIndex] = 1;
//         }
//     }
//     return sigma;
// }

// std::vector<std::vector<double>> getTauPermutationMatrix(size_t rowSize, size_t colSize) {
//     std::vector<std::vector<double>> tau(rowSize * colSize, std::vector<double>(rowSize * colSize, 0));

//     for (size_t i = 0; i < rowSize; i++) {
//         for (size_t j = 0; j < colSize; j++) {
//             size_t rowIndex = colSize * i + j;                          
//             size_t colIndex = colSize * ((i + j) % rowSize) + j;        
//             tau[rowIndex][colIndex] = 1;
//         }
//     }
//     return tau;
// }


// std::vector<std::vector<double>> getPhiDiagonals(size_t rowSize, size_t colSize, size_t numRotations) {
//     std::vector<std::vector<double>> phiDiagonals(2, std::vector<double>(rowSize * colSize, 0));
    
//     // Layout is row-major: linear index = r * rowSize + c
//     for (size_t r = 0; r < colSize; ++r) {
//         for (size_t c = 0; c < rowSize; ++c) {
//             const size_t idx = r * rowSize + c;
//             if (c < rowSize - numRotations)
//                 phiDiagonals[0][idx] = 1.0;
//         }
//     }

//     for (size_t r = 0; r < colSize; ++r) {
//         for (size_t c = 0; c < rowSize; ++c) {
//             const size_t idx = r * rowSize + c;
//             if (c >= rowSize - numRotations)
//                 phiDiagonals[1][idx] = 1.0;
//         }
//     }
//     return phiDiagonals;
// }

// std::vector<std::vector<double>> hconcat(
//     const std::vector<std::vector<double>>& A,
//     const std::vector<std::vector<double>>& B) {

//     if (A.size() != B.size())
//         throw std::invalid_argument("row counts differ");

//     const size_t R = A.size();
//     std::vector<std::vector<double>> C(R);
//     for (size_t i = 0; i < R; ++i) {
//         if (A[i].size() != A[0].size() || B[i].size() != B[0].size())
//             throw std::invalid_argument("ragged rows not allowed");

//         C[i].reserve(A[i].size() + B[i].size());
//         C[i].insert(C[i].end(), A[i].begin(), A[i].end());
//         C[i].insert(C[i].end(), B[i].begin(), B[i].end());
//     }
//     return C;
// }

// struct MatrixMatrixProductPrecomputations getMatrixMatrixProductPrecomputations(
//     lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize, int colSize) {

//     bool print = false;
//     if (print) std::cout << "rowsize: " << rowSize << ", colsize: " << colSize << std::endl;
//     if (colSize == 0) { colSize = rowSize; }

//     // std::vector<std::vector<double>> sigmaDiagonals = getDiagonals(getSigmaPermutationMatrix(colSize, rowSize));
//     // std::vector<std::vector<double>> tauDiagonals = getDiagonals(getTauPermutationMatrix(colSize, rowSize));
//     std::vector<std::vector<double>> sigmaDiagonals = getDiagonals(getSigmaPermutationMatrix(rowSize, colSize));
//     std::vector<std::vector<double>> tauDiagonals = getDiagonals(getTauPermutationMatrix(rowSize, colSize));

//     std::vector<lbcrypto::Plaintext> sigmaPlaintexts(sigmaDiagonals.size());
//     std::vector<lbcrypto::Plaintext> tauPlaintexts(tauDiagonals.size());
//     std::vector<std::vector<lbcrypto::Plaintext>> phiPlaintexts;
//     std::vector<std::vector<lbcrypto::Plaintext>> phiPlaintexts_new;
//     std::vector<std::vector<lbcrypto::Plaintext>> uPlaintexts;

//     // assert(sigmaDiagonals.size() == rowSize * colSize);
//     // assert(tauDiagonals.size() == rowSize * colSize);
//     if (print) std::cout << "0: " << std::endl << std::flush;

//     for (int i = 0; i < rowSize; i++) {
//         // std::cout << "idx: " << i << std::endl << std::flush;
//         lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(sigmaDiagonals[i]);
//         sigmaPlaintexts[i] = ptxtSigma;
//     }
//     for (int k = 0; k < rowSize - 1; ++k) {
//         int idx = rowSize * colSize - 1 - k;
//         // int idx = rowSize * colSize - 1 - k;
//         // std::cout << "idx: " << idx << std::endl << std::flush;
//         sigmaPlaintexts[idx] = context->MakeCKKSPackedPlaintext(sigmaDiagonals[idx]);
//     }

//     // for (int i = 0; i < rowSize; i++) {
//     //     lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(sigmaDiagonals[i]);
//     //     sigmaPlaintexts[i] = ptxtSigma;
//     // }
//     // for (int i = rowSize * rowSize - rowSize; i < rowSize * rowSize; i++) {
//     //     lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(sigmaDiagonals[i]);
//     //     sigmaPlaintexts[i] = ptxtSigma;
//     // }
//     if (print) {
//         std::cout << "sigmaDiagonals: " << std::endl << std::setprecision(0);
//         for (size_t j = 0; j < sigmaDiagonals.size(); ++j) {
//             for (size_t i = 0; i < sigmaDiagonals[0].size(); ++i) {
//                 std::cout << sigmaDiagonals[j][i] << ", ";
//             }
//             std::cout << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     if (print) std::cout << "2: " << std::endl << std::flush;
//     for (int i = 0; i < rowSize * colSize; i += rowSize) {
//         lbcrypto::Plaintext ptxtTau = context->MakeCKKSPackedPlaintext(tauDiagonals[i]);
//         tauPlaintexts[i] = ptxtTau;
//     }
//     if (print) std::cout << "3: " << std::endl << std::flush;
//     for (int i = 0; i < colSize; i++) {
//         std::vector<std::vector<double>> phi = getPhiDiagonals(rowSize, rowSize, (i+1) % rowSize);
//         // std::vector<std::vector<double>> phi = getPhiDiagonals(rowSize, rowSize, (i + i / rowSize) % rowSize);
//         for (int j = 1; j < (colSize / rowSize); j*=2) {
//             phi = hconcat(phi, phi);
//         }

//         if (print) {
//             std::cout << "phiDiagonals, i = " << i << std::endl << std::setprecision(0);
//             for (size_t j = 0; j < phi.size(); ++j) {
//                 for (size_t k = 0; k < phi[0].size(); ++k) {
//                     std::cout << phi[j][k] << ", ";
//                 }
//                 std::cout << std::endl;
//             }
//             std::cout << std::endl;
//         }

//         lbcrypto::Plaintext ptxtPhi1 = context->MakeCKKSPackedPlaintext(phi[0]);
//         lbcrypto::Plaintext ptxtPhi2 = context->MakeCKKSPackedPlaintext(phi[1]);
//         std::vector<lbcrypto::Plaintext> phiVec = {ptxtPhi1, ptxtPhi2};
//         phiPlaintexts.emplace_back(phiVec);

//     }

//     // for (int i = 0; i < rowSize; i++) {
//     //     std::vector<std::vector<double>> phi = getPhiDiagonals(rowSize, rowSize, i);
//     //     lbcrypto::Plaintext ptxtPhi1 = context->MakeCKKSPackedPlaintext(phi[0]);
//     //     lbcrypto::Plaintext ptxtPhi2 = context->MakeCKKSPackedPlaintext(phi[1]);
//     //     std::vector<lbcrypto::Plaintext> phiVec = {ptxtPhi1, ptxtPhi2};
//     //     phiPlaintexts.emplace_back(phiVec);
//     // }

//     if (print)  std::cout << "4: " << std::endl << std::flush;
//     for (int i = 0; i < colSize; ++i) {
//         std::vector<std::vector<double>> phi = getPhiDiagonals(colSize, rowSize, i);
//         // std::vector<std::vector<double>> phi = getPhiDiagonals(rowSize, colSize, i);
//         if (phi.empty()) continue;
//         // for (int j = 1; j < (colSize / rowSize); j*=2) {
//         //     phi = hconcat(phi, phi);
//         // }
//         const int numSlots = static_cast<int>(phi[0].size());
//         std::vector<double> row0mask(numSlots, 0.0);
//         for (int j = 0; j < std::min(colSize, numSlots); ++j) row0mask[j] = 1.0;

//         for (auto& v : phi) {
//             if (static_cast<int>(v.size()) != numSlots) continue;
//             for (int j = 0; j < numSlots; ++j) v[j] *= row0mask[j];
//         }

//         // Pack
//         lbcrypto::Plaintext ptxtPhi1 = context->MakeCKKSPackedPlaintext(phi[0]);
//         lbcrypto::Plaintext ptxtPhi2 = context->MakeCKKSPackedPlaintext(phi[1]);
//         phiPlaintexts_new.emplace_back(std::vector<lbcrypto::Plaintext>{ptxtPhi1, ptxtPhi2});
//     }
//     if (print) std::cout << "5: " << std::endl << std::flush;

//     struct MatrixMatrixProductPrecomputations precomp;
//     precomp.rowSize = rowSize;
//     precomp.colSize = colSize;
//     precomp.sigmaPlaintexts = sigmaPlaintexts;
//     precomp.tauPlaintexts = tauPlaintexts;
//     precomp.phiPlaintexts = phiPlaintexts;
//     precomp.phiPlaintexts_new = phiPlaintexts_new;

//     return precomp;
// }

// Helper methods to get permutation matrices for matrix multiplication
template <typename Element>
std::vector<std::vector<Element>> getDiagonals(std::vector<std::vector<Element>> matrix) {
    size_t diagonalLength = matrix.size();
    if (diagonalLength == 0) {
        return std::vector<std::vector<Element>>();
    }
    size_t numDiagonals = matrix[0].size();

    std::vector<std::vector<Element>> diagonals;
    for (size_t j = 0; j < numDiagonals; j++) {
        std::vector<Element> diagonal;
        for (size_t i = 0; i < diagonalLength; i++) {
            diagonal.emplace_back(matrix[i][(i + j) % numDiagonals]);
        }
        diagonals.emplace_back(diagonal);
    }
    return diagonals;
}

std::vector<std::vector<double>> getSigmaPermutationMatrix(size_t rowSize) {

    std::vector<std::vector<double>> sigma(rowSize * rowSize, std::vector<double>(rowSize * rowSize, 0));

    for (size_t i = 0; i < rowSize; i++) {
        for (size_t j = 0; j < rowSize; j++) {
            size_t rowIndex = rowSize * i + j;
            size_t colIndex = rowSize * i + ((i + j) % rowSize);
            sigma[rowIndex][colIndex] = 1;
        }
    }
    return sigma;
}

std::vector<std::vector<double>> getTauPermutationMatrix(size_t rowSize) {
    std::vector<std::vector<double>> tau(rowSize * rowSize, std::vector<double>(rowSize * rowSize, 0));
    ;
    for (size_t i = 0; i < rowSize; i++) {
        for (size_t j = 0; j < rowSize; j++) {
            size_t rowIndex = rowSize * i + j;
            size_t colIndex = rowSize * ((i + j) % rowSize) + j;
            tau[rowIndex][colIndex] = 1;
        }
    }
    return tau;
}

std::vector<std::vector<double>> getPhiDiagonals(size_t rowSize, size_t numRotations) {
    std::vector<std::vector<double>> phiDiagonals(2, std::vector<double>(rowSize * rowSize, 0));
    ;
    for (size_t i = 0; i < rowSize * rowSize; i++) {
        if (i % rowSize < rowSize - numRotations) {
            phiDiagonals[0][i] = 1;
        }
    }

    for (size_t i = 0; i < rowSize * rowSize; i++) {
        if (rowSize - numRotations <= i % rowSize && i % rowSize < rowSize) {
            phiDiagonals[1][i] = 1;
        }
    }
    return phiDiagonals;
}


struct MatrixMatrixProductPrecomputations getMatrixMatrixProductPrecomputations(
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context, int rowSize, int colSize) {

    std::vector<std::vector<double>> sigmaDiagonals = getDiagonals(getSigmaPermutationMatrix(rowSize));
    std::vector<std::vector<double>> tauDiagonals = getDiagonals(getTauPermutationMatrix(rowSize));

    std::vector<lbcrypto::Plaintext> sigmaPlaintexts(sigmaDiagonals.size());
    std::vector<lbcrypto::Plaintext> tauPlaintexts(tauDiagonals.size());
    std::vector<std::vector<lbcrypto::Plaintext>> phiPlaintexts;

    assert(sigmaDiagonals.size() == rowSize * rowSize);
    assert(tauDiagonals.size() == rowSize * rowSize);

    for (int i = 0; i < rowSize; i++) {
        lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(sigmaDiagonals[i]);
        sigmaPlaintexts[i] = ptxtSigma;
    }
    for (int i = rowSize * rowSize - rowSize; i < rowSize * rowSize; i++) {
        lbcrypto::Plaintext ptxtSigma = context->MakeCKKSPackedPlaintext(sigmaDiagonals[i]);
        sigmaPlaintexts[i] = ptxtSigma;
    }
    for (int i = 0; i < rowSize * rowSize; i += rowSize) {
        lbcrypto::Plaintext ptxtTau = context->MakeCKKSPackedPlaintext(tauDiagonals[i]);
        tauPlaintexts[i] = ptxtTau;
    }
    for (int i = 0; i < rowSize; i++) {
        std::vector<std::vector<double>> phi = getPhiDiagonals(rowSize, i);
        lbcrypto::Plaintext ptxtPhi1 = context->MakeCKKSPackedPlaintext(phi[0]);
        lbcrypto::Plaintext ptxtPhi2 = context->MakeCKKSPackedPlaintext(phi[1]);
        std::vector<lbcrypto::Plaintext> phiVec = {ptxtPhi1, ptxtPhi2};
        phiPlaintexts.emplace_back(phiVec);
    }

    struct MatrixMatrixProductPrecomputations precomp;
    precomp.rowSize = rowSize;
    precomp.sigmaPlaintexts = sigmaPlaintexts;
    //precomp.tauVectors = tauDiagonals;
    precomp.tauPlaintexts = tauPlaintexts;
    precomp.phiPlaintexts = phiPlaintexts;

    return precomp;
}

void MatrixMatrixProductSquare(lbcrypto::CryptoContext<lbcrypto::DCRTPoly>& context,
                               lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cMat1,
                               lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cMat2, uint32_t rowSize,
                               lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cProduct,
                               struct MatrixMatrixProductPrecomputations precomp) {

    auto linearTransform1 = context->EvalMult(cMat1, precomp.sigmaPlaintexts[0]);
    for (size_t i = 1; i < rowSize; i++) {
        auto rotatedCt = context->EvalAtIndex(cMat1, i);
        auto productCt = context->EvalMult(rotatedCt, precomp.sigmaPlaintexts[i]);
        linearTransform1 = context->EvalAdd(linearTransform1, productCt);

        rotatedCt = context->EvalAtIndex(cMat1, -i);
        productCt = context->EvalMult(rotatedCt, precomp.sigmaPlaintexts[rowSize * rowSize - i]);
        linearTransform1 = context->EvalAdd(linearTransform1, productCt);
    }

    // Step 1-2
    auto linearTransform2 = context->EvalMult(cMat2, precomp.tauPlaintexts[0]);
    for (size_t i = 1; i < rowSize; i++) {
        auto rotatedCt = context->EvalAtIndex(cMat2, i * rowSize);
        auto productCt = context->EvalMult(rotatedCt, precomp.tauPlaintexts[i * rowSize]);
        linearTransform2 = context->EvalAdd(linearTransform2, productCt);
    }

    //cProduct = linearTransform1;
    //return;

    // Steps 2 and 3

    cProduct = context->EvalMult(linearTransform1, linearTransform2);
    for (size_t i = 1; i < rowSize; i++) {
        // Step 2
        auto rotatedCt = context->EvalAtIndex(linearTransform1, i);
        auto productCt1 = context->EvalMult(rotatedCt, precomp.phiPlaintexts[i][0]);

        rotatedCt = context->EvalAtIndex(linearTransform1, i - rowSize);
        auto productCt2 = context->EvalMult(rotatedCt, precomp.phiPlaintexts[i][1]);
        auto linearTransformPhi = context->EvalAdd(productCt1, productCt2);

        auto linearTransformPsi = context->EvalAtIndex(linearTransform2, i * rowSize);

        // Step 3
        auto tempProduct = context->EvalMult(linearTransformPhi, linearTransformPsi);
        cProduct = context->EvalAdd(cProduct, tempProduct);
    }
}

// matrix multiplication
void MatrixMatrixProduct(std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix1,
                         std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix2, uint32_t rowSize,
                         std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& product,
                         struct MatrixMatrixProductPrecomputations precomp) {
    auto cc = matrix1[0][0]->GetCryptoContext();

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> row;
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> dotProd;
            MatrixMatrixProductSquare(cc, matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            auto start = std::chrono::high_resolution_clock::now();

            for (size_t k = 1; k < matrix2.size(); k++) {
                lbcrypto::Ciphertext<lbcrypto::DCRTPoly> dotProdNew;
                MatrixMatrixProductSquare(cc, matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                cc->EvalAddInPlace(dotProd, dotProdNew);
            }
            row.emplace_back(dotProd);
            auto end = std::chrono::high_resolution_clock::now();
            if (matrix2.size() > 1)
                std::cout << "Duration: " << (std::chrono::duration_cast<std::chrono::seconds>(end - start).count())
                          << " s." << std::endl;
        }
        product.emplace_back(row);
    }
}

// matrix multiplication with bias
void MatrixMatrixProductwithBias(std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix1,
                                 std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix2,
                                 uint32_t rowSize,
                                 std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& product,
                                 struct MatrixMatrixProductPrecomputations precomp,
                                 lbcrypto::Ciphertext<lbcrypto::DCRTPoly> bias) {
    auto cc = matrix1[0][0]->GetCryptoContext();

    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> row;
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            lbcrypto::Ciphertext<lbcrypto::DCRTPoly> dotProd;
            MatrixMatrixProductSquare(cc, matrix1[i][0], matrix2[0][j], rowSize, dotProd, precomp);

            auto start = std::chrono::high_resolution_clock::now();

            for (size_t k = 1; k < matrix2.size(); k++) {
                lbcrypto::Ciphertext<lbcrypto::DCRTPoly> dotProdNew;
                MatrixMatrixProductSquare(cc, matrix1[i][k], matrix2[k][j], rowSize, dotProdNew, precomp);
                cc->EvalAddInPlace(dotProd, dotProdNew);
            }
            cc->EvalAddInPlace(dotProd, bias);
            row.emplace_back(dotProd);
            auto end = std::chrono::high_resolution_clock::now();

            std::cout << "Duration: " << (std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count())
                      << " s." << std::endl;
        }
        product.emplace_back(row);
    }
}

std::vector<double> extractAndLinearizeMatrixBlock(const std::vector<std::vector<double>>& matrix,
                               size_t numSlots,
                               size_t rowSize,   // #cols per block
                               size_t colSize,   // #rows per block
                               size_t offsetRows,
                               size_t offsetCols) {
    if (rowSize == 0 || colSize == 0) throw std::invalid_argument("rowSize/colSize must be > 0");
    if (matrix.empty() || matrix[0].empty()) return std::vector<double>(numSlots, 0.0);

    const size_t M = matrix.size();
    const size_t N = matrix[0].size();

    const size_t endRows = std::min(offsetRows + rowSize, M); // rows advance by colSize
    const size_t endCols = std::min(offsetCols + colSize, N); // cols advance by rowSize

    std::vector<double> vec(numSlots, 0.0);

    for (size_t i = offsetRows; i < endRows; ++i) {
        const size_t r = i - offsetRows; // 0..colSize-1
        for (size_t j = offsetCols; j < endCols; ++j) {
            const size_t c = j - offsetCols; // 0..rowSize-1
            vec[r * colSize + c] = matrix[i][j]; // stride = rowSize (#cols)
        }
    }
    return vec;
}

std::vector<std::vector<std::vector<double>>>extractAndLinearizeMatrix(const std::vector<std::vector<double>>& matrix, size_t numSlots, size_t rowSize, size_t colSize) {  
    if (colSize == 0) { colSize = rowSize; }
   
    if (matrix.empty() || matrix[0].empty()) return {};
    if (rowSize == 0) throw std::invalid_argument("rowSize must be > 0");
    if (numSlots % rowSize != 0) throw std::invalid_argument("numSlots % rowSize != 0");

    const size_t M = matrix.size();
    const size_t N = matrix[0].size();

    const size_t numBlockRows = static_cast<size_t>(std::ceil(double(M) / double(rowSize)));
    const size_t numBlockCols = static_cast<size_t>(std::ceil(double(N) / double(colSize)));


    std::vector<std::vector<std::vector<double>>> mat(
        numBlockRows, std::vector<std::vector<double>>(numBlockCols));

    for (size_t bi = 0; bi < numBlockRows; ++bi) {
        const size_t offR = bi * rowSize; // step by #rows per block
        for (size_t bj = 0; bj < numBlockCols; ++bj) {
            const size_t offC = bj * colSize; // step by #cols per block
            mat[bi][bj] = extractAndLinearizeMatrixBlock(matrix, numSlots, rowSize, colSize, offR, offC);
        }
    }
    return mat;
}

std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> EncryptMatrix(
    const std::vector<std::vector<std::vector<double>>>& matrix, lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey,
    int level) {
    std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> ctMatrix(matrix.size());
    auto cc = publicKey->GetCryptoContext();
    for (size_t i = 0; i < matrix.size(); i++) {
        for (size_t j = 0; j < matrix[0].size(); j++) {
            lbcrypto::Plaintext ptxt1;
            if (level) {
                ptxt1 = cc->MakeCKKSPackedPlaintext(matrix[i][j], 1, level);
            } else {
                ptxt1 = cc->MakeCKKSPackedPlaintext(matrix[i][j]);
            }
            ctMatrix[i].emplace_back(cc->Encrypt(publicKey, ptxt1));
        }
    }
    return ctMatrix;
}

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> EncryptVector(const std::vector<double>& bias,
                                                       lbcrypto::PublicKey<lbcrypto::DCRTPoly> publicKey) {

    auto cc = publicKey->GetCryptoContext();
    lbcrypto::Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(bias);

    return cc->Encrypt(publicKey, ptxt1);
}

std::vector<std::vector<std::vector<double>>> DecryptMatrix(
    const std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& matrix,
    lbcrypto::PrivateKey<lbcrypto::DCRTPoly> privateKey, int numSlots) {

    std::vector<std::vector<std::vector<double>>> ptMatrix(matrix.size());

    lbcrypto::Plaintext result;
    auto cc = privateKey->GetCryptoContext();

    for (size_t i = 0; i < matrix.size(); i++) {
        ptMatrix[i].reserve(matrix[0].size());
        for (size_t j = 0; j < matrix[0].size(); j++) {
            cc->Decrypt(privateKey, matrix[i][j], &result);
            result->SetLength(numSlots);
            ptMatrix[i].emplace_back(result->GetRealPackedValue());
            // std::cout << "Error: " << result->GetLogError() << " " << result->GetLogPrecision() << std::endl;
        }
    }

    std::cout << std::fixed << std::setprecision(0)
              << "Estimated precision: " << result->GetLogPrecision() << " bits";

    return ptMatrix;
}


std::vector<std::vector<double>> convertToLargeMatrix(
   const std::vector<std::vector<std::vector<double>>>& blockedMatrix, size_t rowSize, size_t colSize) {
  
   if (blockedMatrix.empty() || blockedMatrix[0].empty()) {
       return {};
   }


   if (colSize == 0) { colSize = rowSize; };


   size_t numBlockRows = blockedMatrix.size();
   size_t numBlockCols = blockedMatrix[0].size();
   size_t largeMatrixRows = numBlockRows * rowSize;
   size_t largeMatrixCols = numBlockCols * colSize;


   std::vector<std::vector<double>> largeMatrix(largeMatrixRows, std::vector<double>(largeMatrixCols, 0.0));


   for (size_t blockRow = 0; blockRow < numBlockRows; ++blockRow) {
       for (size_t blockCol = 0; blockCol < numBlockCols; ++blockCol) {
           const auto& block = blockedMatrix[blockRow][blockCol];


           if (block.size() != rowSize * colSize) {
            //    std::cerr << "rowSize: " << rowSize << "colSize: " << colSize << "; blockSize: " << block.size() << std::endl;
            //    throw std::invalid_argument("Inconsistent block size detected");
           }


           for (size_t i = 0; i < rowSize; ++i) {
               for (size_t j = 0; j < colSize; ++j) {
                   largeMatrix[blockRow * rowSize + i][blockCol * colSize + j] = block[i * colSize + j];
               }
           }
       }
   }


   return largeMatrix;
}




// Function to print a matrix nicely
void printMatrix(const std::vector<std::vector<double>>& matrix, uint32_t horizontalPrintSize,
                 uint32_t verticalPrintSize, const std::string& label, bool fullPrint, int precision, bool if_first) {
    uint32_t numRows = matrix.size();
    uint32_t numCols = matrix[0].size();

    std::cout << std::endl << label << " of Dim(" << numRows << "x" << numCols << "): \n";

    // Set precision for double output
    std::cout << std::fixed << std::setprecision(precision);

    if (if_first){
        for (uint32_t i = 0; i < verticalPrintSize; i++) {
                for (uint32_t j = 0; j < horizontalPrintSize; j++) {
                    std::cout << std::fixed << std::setprecision(precision) << matrix[i][j] << " ";
                }
                std::cout << "..." << std::endl;
            }
        std::cout << std::endl;
    }

    else {
        if (fullPrint) {
            // Print the entire matrix
            for (uint32_t i = 0; i < numRows; i++) {
                for (uint32_t j = 0; j < numCols; j++) {
                    // std::cout << std::setw(20) << matrix[i][j] << " ";
                    std::cout << std::fixed << std::setprecision(precision) << matrix[i][j] << " ";
                }
                std::cout << std::endl;
            }
        } else {
            // Print first verticalPrintSize rows
            for (uint32_t i = 0; i < verticalPrintSize; i++) {
                for (uint32_t j = 0; j < numCols; j++) {
                    if (j < horizontalPrintSize || j >= numCols - horizontalPrintSize) {
                        // Align entries with setw
                        // std::cout << std::setw(20) << matrix[i][j] << " ";
                        std::cout << std::fixed << std::setprecision(precision) << matrix[i][j] << " ";
                    } else if (j == horizontalPrintSize) {
                        std::cout << "   ... ";
                    }
                }
                std::cout << std::endl;
            }

            // Print ... if there are more rows
            if (numRows > verticalPrintSize * 2) {
                std::cout << "     ...     " << std::endl;
                std::cout << "     ...     " << std::endl;
            }

            // Print last verticalPrintSize rows
            for (uint32_t i = numRows - verticalPrintSize; i < numRows; i++) {
                for (uint32_t j = 0; j < numCols; j++) {
                    if (j < horizontalPrintSize || j >= numCols - horizontalPrintSize) {
                        // std::cout << std::setw(20) << matrix[i][j] << " ";
                        std::cout << std::fixed << std::setprecision(precision) << matrix[i][j] << " ";
                    } else if (j == horizontalPrintSize) {
                        std::cout << "   ... ";
                    }
                }
                std::cout << std::endl;
            }
        }
        std::cout << "" << std::endl << std::endl;
    }
}

std::vector<std::vector<double>> clear_MM(std::vector<std::vector<double>> matrix1,
                                          std::vector<std::vector<double>> matrix2) {
    std::vector<std::vector<double>> product;
    for (size_t i = 0; i < matrix1.size(); i++) {
        std::vector<double> row;
        for (size_t j = 0; j < matrix2[0].size(); j++) {
            double dotProd = 0;
            for (size_t k = 0; k < matrix2.size(); k++) {
                dotProd += matrix1[i][k] * matrix2[k][j];
            }
            row.emplace_back(dotProd);
        }
        product.emplace_back(row);
    }
    return product;
}

// std::vector<double> clear_naive_MM(const std::vector<double>& input, const std::vector<double>& weights, const std::vector<double>& bias, int d_in, int d_out) {
//     assert(input.size() == d_in);
//     assert(weights.size() == d_in * d_out);
//     assert(bias.size() == d_out);

//     std::vector<double> output(d_out, 0.0);
//     for (int j = 0; j < d_out; ++j) {
//         for (int i = 0; i < d_in; ++i) {
//             output[j] += input[i] * weights[j + i * d_out];
//         }
//         output[j] += bias[j];
//     }

//     return output;
// }

// void naive_CCMM(std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>>& ctxt1,
//             std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &ctxt2,
//             std::vector<std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>> &product, int rowSize){
//     auto context = ctxt1[0][0]->GetCryptoContext();

//     for (size_t i = 0; i < ctxt1.size(); i++) {
//         std::vector<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>> columns;
//         for (size_t j = 0; j < ctxt2.size(); j++) {

//             lbcrypto::Ciphertext<lbcrypto::DCRTPoly> sum = context->EvalMult(ctxt1[i][0], ctxt2[0][j]);

//             for (size_t k = 1; k < ctxt1[0].size(); k++) {
//                 lbcrypto::Ciphertext<lbcrypto::DCRTPoly> dotProd = context->EvalMult(ctxt1[i][k], ctxt2[k][j]);
//                 dotProd = rotsum(dotProd, rowSize);
//                 sum = context->EvalAdd(sum, dotProd);
//             }
//             columns.push_back(sum);
//         }
//         product.push_back(columns);
//     }
// }

lbcrypto::Ciphertext<lbcrypto::DCRTPoly> rotsum(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& in, int blockSize,
                                                int padding) {
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> result = in->Clone();
    auto context = in->GetCryptoContext();

    for (int i = 0; i < log2(blockSize); i++) {
        result = context->EvalAdd(result, context->EvalRotate(result, padding * pow(2, i)));
    }

    return result;
}

}  // namespace FIDESlib::CKKS
