#ifndef HESSIAN_HEADER
#define HESSIAN_HEADER

#include <vector>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>

#include <FML/FFTWGrid/FFTWGrid.h>

namespace FML {
    namespace HESSIAN {

        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;

        //=================================================================================
        // Computes the Hessian matrix of a grid [norm * f] via Fourier transforms.
        // If hessian_of_potential_of_f is true then we compute the Hessian
        // phi_ij where D^2 phi = norm * f_real
        // Since f_ij = f_ji we only compute the elements for j >= i and they are stored in
        // the order fxx fxy ... fyy fyz ... etc. in hessian_real
        // In 2D: [fxx fxy fyy]
        // In 3D: [fxx fxy fxz fyy fyz fzz]
        //=================================================================================

        template <int N>
        void ComputeHessianWithFT(FFTWGrid<N> & f_real,
                                  std::vector<FFTWGrid<N>> & hessian_real,
                                  double norm = 1.0,
                                  bool hessian_of_potential_of_f = false) {

            assert_mpi(f_real.get_nmesh() > 0, "[ComputeHessianWithFT] f_real grid is not allocated\n");

            // Helper function to go from f(k) -> DiDj f
            // Assuing we have f(k) in grid
            auto ComputeSecondDerivative = [&](FFTWGrid<N> & grid, int i1, int i2) {
                if (FML::ThisTask == 0)
                    std::cout << "[ComputeHessianWithFT::ComputeSecondDerivative] Computing phi_" << i1 << "," << i2
                              << "\n";
                std::array<double, N> kvec;
                double kmag2;
                for (auto & fourier_index : grid.get_fourier_range()) {
                    grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                    // From f(k) -> -ika ikb f(k) / k^2 = (ka kb / k^2) f(k)
                    auto value = grid.get_fourier_from_index(fourier_index);
                    double factor = -norm * kvec[i1] * kvec[i2];
                    if (hessian_of_potential_of_f)
                        factor *= -1.0 / kmag2;
                    value *= factor;

                    grid.set_fourier_from_index(fourier_index, value);
                }

                // Set DC mode to zero (we divide by 0 in the loop so fix this)
                if (FML::ThisTask == 0)
                    grid.set_fourier_from_index(0, 0.0);

                // Back to real space
                grid.fftw_c2r();
            };

            // Fourier transform it
            f_real.fftw_r2c();
            FFTWGrid<N> & f_fourier = f_real;

            // Allocate grids
            hessian_real.resize(N * (N - 1));

            // Compute hessian matrix
            int count = 0;
            for (int idim = 0; idim < N; idim++) {
                for (int idim2 = idim; idim2 < N; idim2++) {
                    hessian_real[count] = f_fourier;
                    ComputeSecondDerivative(hessian_real[count], idim, idim2);
                    count++;
                }
            }
        }

        //=================================================================================
        // For each point in the grid compute eigenvectors and eigenvalues of the tensor
        // H_ij where tensor_real contains the N(N-1)/2 grids [ 00,01,02,..,11,12,...,NN ]
        //
        // Eigenvalues are ordered in descending order
        // Eigenvectors are stored in row major order in the grid vector
        // This allocates N grids if compute_eigenvectors = false and N(N+1) grids otherwise
        //=================================================================================

        template <int N>
        void SymmetricTensorEigensystem(std::vector<FFTWGrid<N>> & tensor_real,
                                        std::vector<FFTWGrid<N>> & eigenvalues,
                                        std::vector<FFTWGrid<N>> & eigenvectors,
                                        bool compute_eigenvectors = false) {

            assert_mpi(tensor_real.size() > 0, "[SymmetricTensorEigensystem] tensor_real is not allocated\n");
            assert_mpi(tensor_real[0].get_nmesh() > 0,
                       "[SymmetricTensorEigensystem] tensor_real[0] is not allocated\n");
            for (size_t i = 1; i < tensor_real.size(); i++)
                assert_mpi(tensor_real[i].get_nmesh() == tensor_real[i - 1].get_nmesh(),
                           "[SymmetricTensorEigensystem] tensor_real[i] is not allocated\n");

            // N eigenvalues
            eigenvalues.resize(N);
            for (int idim = 0; idim < N; idim++)
                eigenvalues[idim] = tensor_real[0];

            // N eigenvectors with N components
            // We store the components in the same (row major) order as GSL uses
            if (compute_eigenvectors) {
                eigenvectors.resize(N * N);
                for (int i = 0; i < N * N; i++)
                    eigenvectors[i] = tensor_real[0];
            }

            // Set up the GSL stuff we need
            gsl_matrix * matrix = gsl_matrix_alloc(N, N);
            gsl_matrix * evec = gsl_matrix_alloc(N, N);
            gsl_vector * eval = gsl_vector_alloc(N);
            gsl_eigen_symm_workspace * workspace = gsl_eigen_symm_alloc(N);
            gsl_eigen_symmv_workspace * workspacev = gsl_eigen_symmv_alloc(N);

            // Solves the full eigensystem
            auto SolveEigensystem = [&](gsl_matrix * _matrix,
                                        gsl_vector * _eval,
                                        gsl_matrix * _evec,
                                        gsl_eigen_symmv_workspace * _workspace) {
                // Compute eigenvalues and eigenvectors
                gsl_eigen_symmv(_matrix, _eval, _evec, _workspace);
                // Sort in descending order
                gsl_eigen_symmv_sort(_eval, _evec, GSL_EIGEN_SORT_VAL_DESC);
            };

            // Solves just for eigenvalues
            auto SolveEigenvalues =
                [&](gsl_matrix * _matrix, gsl_vector * _eval, gsl_eigen_symm_workspace * _workspace) {
                    // Compute eigenvalues and eigenvectors
                    gsl_eigen_symm(_matrix, _eval, _workspace);
                    // Order the eigenvalues in descending order
                    std::sort(_eval->data, _eval->data + _matrix->size1, std::greater<double>());
                };

            // Loop over all cells
            for (auto & real_index : tensor_real[0].get_real_range()) {

                // Set the matrix
                int count = 0;
                for (int idim = 0; idim < N; idim++) {
                    auto value = tensor_real[count].get_real_from_index(real_index);
                    gsl_matrix_set(matrix, idim, idim, value);
                    count++;
                    for (int idim2 = idim + 1; idim2 < N; idim2++) {
                        value = tensor_real[count].get_real_from_index(real_index);
                        gsl_matrix_set(matrix, idim, idim2, value);
                        gsl_matrix_set(matrix, idim2, idim, value);
                        count++;
                    }
                }

                // Compute eigenvectors+eigenvalues or just eigenvalues
                // In the latter case we sort the eigenvalues
                if (compute_eigenvectors) {
                    SolveEigensystem(matrix, eval, evec, workspacev);

                    // Set eigenvectors
                    for (int i = 0; i < N * N; i++) {
                        eigenvectors[i].set_real_from_index(real_index, evec->data[i]);
                        // For column major order: gsl_matrix_get(evec, i / N, i % N);
                    }

                } else {
                    SolveEigenvalues(matrix, eval, workspace);
                }

                // Store the eigenvalues
                for (int idim = 0; idim < N; idim++)
                    eigenvalues[idim].set_real_from_index(real_index, eval->data[idim]);
            }

            // Free up GSL allocations
            gsl_matrix_free(matrix);
            gsl_matrix_free(evec);
            gsl_vector_free(eval);
            gsl_eigen_symm_free(workspace);
            gsl_eigen_symmv_free(workspacev);
        }

    } // namespace HESSIAN
} // namespace FML
#endif