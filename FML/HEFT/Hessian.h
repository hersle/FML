#ifndef HESSIAN_HEADER
#define HESSIAN_HEADER

#include <vector>

#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>

#include <FML/FFTWGrid/FFTWGrid.h>

namespace FML {

    /// This namespace deals with computing the Hessian of a grid and related algorithms that rely on this.
    namespace HESSIAN {

        template <int N>
        using FFTWGrid = FML::GRID::FFTWGrid<N>;

        //=================================================================================
        /// Computes the Hessian matrix of a grid [norm * f] via Fourier transforms.
        /// If hessian_of_potential_of_f is true then we compute the Hessian
        /// \f$ \phi_{ij} \f$ where \f$ \nabla^2 \phi = norm * f_{\rm real} \f$
        /// Since \f$ f_{ij} = f_{ji} \f$ we only compute the elements for \f$ j \geq i \f$ and they are stored in
        /// the order fxx fxy ... fyy fyz ... etc. in hessian_real
        /// In 2D: [fxx fxy fyy]
        /// In 3D: [fxx fxy fxz fyy fyz fzz]
        ///
        /// @tparam N The dimension we are working in
        ///
        /// @param[in] f_real The grid we are to compute the hessian of
        /// @param[out] hessian_real The hessian of the grid (or its potential, see below)
        /// @param[in] norm A number to scale the grid by if needed (default is 1.0)
        /// @param[in] hessian_of_potential_of_f Compute the hessian of the potential of the grid (default is false)
        ///
        //=================================================================================
        template <int N>
        void ComputeHessianWithFT(const FFTWGrid<N> & f_real,
                                  FFTWGrid<N> & shear_squared,
                                  double norm = 1.0,
                                  bool hessian_of_potential_of_f = false) {

            assert_mpi(f_real.get_nmesh() > 0, "[ComputeHessianWithFT] f_real grid is not allocated\n");

            // Helper function to go from f(k) -> DiDj f (or tidal tensor if hessian_of_potential_of_f=true)
            auto ComputeSecondDerivative = [&](FFTWGrid<N> & grid, int i1, int i2) {
                if (FML::ThisTask == 0)
                    std::cout << "[ComputeHessianWithFT::ComputeSecondDerivative] Computing component (" << i1 << "," << i2 << ")\n";

                // Copy of original f(k) so we can subtract (1/3) f(x) on the diagonal after going to real space
                FFTWGrid<N> original_grid = grid;
                original_grid.fftw_c2r();

                auto Local_nx      = grid.get_local_nx();
                auto Local_x_start = grid.get_local_x_start();

        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    double kmag2;
                    std::array<double, N> kvec;
                    for (auto && fourier_index : grid.get_fourier_range(islice, islice + 1)) {

                        // Skip DC (k=0)
                        if (Local_x_start == 0 && fourier_index == 0)
                            continue;

                        grid.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                        // Start from f(k)
                        auto value = grid.get_fourier_from_index(fourier_index);

                        // DiDj f in Fourier:  (-k_i k_j) f(k)
                        // If we want Hessian of potential:  (+k_i k_j / k^2) f(k)
                        double factor = -norm * kvec[i1] * kvec[i2];
                        if (hessian_of_potential_of_f)
                            factor *= -1.0 / kmag2;

                        value *= factor;
                        grid.set_fourier_from_index(fourier_index, value);
                    }
                }

                // Ensure DC is zero
                if (FML::ThisTask == 0)
                    grid.set_fourier_from_index(0, 0.0);

                // Back to real space
                grid.fftw_c2r();

                // Make it traceless (shear / tidal tensor): subtract (1/3) f(x) from diagonals
                if (i1 == i2) {
                    auto Local_nx_r = grid.get_local_nx();
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
                    for (int islice = 0; islice < Local_nx_r; islice++) {
                        for (auto && real_index : grid.get_real_range(islice, islice + 1)) {
                            auto subtract = original_grid.get_real_from_index(real_index);
                            subtract /= FML::GRID::FloatType(3.0);

                            auto value = grid.get_real_from_index(real_index);
                            grid.set_real_from_index(real_index, value - subtract);
                        }
                    }
                } else {
                    // Pre-scale off-diagonals by sqrt(2) so that sum(value^2) = s_ij s_ij
                    auto Local_nx_r = grid.get_local_nx();
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
                    for (int islice = 0; islice < Local_nx_r; islice++) {
                        for (auto && real_index : grid.get_real_range(islice, islice + 1)) {
                            auto value = grid.get_real_from_index(real_index);
                            value *= FML::GRID::FloatType(std::sqrt(2.0));
                            grid.set_real_from_index(real_index, value);
                        }
                    }
                }
            };

            // ---------------------------------------------------------------------
            // FFT: real -> fourier
            // ---------------------------------------------------------------------
            FFTWGrid<N> f_fourier = f_real;
            f_fourier.fftw_r2c();

            // Symmetric components count = N(N+1)/2
            std::vector<FFTWGrid<N>> hessian_real;
            hessian_real.resize(N * (N + 1) / 2);

            // Compute components (stored as flattened vector over i<=j)
            int count = 0;
            for (int idim = 0; idim < N; idim++) {
                for (int idim2 = idim; idim2 < N; idim2++) {
                    hessian_real[count] = f_fourier;
                    ComputeSecondDerivative(hessian_real[count], idim, idim2);
                    count++;
                }
            }

            if (FML::ThisTask == 0) {
                std::cout << "[ComputeHessianWithFT] Components in shear matrix (flatten vector) = " << count << "\n";
                std::cout << "[ComputeHessianWithFT] Computing shear squared and subtracting <s^2>\n";
            }

            // ---------------------------------------------------------------------
            // Build s^2(x) = s_ij(x) s_ij(x) (with off-diagonals pre-weighted by sqrt(2))
            // ---------------------------------------------------------------------
            auto Local_nx = shear_squared.get_local_nx();
        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : shear_squared.get_real_range(islice, islice + 1)) {
                    double sum_squared = 0.0;
                    for (int ipair = 0; ipair < count; ipair++) {
                        auto v = hessian_real[ipair].get_real_from_index(real_index);
                        sum_squared += v * v;
                    }
                    shear_squared.set_real_from_index(real_index, sum_squared);
                }
            }

            // ---------------------------------------------------------------------
            // Subtract the mean: s^2(x) -> s^2(x) - <s^2>
            // ---------------------------------------------------------------------
            double local_sum = 0.0;
            long long int local_count = 0;

            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : shear_squared.get_real_range(islice, islice + 1)) {
                    local_sum += shear_squared.get_real_from_index(real_index);
                    local_count++;
                }
            }

            double global_sum = local_sum;
            long long int global_count = local_count;

            // MPI-reduce over tasks (FML helpers)
            FML::SumOverTasks(&global_sum);
            FML::SumOverTasks(&global_count);

            const double mean_s2 = global_sum / double(global_count);

        #ifdef USE_OMP
        #pragma omp parallel for
        #endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : shear_squared.get_real_range(islice, islice + 1)) {
                    auto v = shear_squared.get_real_from_index(real_index);
                    shear_squared.set_real_from_index(real_index, v - mean_s2);
                }
            }

            if (FML::ThisTask == 0)
                std::cout << "[ComputeHessianWithFT] <s^2> = " << mean_s2 << " subtracted\n";
        }

        
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template <int N>
        void Compute_k2delta(const FFTWGrid<N> & delta_fourier, const double box, FFTWGrid<N> & k2delta) {

            assert_mpi(delta_fourier.get_nmesh() > 0, "[Compute_k2delta] delta_fourier grid is not allocated\n");
            
            const auto Nmesh = delta_fourier.get_nmesh();
            const auto Local_nx = delta_fourier.get_local_nx();
            const auto Local_x_start = delta_fourier.get_local_x_start();

#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                [[maybe_unused]] double kmag2;
                [[maybe_unused]] std::array<double, N> kvec;
                for (auto && fourier_index : k2delta.get_fourier_range(islice, islice + 1)) {
                    if (Local_x_start == 0 and fourier_index == 0)
                        continue; // DC mode (k=0)

                    // Get wavevector and magnitude
                    k2delta.get_fourier_wavevector_and_norm2_by_index(fourier_index, kvec, kmag2);

                    auto value = - FML::GRID::FloatType(kmag2) * FML::GRID::FloatType(1./box/box) * delta_fourier.get_fourier_from_index(fourier_index);
                    k2delta.set_fourier_from_index(fourier_index, value);
                }
            }
            // Deal with DC mode
            if (Local_x_start == 0)
                k2delta.set_fourier_from_index(0, 0.0);
                
            k2delta.fftw_c2r();
            double k2delta_mean = 0.0;
            long long int ncells = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : k2delta_mean, ncells)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : k2delta.get_real_range(islice, islice + 1)) {
                        auto value = k2delta.get_real_from_index(real_index);
                        k2delta_mean += value;
                        ncells += 1;
                    }
                }
                FML::SumOverTasks(&k2delta_mean);
                FML::SumOverTasks(&ncells);
                
                assert_mpi(ncells == (long long int)(FML::power(Nmesh, N)),
                           "[Compute_k2delta] Number of cells we have summed over does "
                           "not agree with how many cells are in the grid");
                k2delta_mean /= std::pow(Nmesh, N);

                if (FML::ThisTask == 0)
                    std::cout << "[Compute_k2delta] <k2delta>: " << k2delta_mean << "\n";

#ifdef USE_OMP
#pragma omp parallel for
#endif
                // Subtract <delta^2>
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : k2delta.get_real_range(islice, islice + 1)) {
                        auto k2del = k2delta.get_real_from_index(real_index);
                        auto value = (k2del - k2delta_mean);
                        k2delta.set_real_from_index(real_index, value);
                    }
                }
        }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        template <int N>
        void Compute_delta2(const FFTWGrid<N> & delta_fourier, FFTWGrid<N> & delta2_real) {
        
                const auto Nmesh = delta_fourier.get_nmesh();
                const auto Local_nx = delta_fourier.get_local_nx();
                
                // Get delta in real space
                FFTWGrid<N> delta_real = delta_fourier;
                delta_real.add_memory_label("FFTWGrid::compute_delta2::delta_real");
                delta_real.set_grid_status_real(true);
                delta_real.fftw_c2r();

                // Compute delta_real^2
                delta2_real.add_memory_label("FFTWGrid::compute_delta2::delta2_real");
                double delta_squared_mean = 0.0;
                double delta_mean = 0.0;
                long long int ncells = 0;
#ifdef USE_OMP
#pragma omp parallel for reduction(+ : delta_squared_mean, delta_mean, ncells)
#endif
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : delta_real.get_real_range(islice, islice + 1)) {
                        auto delta = delta_real.get_real_from_index(real_index);
                        auto value = delta * delta;
                        delta2_real.set_real_from_index(real_index, value);
                        delta_squared_mean += value;
                        delta_mean += delta;
                        ncells += 1;
                    }
                }
                FML::SumOverTasks(&delta_squared_mean);
                FML::SumOverTasks(&delta_mean);
                FML::SumOverTasks(&ncells);
                
                assert_mpi(ncells == (long long int)(FML::power(Nmesh, N)),
                           "[compute_delta2] Number of cells we have summed over does "
                           "not agree with how many cells are in the grid");
                delta_squared_mean /= std::pow(Nmesh, N);
                delta_mean /= std::pow(Nmesh, N);

                if (FML::ThisTask == 0)
                    std::cout << "[compute_delta2] <delta^2>: " << delta_squared_mean
                              << " <delta>: " << delta_mean << "\n";
                              
#ifdef USE_OMP
#pragma omp parallel for
#endif
                // Subtract <delta^2>
                for (int islice = 0; islice < Local_nx; islice++) {
                    for (auto && real_index : delta2_real.get_real_range(islice, islice + 1)) {
                        auto delta2 = delta2_real.get_real_from_index(real_index);
                        auto value = (delta2 - delta_squared_mean);
                        delta2_real.set_real_from_index(real_index, value);
                    }
                }
        }
        
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

        //=================================================================================
        /// For each point in the grid compute eigenvectors and eigenvalues of the tensor
        /// \f$ H_{ij} \f$ where tensor_real contains the \f$ N(N-1)/2 \f$ grids [ 00,01,02,..,11,12,...,NN ]
        ///
        /// Eigenvalues are ordered in descending order
        ///
        /// Eigenvectors are stored in row major order in the grid vector
        ///
        /// This allocates N grids if compute_eigenvectors = false and N(N+1) grids otherwise
        ///
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
            auto Local_nx = tensor_real[0].get_local_nx();
#ifdef USE_OMP
#pragma omp parallel for
#endif
            for (int islice = 0; islice < Local_nx; islice++) {
                for (auto && real_index : tensor_real[0].get_real_range(islice,islice+1)) {

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
