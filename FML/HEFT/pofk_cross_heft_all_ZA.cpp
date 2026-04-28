#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <FML/MemoryLogging/MemoryLogging.h>
#include <FML/RandomFields/GaussianRandomField.h>
#include <FML/GadgetUtils/GadgetUtils.h>
#include <FML/HEFT/Hessian.h>
#include <FML/HEFT/HEFTParticle.h>
#include <FML/ParticleTypes/SimpleParticle.h>
#include <mpi.h>

#include <array>
#include <cmath>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>

//=====================================================
// Parameters
//=====================================================
const int Ndim = 3;
double box = 0.0; // to be set with e.g. -L 128.0

const bool fix_amplitude = true;
unsigned int random_seed = 0; // to be set with e.g. -s 1234

int Nmesh = 0; // to be set with e.g. -N 128
int Npart_1D = 0; // to be set with e.g. -N 128
const double buffer_factor = 1.5;

const std::string interpolation_method = "PCS";
const bool interlacing = true;

std::string filename = ""; // to be set with e.g. -fP example_power_spectrum_cb_z0.000.txt

// ztarget snapshot path (edit)
std::string pathandfileprefix_zt = ""; // to be set with e.g. -fg snapshot_TestSim_z0.000/gadget_z0.000

//=====================================================
// Type aliases
//=====================================================
template <class T>
using MPIParticles = FML::PARTICLE::MPIParticles<T>;
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;
using GadgetReader = FML::FILEUTILS::GADGET::GadgetReader;
using RandomGenerator = FML::RANDOM::RandomGenerator;
using GSLRandomGenerator = FML::RANDOM::GSLRandomGenerator;
using Particle = HEFTParticle<Ndim>;

//=====================================================
// Regular3DGrid: map global ID -> q position on [0,1)^3
//=====================================================
class Regular3DGrid {
public:
    explicit Regular3DGrid(std::size_t N)
        : N_(N), totalPoints_(N * N * N), dx_(1.0 / static_cast<double>(N)) {
        if (N == 0) throw std::invalid_argument("N must be >= 1");
    }
    std::array<double, 3> getCoordinates(std::size_t ID) const {
        if (ID >= totalPoints_) throw std::out_of_range("Point ID out of range");
        const std::size_t k = ID % N_;
        const std::size_t tmp = ID / N_;
        const std::size_t j = tmp % N_;
        const std::size_t i = tmp / N_;
        return {double(i) * dx_, double(j) * dx_, double(k) * dx_};
    }
private:
    std::size_t N_;
    std::size_t totalPoints_;
    double dx_;
};

//=====================================================
// Distributed K-weight lookup by contiguous ID ranges
//=====================================================
class DistributedKWeightsByIDRange {
public:
    explicit DistributedKWeightsByIDRange(MPI_Comm comm = MPI_COMM_WORLD) : comm(comm) {}

    // weights_local[k][i] where i runs over local IDs in [id_start_local, id_start_local+nlocal)
    void init(std::vector<std::vector<float>> weights_local_in,
              long long id_start_local_in) {

        weights_local = std::move(weights_local_in);
        id_start_local = id_start_local_in;

        K = (int)weights_local.size();
        if (K <= 0) {
            if (FML::ThisTask == 0) std::cerr << "[FATAL] K=0\n";
            MPI_Abort(comm, 7100);
        }

        nlocal = (long long)weights_local[0].size();
        for (int k = 1; k < K; ++k) {
            if ((long long)weights_local[k].size() != nlocal) {
                if (FML::ThisTask == 0) std::cerr << "[FATAL] weights_local size mismatch\n";
                MPI_Abort(comm, 7101);
            }
        }

        id_start_all.resize(FML::NTasks);
        nlocal_all.resize(FML::NTasks);
        id_end_all.resize(FML::NTasks);

        long long nlocal_send = nlocal;
        MPI_Allgather(&id_start_local, 1, MPI_LONG_LONG, id_start_all.data(), 1, MPI_LONG_LONG, comm);
        MPI_Allgather(&nlocal_send,     1, MPI_LONG_LONG, nlocal_all.data(), 1, MPI_LONG_LONG, comm);

        for (int r = 0; r < FML::NTasks; ++r)
            id_end_all[r] = id_start_all[r] + nlocal_all[r];

        if (FML::ThisTask == 0) {
            for (int r = 1; r < FML::NTasks; ++r) {
                if (id_start_all[r] < id_start_all[r - 1]) {
                    std::cerr << "[FATAL] id_start_all not sorted by rank\n";
                    MPI_Abort(comm, 7102);
                }
            }
        }
        MPI_Barrier(comm);
    }

    // ids are 0-based global IDs in LOCAL SNAPSHOT ORDER
    // returns weights_out[k][i] aligned with ids[i]
    void fetch(const std::vector<long long>& ids,
               std::vector<std::vector<float>>& weights_out) const {

        const int nt = FML::NTasks;
        const int me = FML::ThisTask;
        const size_t N = ids.size();

        weights_out.assign(K, std::vector<float>(N, 0.0f));

        // Build per-destination request lists: send (id, pos)
        std::vector<std::vector<long long>> req_id(nt);
        std::vector<std::vector<int>>       req_pos(nt);

        for (size_t i = 0; i < N; ++i) {
            const long long id = ids[i];
            const int owner = owner_of_id(id);
            if (owner < 0) {
                if (me == 0) std::cerr << "[FATAL] ID " << id << " not covered by any rank\n";
                MPI_Abort(comm, 7110);
            }
            req_id[owner].push_back(id);
            req_pos[owner].push_back((int)i);
        }

        // counts
        std::vector<int> sendcounts(nt, 0), recvcounts(nt, 0);
        for (int r = 0; r < nt; ++r) sendcounts[r] = (int)req_id[r].size();
        MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, comm);

        // displs
        std::vector<int> sdispls(nt, 0), rdispls(nt, 0);
        for (int r = 1; r < nt; ++r) {
            sdispls[r] = sdispls[r - 1] + sendcounts[r - 1];
            rdispls[r] = rdispls[r - 1] + recvcounts[r - 1];
        }
        const int sendtot = (nt > 0 ? sdispls.back() + sendcounts.back() : 0);
        const int recvtot = (nt > 0 ? rdispls.back() + recvcounts.back() : 0);

        // pack send
        std::vector<long long> send_id(sendtot);
        std::vector<int>       send_pos(sendtot);
        for (int r = 0; r < nt; ++r) {
            std::copy(req_id[r].begin(),  req_id[r].end(),  send_id.begin()  + sdispls[r]);
            std::copy(req_pos[r].begin(), req_pos[r].end(), send_pos.begin() + sdispls[r]);
        }

        // recv on owner
        std::vector<long long> recv_id(recvtot);
        std::vector<int>       recv_pos(recvtot);

        MPI_Alltoallv(send_id.data(), sendcounts.data(), sdispls.data(), MPI_LONG_LONG,
                      recv_id.data(), recvcounts.data(), rdispls.data(), MPI_LONG_LONG, comm);

        MPI_Alltoallv(send_pos.data(), sendcounts.data(), sdispls.data(), MPI_INT,
                      recv_pos.data(), recvcounts.data(), rdispls.data(), MPI_INT, comm);

        // owner computes replies: send back pos + K floats
        std::vector<int> reply_pos(recvtot);
        std::vector<std::vector<float>> reply_w(K, std::vector<float>(recvtot));

        for (int j = 0; j < recvtot; ++j) {
            const long long id = recv_id[j];
            const int pos = recv_pos[j];
            reply_pos[j] = pos;

            const long long off = id - id_start_local;
            if (off < 0 || off >= nlocal) {
                std::cerr << "[Rank " << me << "] [FATAL] owner mismatch id=" << id
                          << " off=" << off << " id_start=" << id_start_local
                          << " nlocal=" << nlocal << "\n";
                MPI_Abort(comm, 7111);
            }
            for (int k = 0; k < K; ++k)
                reply_w[k][j] = weights_local[k][(size_t)off];
        }

        // send replies back
        std::vector<int> back_pos(sendtot);
        MPI_Alltoallv(reply_pos.data(), recvcounts.data(), rdispls.data(), MPI_INT,
                      back_pos.data(), sendcounts.data(), sdispls.data(), MPI_INT, comm);

        std::vector<std::vector<float>> back_w(K, std::vector<float>(sendtot));
        for (int k = 0; k < K; ++k) {
            MPI_Alltoallv(reply_w[k].data(), recvcounts.data(), rdispls.data(), MPI_FLOAT,
                          back_w[k].data(), sendcounts.data(), sdispls.data(), MPI_FLOAT, comm);
        }

        // unpack (preserves local snapshot order)
        for (int j = 0; j < sendtot; ++j) {
            const int pos = back_pos[j];
            if (pos < 0 || (size_t)pos >= N) {
                std::cerr << "[Rank " << me << "] [FATAL] returned pos out of range " << pos << "\n";
                MPI_Abort(comm, 7112);
            }
            for (int k = 0; k < K; ++k)
                weights_out[k][(size_t)pos] = back_w[k][j];
        }
    }

private:
    int owner_of_id(long long id) const {
        int lo = 0, hi = (int)id_start_all.size() - 1;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (id < id_start_all[mid]) hi = mid - 1;
            else if (id >= id_end_all[mid]) lo = mid + 1;
            else return mid;
        }
        return -1;
    }

private:
    MPI_Comm comm;
    int K = 0;

    std::vector<std::vector<float>> weights_local; // [K][nlocal]
    long long id_start_local = 0;
    long long nlocal = 0;

    std::vector<long long> id_start_all, nlocal_all, id_end_all;
};

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char *argv[]) {
    // Iterate through arguments starting from index 1
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-N" && i + 1 < argc) {
            Npart_1D = std::stoi(argv[++i]);
            Nmesh = Npart_1D;
            std::cout << "Read Npart = " << Npart_1D << std::endl;
            std::cout << "Read Nmesh = " << Nmesh << std::endl;
        } else if (arg == "-L" && i + 1 < argc) {
            box = std::stof(argv[++i]);
            std::cout << "Read boxsize = " << box << std::endl;
        } else if (arg == "-s" && i + 1 < argc) {
            random_seed = std::stoi(argv[++i]);
            std::cout << "Read random seed = " << random_seed << std::endl;
        } else if (arg == "-fP" && i + 1 < argc) {
            filename = argv[++i];
            std::cout << "Read initial power spectrum filename = " << filename << std::endl;
        } else if (arg == "-fg" && i + 1 < argc) {
            pathandfileprefix_zt = argv[++i];
            std::cout << "Read gadget snapshot filename stem = " << pathandfileprefix_zt << std::endl;
        } else {
            std::cerr << "Unknown argument " << arg << std::endl;
            return 1;
        }
    }

#ifdef MEMORY_LOGGING
    auto * mem = FML::MemoryLog::get();
#endif

    auto fatal = [&](const std::string& msg) {
        if (FML::ThisTask == 0) std::cerr << "[FATAL] " << msg << "\n";
        MPI_Abort(MPI_COMM_WORLD, 999);
    };

    //============================================================
    // Read P(k)
    //============================================================
    if (FML::ThisTask == 0) {
        std::cout << "#=====================================================\n";
        std::cout << " Reading and splining P(k)\n";
        std::cout << "#=====================================================\n";
    }

    std::ifstream fp(filename.c_str());
    if (!fp.is_open()) throw std::runtime_error("Cannot open file: " + filename);

    std::vector<double> logk, logpofk;
    std::string line;
    for (;;) {
        if (fp.peek() == '#') std::getline(fp, line); // skip comments
        double kin, pofkin;
        fp >> kin;
        if (fp.eof()) break;
        fp >> pofkin;
        logk.push_back(std::log(kin));
        logpofk.push_back(std::log(pofkin));
    }
    FML::INTERPOLATION::SPLINE::Spline logpofk_spline(logk, logpofk);

    auto Pofk_of_kBox_over_volume = [&](double kBox) {
        const double lk = std::log(kBox / box);
        const double volume = std::pow(box, Ndim);
        return std::exp(logpofk_spline(lk)) / volume;
    };

    //============================================================
    // RNG
    //============================================================
#ifdef USE_GSL
    std::shared_ptr<RandomGenerator> rng = std::make_shared<GSLRandomGenerator>();
#else
    throw std::runtime_error("Compile with -DUSE_GSL");
#endif
    rng->set_seed(random_seed);

    //============================================================
    // (1) Build 4 Lagrangian fields from ONE delta_L realization
    //============================================================
    if (FML::ThisTask == 0) {
        std::cout << "#=====================================================\n";
        std::cout << " Building Lagrangian Fields\n";
        std::cout << "#=====================================================\n";
    }
    const auto nextra =
        FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);

    // One realization δ_L(k)
    FFTWGrid<Ndim> delta_L_fourier(Nmesh, nextra.first, nextra.second);
    FML::RANDOM::GAUSSIAN::generate_gaussian_random_field_fourier(
        delta_L_fourier, rng.get(), Pofk_of_kBox_over_volume, fix_amplitude);

    // δ_L(x)
    FFTWGrid<Ndim> delta_L_real = delta_L_fourier;
    delta_L_real.fftw_c2r();

    // s^2(x)
    const double norm = 1.0;
    const bool hessian_of_potential_of_f = true;
    FFTWGrid<Ndim> s2_L_real(Nmesh, nextra.first, nextra.second);
    s2_L_real.set_grid_status_real(true);
    FML::HESSIAN::ComputeHessianWithFT(delta_L_real, s2_L_real, norm, hessian_of_potential_of_f);

    // δ^2(x) from δ(k)
    FFTWGrid<Ndim> delta2_L_real(Nmesh, nextra.first, nextra.second);
    delta2_L_real.set_grid_status_real(true);
    FML::HESSIAN::Compute_delta2(delta_L_fourier, delta2_L_real);

    // 1(x)
    FFTWGrid<Ndim> one_L_real(Nmesh, nextra.first, nextra.second);
    one_L_real.set_grid_status_real(true);
    one_L_real.fill_real_grid(1.0);

    // free Fourier
    delta_L_fourier.free();

    //============================================================
    // (2) Lagrangian particle grid with contiguous IDs, interpolate 4 fields -> local weights
    //============================================================
    if (FML::ThisTask == 0) {
        std::cout << "#=====================================================\n";
        std::cout << " Building Lagrangian particle grid (Np1d^3)\n";
        std::cout << "#=====================================================\n";
    }
    MPIParticles<Particle> part;
    part.create_particle_grid(Npart_1D, buffer_factor, FML::xmin_domain, FML::xmax_domain);
    part.info();

    long long id_start_local = 0;
    if constexpr (FML::PARTICLE::has_set_id<Particle>()) {

        long long npart_local = part.get_npart();
        auto part_per_task = FML::GatherFromTasks(&npart_local);

        id_start_local = 0;
        for (int i = 0; i < FML::ThisTask; i++) id_start_local += part_per_task[i];

        long long count = 0;
        for (auto & p : part) FML::PARTICLE::SetID(p, id_start_local + count++);
    } else {
        fatal("Particle type has no set_id()");
    }

    // set q positions based on ID
    Regular3DGrid qgrid(Npart_1D);
    auto * lag_ptr = part.get_particles_ptr();
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t ind = 0; ind < part.get_npart(); ind++) {
        auto * pos = lag_ptr[ind].get_pos();
        const size_t id = lag_ptr[ind].get_id();
        const auto q = qgrid.getCoordinates(id);
        for (int d = Ndim - 1; d >= 0; --d) pos[d] = q[d];
    }

    // interpolate 4 grids to particles -> local weights by ID
    std::vector<float> w0_local, w1_local, w2_local, w3_local;

    FML::INTERPOLATION::interpolate_grid_to_particle_positions<Ndim, Particle>(
        one_L_real, part.get_particles_ptr(), part.get_npart(), w0_local, interpolation_method);

    FML::INTERPOLATION::interpolate_grid_to_particle_positions<Ndim, Particle>(
        delta_L_real, part.get_particles_ptr(), part.get_npart(), w1_local, interpolation_method);

    FML::INTERPOLATION::interpolate_grid_to_particle_positions<Ndim, Particle>(
        delta2_L_real, part.get_particles_ptr(), part.get_npart(), w2_local, interpolation_method);

    FML::INTERPOLATION::interpolate_grid_to_particle_positions<Ndim, Particle>(
        s2_L_real, part.get_particles_ptr(), part.get_npart(), w3_local, interpolation_method);

    // Free Lagrangian particles (keep grids if you want later; otherwise free them too)
    part.free();

    // Free grids (optional, saves memory before snapshot)
    one_L_real.free();
    delta_L_real.free();
    delta2_L_real.free();
    s2_L_real.free();

    // setup distributed fetcher
    std::vector<std::vector<float>> weights_local(4);
    weights_local[0] = std::move(w0_local);
    weights_local[1] = std::move(w1_local);
    weights_local[2] = std::move(w2_local);
    weights_local[3] = std::move(w3_local);

    DistributedKWeightsByIDRange WK;
    WK.init(std::move(weights_local), id_start_local);

    //============================================================
    // (3) Read snapshot ztarget
    //============================================================
    GadgetReader g;
    FML::Vector<Particle> external_part;
    const bool only_keep_part_in_domain = true;
    const bool verbose = false;
    g.read_gadget(pathandfileprefix_zt, external_part, buffer_factor, only_keep_part_in_domain, verbose);

    long long NumPartTotal_local = external_part.size();
    long long NumPartTotal = NumPartTotal_local;
    FML::SumOverTasks(&NumPartTotal);

    part.move_from(std::move(external_part));

    // infer id_offset (0-based vs 1-based)
    long long id_min_local = LLONG_MAX, id_max_local = LLONG_MIN;
    if (part.get_npart() > 0) {
        for (size_t i = 0; i < part.get_npart(); ++i) {
            const long long id = part[i].get_id();
            id_min_local = std::min(id_min_local, id);
            id_max_local = std::max(id_max_local, id);
        }
    }
    long long id_min = id_min_local, id_max = id_max_local;
    MPI_Allreduce(MPI_IN_PLACE, &id_min, 1, MPI_LONG_LONG, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &id_max, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (id_min == LLONG_MAX && id_max == LLONG_MIN) fatal("Snapshot appears empty");
    if (id_min < 0) fatal("Snapshot has negative IDs");
    if (id_min == id_max) fatal("Snapshot has no usable IDs");

    long long id_offset = 0;
    if (id_min == 1 && id_max == NumPartTotal) id_offset = 1;

    if (FML::ThisTask == 0) {
        std::cout << "[DBG] snapshot ID range = [" << id_min << "," << id_max << "], NumPartTotal=" << NumPartTotal << "\n";
        std::cout << "[DBG] inferred id_offset = " << id_offset << "\n";
    }

    std::vector<long long> ids0(part.get_npart());
    for (size_t i = 0; i < part.get_npart(); ++i) {
        const long long id0 = part[i].get_id() - id_offset;
        if (id0 < 0) fatal("Found id_raw < id_offset");
        ids0[i] = id0;
    }

    // fetch 4 weights aligned to LOCAL SNAPSHOT ORDER
    std::vector<std::vector<float>> w_snap;
    WK.fetch(ids0, w_snap);
    if ((int)w_snap.size() != 4) fatal("w_snap.size()!=4");

    //============================================================
    // (4) Full 4x4 advected spectra with memory-light double-Fourier loop
    //============================================================
    const int K = 4;
    const std::array<std::string, K> tag = {"11", "dd", "d2d2", "s2s2"};

    // Save original masses (double)
    std::vector<double> m0(part.get_npart());
    for (size_t i = 0; i < part.get_npart(); ++i) m0[i] = part[i].mass;

    auto deposit_fourier = [&](const std::vector<float>& w) {
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (size_t i = 0; i < part.get_npart(); ++i) {
            part[i].mass = m0[i] * (double)w[i];
        }

        FFTWGrid<Ndim> grid(Nmesh, nextra.first, nextra.second);
        FML::INTERPOLATION::particles_to_grid(part.get_particles_ptr(),
                                             part.get_npart(),
                                             part.get_npart_total(),
                                             grid,
                                             interpolation_method);

        grid.fftw_r2c();
        FML::INTERPOLATION::deconvolve_window_function_fourier<Ndim>(grid, interpolation_method);
        return grid;
    };

    // Store only i<=j spectra (10 of them)
    auto idx = [&](int i, int j) { return i * K + j; };

    std::vector<FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim>>
        P(K * K, FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim>(Nmesh / 2));
    std::vector<bool> have(K * K, false);

    for (int i = 0; i < K; ++i) {

        auto Gi = deposit_fourier(w_snap[i]);

        // auto
        {
            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> p(Nmesh / 2);
            FML::CORRELATIONFUNCTIONS::bin_up_power_spectrum(Gi, p);
            p.scale(box);
            P[idx(i, i)] = std::move(p);
            have[idx(i, i)] = true;
        }

        for (int j = i + 1; j < K; ++j) {

            auto Gj = deposit_fourier(w_snap[j]);

            FML::CORRELATIONFUNCTIONS::PowerSpectrumBinning<Ndim> p(Nmesh / 2);
            FML::CORRELATIONFUNCTIONS::bin_up_cross_power_spectrum(Gi, Gj, p);
            p.scale(box);

            P[idx(i, j)] = std::move(p);
            have[idx(i, j)] = true;

            Gj.free();
        }

        Gi.free();
    }

    // restore masses
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < part.get_npart(); ++i) part[i].mass = m0[i];

    // Output (10 unique spectra)
    if (FML::ThisTask == 0) {

        std::ofstream out("pofk_advected_4x4.txt");
        out << "# k[h/Mpc]  "
            << "P_11            P_dd            P_d2d2            P_s2s2            "
            << "P_1d            P_1d2            P_1s2            P_dd2            P_ds2            P_d2s2\n";

        const int nbin = P[idx(0, 0)].n;
        for (int b = 0; b < nbin; ++b) {

            const double k = P[idx(0, 0)].kbin[b];
            out << k << " ";

            // autos
            out << P[idx(0, 0)].pofk[b] << " ";
            out << P[idx(1, 1)].pofk[b] << " ";
            out << P[idx(2, 2)].pofk[b] << " ";
            out << P[idx(3, 3)].pofk[b] << " ";

            // crosses
            out << P[idx(0, 1)].pofk[b] << " ";
            out << P[idx(0, 2)].pofk[b] << " ";
            out << P[idx(0, 3)].pofk[b] << " ";
            out << P[idx(1, 2)].pofk[b] << " ";
            out << P[idx(1, 3)].pofk[b] << " ";
            out << P[idx(2, 3)].pofk[b] << "\n";
        }

        std::cout << "[OK] wrote pofk_advected_4x4.txt\n";
    }

#ifdef MEMORY_LOGGING
    mem->print();
#endif

}

