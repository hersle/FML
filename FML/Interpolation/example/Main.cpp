#include <FML/ComputePowerSpectra/ComputePowerSpectrum.h>
#include <FML/Interpolation/ParticleGridInterpolation.h>
#include <FML/MPIParticles/MPIParticles.h>
#include <cstring>
#include <fstream>

//=======================================================
// Some type aliases
//=======================================================
template <int N>
using FFTWGrid = FML::GRID::FFTWGrid<N>;

//=======================================================
// A simple particle class compatible with MPIParticles
//=======================================================
const int NDIM = 3;
struct Particle {
    double x[NDIM], v[NDIM];
    Particle() {}
    Particle(double * _x, double * _v = nullptr) {
        std::memcpy(x, _x, NDIM * sizeof(double));
        if (_v) {
            std::memcpy(v, _v, NDIM * sizeof(double));
        } else {
            for (int idim = 0; idim < NDIM; idim++)
                v[idim] = 0.0;
        }
    }
    double * get_pos() { return x; }
    double * get_vel() { return v; }
    int get_particle_byte_size() { return NDIM * sizeof(double); }
    void append_to_buffer(char * data) { std::memcpy(data, x, NDIM * sizeof(double)); }
    void assign_from_buffer(char * data) { std::memcpy(x, data, NDIM * sizeof(double)); }
};

void ExamplesInterpolation() {

    if (FML::ThisTask == 0)
        std::cout << "Running interpolate_grid_to_particle_positions\n";

    //=======================================================
    // Parameters below
    //=======================================================
    const int Nmesh = 128;
    const std::string interpolation_method = "CIC";
    const int npos = 100;

    //=======================================================
    // Function to fill the grid with
    //=======================================================
    std::function<FML::GRID::FloatType(std::array<double, NDIM> &)> func =
        [](std::array<double, NDIM> & pos) -> FML::GRID::FloatType {
        FML::GRID::FloatType value = 0.0;
        for (auto & x : pos) {
            if (x < 0.5)
                continue;
            value += std::sin(2.0 * M_PI * 2 * x);
        }
        return value;
    };

    //=======================================================
    // Density assignment method and the number of extra slices we need for this
    //=======================================================
    auto nleftright = FML::INTERPOLATION::get_extra_slices_needed_for_density_assignment(interpolation_method);

    //=======================================================
    // Make density grid (make sure we have enough slices)
    //=======================================================
    FFTWGrid<NDIM> grid(Nmesh, nleftright.first, nleftright.second);

    //=======================================================
    // Fill the grid from a function
    //=======================================================
    grid.fill_real_grid(func);

    //=======================================================
    // Make positions to look up
    //=======================================================
    std::vector<Particle> positions;
    for (int i = 0; i < npos; i++) {
        double pos[NDIM];
        for (int idim = 0; idim < NDIM; idim++) {
            pos[idim] = i / double(npos);
        }
        positions.push_back(Particle(pos));
    }

    //=======================================================
    // Make MPIParticles out of these positions
    //=======================================================
    FML::PARTICLE::MPIParticles<Particle> p;
    const bool all_tasks_have_the_same_particles = true;
    const int nalloc_per_task = 2 * npos;
    p.create(positions.data(),
             positions.size(),
             nalloc_per_task,
             FML::xmin_domain,
             FML::xmax_domain,
             all_tasks_have_the_same_particles);

    //=======================================================
    // Interpolate to the particle positions
    //=======================================================
    std::vector<double> interpolated_values;
    FML::INTERPOLATION::interpolate_grid_to_particle_positions<NDIM, Particle>(
        grid, p.get_particles().data(), p.get_npart(), interpolated_values, interpolation_method);

    //=======================================================
    // Output interpolation together with exact result
    // (Output here only outputs it all when NTasks=1)
    //=======================================================
    if (FML::ThisTask == 0) {
        auto & part = p.get_particles();
        std::cout << "Output [x interpol exact] to data.txt\n";
        std::ofstream fp("data.txt");
        for (size_t i = 0; i < p.get_npart(); i++) {
            std::array<double, NDIM> pos;
            double xyz = 0.0;
            for (int idim = 0; idim < NDIM; idim++) {
                pos[idim] = part[i].get_pos()[idim];
                xyz += pos[idim];
            }
            auto anal = func(pos);
            fp << xyz / 3.0 << " " << interpolated_values[i] << " " << anal << "\n";
        }
    }
}

int main() { ExamplesInterpolation(); }