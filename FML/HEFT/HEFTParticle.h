template <int NDIM>
struct HEFTParticle {

    /// Position
    double pos[NDIM];
    /// Lagrangian Position
    double q[NDIM];
    /// Velocity
    double vel[NDIM];
    /// Weight
    double mass{1.0};
    /// ID of the particle
    long long int id;

    /// Get the ID of the particle
    long long int get_id() const { return id; }
    /// Set the ID of the particle
    void set_id(long long int _id) { id = _id; }
    /// Get the dimension of the position
    constexpr int get_ndim() const { return NDIM; }
    /// Get a pointer to the position of the particle
    double * get_pos() { return pos; }
    /// Get a pointer to the velocity of the particle
    double * get_vel() { return vel; }
    /// Get a pointer to the Lagrangian position of the particle
    double * get_q() { return q; }
    /// Weight stuff of the particle
    double get_mass() const { return mass; }
    void set_mass(double _mass) { mass = _mass; }
};
