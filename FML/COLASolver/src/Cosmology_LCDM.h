#ifndef COSMOLOGY_LCDM_HEADER
#define COSMOLOGY_LCDM_HEADER

#include <FML/FFTWGrid/FFTWGrid.h>
#include <FML/Global/Global.h>
#include <FML/LPT/DisplacementFields.h>
#include <FML/ODESolver/ODESolver.h>
#include <FML/ParameterMap/ParameterMap.h>
#include <FML/Spline/Spline.h>

#include "Cosmology.h"

class BackgroundCosmologyLCDM final : public BackgroundCosmology {
  public:
    BackgroundCosmologyLCDM() { name = "LCDM"; }

    double HoverH0_of_a(double a) const override {
        return std::sqrt(OmegaLambda + OmegaK / (a * a) + OmegaM / (a * a * a) + OmegaRtot / (a * a * a * a));
    }

    double dlogHdloga_of_a(double a) const override {
        double E = HoverH0_of_a(a);
        return 1.0 / (2.0 * E * E) *
               (-2.0 * OmegaK / (a * a) - 3.0 * OmegaM / (a * a * a) - 4.0 * OmegaRtot / (a * a * a * a));
    }

    void info() const override {
        BackgroundCosmology::info();
        if (FML::ThisTask == 0) {
            std::cout << "#=====================================================\n";
            std::cout << "\n";
        }
    }

  protected:
};

#endif
