#pragma once

#include "MaterialData.hpp"

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/matrix_free/matrix_free.h>

#include <memory>
#include <vector>


namespace solver {

  template <typename number>
  struct MaterialCache {

    dealii::AlignedVector<dealii::VectorizedArray<number>> diffusion;
    dealii::AlignedVector<dealii::VectorizedArray<number>> sigma_rem;
    dealii::AlignedVector<dealii::VectorizedArray<number>> fission_distribution;
    dealii::AlignedVector<dealii::VectorizedArray<number>> nu_sigma_f;

    // Vector (each element is a different g_to) of AlignedVector<VectorizedArray>
    std::vector<dealii::AlignedVector<dealii::VectorizedArray<number>>> sigma_s;
    
    dealii::AlignedVector<dealii::VectorizedArray<number>> inner_face_diffusion_interior;
    dealii::AlignedVector<dealii::VectorizedArray<number>> inner_face_diffusion_exterior;
    dealii::AlignedVector<dealii::VectorizedArray<number>> inner_face_disc_fact_interior;
    dealii::AlignedVector<dealii::VectorizedArray<number>> inner_face_disc_fact_exterior;

  };


  template <unsigned int dim, typename number>
  class CrossSectionManager {
  public:
    void setup_levels(unsigned int n_levels, unsigned int n_groups);
    void update(const dealii::MatrixFree<dim, number> &, unsigned int, unsigned int, const data::MaterialData &);
    std::shared_ptr<const MaterialCache<number>> get_cache(unsigned int) const;

  private:
    std::vector<std::shared_ptr<MaterialCache<number>>> level_caches;

  };

}