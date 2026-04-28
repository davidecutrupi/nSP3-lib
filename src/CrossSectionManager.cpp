#include "CrossSectionManager.hpp"

#include <deal.II/base/types.h>


namespace solver {
  using namespace dealii;

  
  template <unsigned int dim, typename number>
  void CrossSectionManager<dim, number>::setup_levels(unsigned int n_levels, unsigned int n_groups) {
    level_caches.resize(n_levels);
    for (unsigned int level = 0; level < n_levels; ++level) {
      level_caches[level] = std::make_shared<MaterialCache<number>>();
      level_caches[level]->sigma_s.resize(n_groups);
    }
  }


  template <unsigned int dim, typename number>
  std::shared_ptr<const MaterialCache<number>> CrossSectionManager<dim, number>::get_cache(unsigned int level) const {
    return level_caches[level];
  }


  template <unsigned int dim, typename number>
  void CrossSectionManager<dim, number>::update(const MatrixFree<dim, number> &mf, unsigned int level, unsigned int active_group, const data::MaterialData& material_data) {
    const unsigned int n_groups = material_data.get_n_groups();
    auto &cache = *level_caches[level];

    const unsigned int n_cell_batches = mf.n_cell_batches();
    const unsigned int n_inner_faces = mf.n_inner_face_batches();

    // 1. Resize di tutti gli AlignedVector
    cache.diffusion.resize(n_cell_batches);
    cache.sigma_rem.resize(n_cell_batches);
    cache.fission_distribution.resize(n_cell_batches);
    cache.nu_sigma_f.resize(n_cell_batches);
    for (unsigned int g = 0; g < n_groups; ++g) {
      cache.sigma_s[g].resize(n_cell_batches);
    }

    cache.inner_face_diffusion_interior.resize(n_inner_faces);
    cache.inner_face_diffusion_exterior.resize(n_inner_faces);
    cache.inner_face_disc_fact_interior.resize(n_inner_faces);
    cache.inner_face_disc_fact_exterior.resize(n_inner_faces);

    // Fill cells aligned vectors
    for (unsigned int batch = 0; batch < n_cell_batches; ++batch) {
      VectorizedArray<number> v_diff, v_sigrem, v_chi, v_nusigf;
      std::vector<VectorizedArray<number>> v_sigs(n_groups);

      const unsigned int n_active_lanes = mf.n_active_entries_per_cell_batch(batch);
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v) {
        if (v < n_active_lanes) {
          auto cell = mf.get_cell_iterator(batch, v);
          types::material_id mat_id = cell->material_id();

          v_diff[v] = number(material_data.get_diffusion(mat_id, active_group));
          v_sigrem[v] = number(material_data.get_sigma_rem(mat_id, active_group));
          v_chi[v] = number(material_data.get_chi(mat_id, active_group));
          v_nusigf[v] = number(material_data.get_nu_sigma_f(mat_id, active_group));

          // Take scattering XS to all other groups (double *) and fill the vector
          const double *sigma_s_row = material_data.get_sigma_s(mat_id, active_group);
          for (unsigned int g_to = 0; g_to < n_groups; ++g_to) {
            v_sigs[g_to][v] = number(sigma_s_row[g_to]);
          }
        }
        else { // Inactive lanes
          v_diff[v] = 0.0;
          v_sigrem[v] = 0.0;
          v_chi[v] = 0.0;
          v_nusigf[v] = 0.0;
          for (unsigned int g_to = 0; g_to < n_groups; ++g_to) v_sigs[g_to][v] = 0.0;
        }
      }

      cache.diffusion[batch] = v_diff;
      cache.sigma_rem[batch] = v_sigrem;
      cache.fission_distribution[batch] = v_chi;
      cache.nu_sigma_f[batch] = v_nusigf;
      for (unsigned int g_to = 0; g_to < n_groups; ++g_to) cache.sigma_s[g_to][batch] = v_sigs[g_to];
    }

    // Fill faces aligned vectors
    for (unsigned int batch = 0; batch < n_inner_faces; ++batch) {
      VectorizedArray<number> v_diff_in, v_diff_ex, v_df_in, v_df_ex;
      
      const unsigned int n_active_lanes = mf.n_active_entries_per_face_batch(batch);
      for (unsigned int v = 0; v < VectorizedArray<number>::size(); ++v) {
        if (v < n_active_lanes) {
          // Interior side
          auto pair_in = mf.get_face_iterator(batch, v, true); 
          types::material_id mat_id_in = pair_in.first->material_id();

          v_diff_in[v] = number(material_data.get_diffusion(mat_id_in, active_group));
          v_df_in[v] = number(material_data.get_discontinuity_factor(mat_id_in, active_group)); 

          // Exterior side
          auto pair_ex = mf.get_face_iterator(batch, v, false);
          types::material_id mat_id_ex = pair_ex.first->material_id();

          v_diff_ex[v] = number(material_data.get_diffusion(mat_id_ex, active_group));
          v_df_ex[v] = number(material_data.get_discontinuity_factor(mat_id_ex, active_group));
        }
        else { // Inactive lanes
          v_diff_in[v] = 0.0;
          v_diff_ex[v] = 0.0;
          v_df_in[v] = 1.0;
          v_df_ex[v] = 1.0; 
        }
      }
      
      cache.inner_face_diffusion_interior[batch] = v_diff_in;
      cache.inner_face_diffusion_exterior[batch] = v_diff_ex;
      cache.inner_face_disc_fact_interior[batch] = v_df_in;
      cache.inner_face_disc_fact_exterior[batch] = v_df_ex;
    }
  }

}



template class solver::CrossSectionManager<3u, double>;
template class solver::CrossSectionManager<3u, float>;

template class solver::CrossSectionManager<2u, double>;
template class solver::CrossSectionManager<2u, float>;