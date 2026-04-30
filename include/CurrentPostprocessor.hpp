#pragma once

#include "MaterialData.hpp"
#include <deal.II/numerics/data_postprocessor.h>


namespace solver {

  template <unsigned int dim>
  class CurrentPostprocessor : public dealii::DataPostprocessorVector<dim> {
  public:

    CurrentPostprocessor(const unsigned int energy_group, const data::MaterialData &material_data) :
      dealii::DataPostprocessorVector<dim>("current", dealii::update_gradients),
      energy_group(energy_group),
      material_data(material_data)
    {}
  
    virtual void evaluate_scalar_field(const dealii::DataPostprocessorInputs::Scalar<dim> &input_data, std::vector<dealii::Vector<double>> &computed_quantities) const override {
      for (unsigned int q = 0; q < computed_quantities.size(); ++q) {  
        for (unsigned int d = 0; d < dim; ++d) {
          const auto cell = input_data.template get_cell<dim>();
          dealii::types::material_id mat_id = cell->material_id();
          computed_quantities[q][d] = -material_data.get_diffusion(mat_id, energy_group) * input_data.solution_gradients[q][d];
        }
      }
    }


  private:
    const unsigned int energy_group;
    const data::MaterialData &material_data;

  };

}
