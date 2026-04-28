#pragma once

#include <deal.II/base/table.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace data {
  using namespace dealii;
  using json = nlohmann::json;

  class MaterialData {
  public:

    MaterialData(const json &j) :
      n_groups(j.at("n_groups").get<unsigned int>()),
      n_materials(j.at("n_materials").get<unsigned int>()),
      diffusion(n_materials, n_groups),
      sigma_rem(n_materials, n_groups),
      nu_sigma_f(n_materials, n_groups),
      chi(n_materials, n_groups),
      discontinuity_factor(n_materials, n_groups),
      sigma_s(n_materials, n_groups, n_groups)
    {
      const auto &materials = j.at("materials");

      if (materials.size() != n_materials)
        throw std::runtime_error("MaterialData: 'n_materials' (" + std::to_string(n_materials) + ") does not match the actual number of materials in the array (" + std::to_string(materials.size()) + ").");

      // Load materials
      for (const auto &mat : materials) {
        const unsigned int m = mat.at("id").get<unsigned int>();

        if (m >= n_materials)
          throw std::out_of_range("MaterialData: material id " + std::to_string(m) + " is out of range for n_materials = " + std::to_string(n_materials));

        fill_1d(mat.at("diffusion"), m, diffusion);
        fill_1d(mat.at("sigma_rem"), m, sigma_rem);
        fill_1d(mat.at("nu_sigma_f"), m, nu_sigma_f);
        fill_1d(mat.at("chi"), m, chi);
        fill_1d(mat.at("discontinuity_factors"), m, discontinuity_factor);
        fill_2d(mat.at("sigma_s"), m, sigma_s);
      }
    }

    static MaterialData from_file(const std::string &filename) {
      std::ifstream file(filename);
      if (!file.is_open())
        throw std::runtime_error("MaterialData: cannot open file '" + filename + "'.");
      json j;
      file >> j;
      return MaterialData(j);
    }

    unsigned int get_n_groups() const { return n_groups; }
    unsigned int get_n_materials() const { return n_materials; }

    double get_diffusion(unsigned int m, unsigned int g) const { return diffusion[m][g]; }
    double get_sigma_rem(unsigned int m, unsigned int g) const { return sigma_rem[m][g]; }
    double get_nu_sigma_f(unsigned int m, unsigned int g) const { return nu_sigma_f[m][g]; }
    double get_chi(unsigned int m, unsigned int g) const { return chi[m][g]; }
    double get_discontinuity_factor(unsigned int m, unsigned int g) const { return discontinuity_factor[m][g]; }
    double get_sigma_s(unsigned int m, unsigned int g_from, unsigned int g_to) const { return sigma_s[m][g_from][g_to]; }
    const double * get_sigma_s(unsigned int m, unsigned int g_from) const { return &sigma_s[m][g_from][0]; }

    bool has_discontinuity_factors() const {
      for (unsigned int m = 0; m < n_materials; ++m)
        for (unsigned int g = 0; g < n_groups; ++g)
          if (std::abs(discontinuity_factor[m][g] - 1.0) > 1e-12)
            return true;
      return false;
    }

    
  private:
    const unsigned int n_groups;
    const unsigned int n_materials;

    Table<2, double> diffusion;
    Table<2, double> sigma_rem;
    Table<2, double> nu_sigma_f;
    Table<2, double> chi;
    Table<2, double> discontinuity_factor;
    Table<3, double> sigma_s;


    // Fill one row (material m) of a Table<2> from a flat JSON array [g0, g1, ...]
    void fill_1d(const json &arr, unsigned int m, Table<2, double> &table) const {
      if (arr.size() != n_groups)
        throw std::runtime_error("MaterialData: expected " + std::to_string(n_groups) + " groups, got " + std::to_string(arr.size()));

      for (unsigned int g = 0; g < n_groups; ++g)
        table[m][g] = arr[g].get<double>();
    }


    // Fill one material slice of a Table<3> from a 2-D JSON array [[...], [...]]
    void fill_2d(const json &arr, unsigned int m, Table<3, double> &table) const {
      if (arr.size() != n_groups)
        throw std::runtime_error("MaterialData: sigma_s outer dimension expected " + std::to_string(n_groups) + ", got " + std::to_string(arr.size()));

      for (unsigned int g_from = 0; g_from < n_groups; ++g_from) {
        const auto &row = arr[g_from];
        if (row.size() != n_groups)
          throw std::runtime_error("MaterialData: sigma_s inner dimension expected " + std::to_string(n_groups) + ", got " + std::to_string(row.size()));
        
        for (unsigned int g_to = 0; g_to < n_groups; ++g_to)
          table[m][g_from][g_to] = row[g_to].get<double>();
      }
    }

  };

}