#pragma once

#include <deal.II/base/table.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <stdexcept>
#include <string>

namespace data {
  using namespace dealii;
  using json = nlohmann::json;

  class GeometryData {
  public:

    struct BoundaryConditions {
      enum class BoundaryConditionType { Albedo, Dirichlet };
      public: 
        BoundaryConditions(BoundaryConditionType type, double param) : type(type), param(param) {};
        const BoundaryConditionType type;
        double param;
    };


    struct ExplicitPinsData {
      bool enabled = false;
      double radius = 0.0;
      unsigned int moderator_material = 0;
      unsigned int expected_refinements = 1;
    };


    GeometryData(const json &j) {
      const auto &geom = j.at("geometry");

      dimension = geom.at("dimension").get<unsigned int>();
      pin_pitch_x = geom.at("pin_pitch_x").get<double>();
      pin_pitch_y = geom.at("pin_pitch_y").get<double>();
      assembly_height = geom.at("assembly_height").get<double>();

      if (geom.contains("explicit_pins")) {
        const auto &ep = geom.at("explicit_pins");
        explicit_pins.enabled = ep.value("enabled", false);
        if (explicit_pins.enabled) {
          explicit_pins.radius = ep.at("radius").get<double>();
          explicit_pins.moderator_material = ep.at("moderator_material").get<unsigned int>();
          explicit_pins.expected_refinements = ep.value("expected_refinements", 1);
        }
      }

      parse_assemblies(geom.at("assemblies"));
      parse_core_map(geom.at("core_map"));
      parse_bc(geom.at("boundary_conditions"));
    }

    static GeometryData from_file(const std::string &filename) {
      std::ifstream file(filename);
      if (!file.is_open())
        throw std::runtime_error("GeometryData: cannot open file '" + filename + "'.");
      json j;
      file >> j;
      return GeometryData(j);
    }

    unsigned int get_core_n_assemblies_x() const { return core_n_assemblies_x; }
    unsigned int get_core_n_assemblies_y() const { return core_n_assemblies_y; }
    unsigned int get_core_n_assemblies_z() const { return core_n_assemblies_z; }

    unsigned int get_rods_per_assembly_x() const { return rods_per_assembly_x; }
    unsigned int get_rods_per_assembly_y() const { return rods_per_assembly_y; }
    
    double get_pin_pitch_x() const { return pin_pitch_x; }
    double get_pin_pitch_y() const { return pin_pitch_y; }

    double get_assembly_height() const { return assembly_height; }

    const ExplicitPinsData& get_explicit_pins_data() const { return explicit_pins; }

    unsigned int get_n_assemblies() const { return static_cast<unsigned int>(assembly_name_to_index.size()); }
 
    unsigned int get_assembly_index(const std::string &name) const {
      auto it = assembly_name_to_index.find(name);
      if (it == assembly_name_to_index.end())
        throw std::runtime_error("GeometryData: unknown assembly name '" + name + "'.");
      return it->second;
    }

    unsigned int get_assembly_pin(unsigned int assembly_idx, unsigned int row, unsigned int col) const {
      return assemblies[assembly_idx][row][col];
    }

    unsigned int get_assembly_pin(const std::string &assembly_name, unsigned int row, unsigned int col) const {
      return assemblies[get_assembly_index(assembly_name)][row][col];
    }

    unsigned int get_core_map(unsigned int row, unsigned int col, unsigned int z) const {
      return core_map[row][col][z];
    }
 
    BoundaryConditions get_boundary_condition(unsigned int side) const {
      return boundary_conditions[side];
    }


  private:
    unsigned int dimension;
    unsigned int core_n_assemblies_x;
    unsigned int core_n_assemblies_y;
    unsigned int core_n_assemblies_z;

    unsigned int rods_per_assembly_x;
    unsigned int rods_per_assembly_y;

    double pin_pitch_x;
    double pin_pitch_y;
    double assembly_height;

    ExplicitPinsData explicit_pins;

    Table<3, unsigned int> assemblies;
    Table<3, unsigned int> core_map;

    std::map<std::string, unsigned int> assembly_name_to_index;
    std::vector<BoundaryConditions> boundary_conditions;


    void parse_assemblies(const json &j) {
      // Start parsing assemblies
      if (j.empty())
        throw std::runtime_error("GeometryData: 'assemblies' object is empty.");

      rods_per_assembly_x = 0;
      rods_per_assembly_y = 0;
      for (const auto &[name, grid] : j.items()) {
        const unsigned int ny = static_cast<unsigned int>(grid.size());
        const unsigned int nx = static_cast<unsigned int>(grid[0].size());
        
        if (rods_per_assembly_x == 0 && rods_per_assembly_y == 0) {
          rods_per_assembly_x = nx;
          rods_per_assembly_y = ny;
        }
        else if (ny != rods_per_assembly_y || nx != rods_per_assembly_x) // Check if dimension of assemblies are consistent
          throw std::runtime_error("GeometryData: assembly '" + name + "' dimensions (" + std::to_string(ny) + "x" + std::to_string(nx) + ") differ from expected (" + std::to_string(rods_per_assembly_y) + "x" + std::to_string(rods_per_assembly_x) + ").");

        // Register the new assembly type
        assembly_name_to_index[name] = get_n_assemblies();
      }

      const unsigned int n_type_assemblies = get_n_assemblies();
      assemblies.reinit(n_type_assemblies, rods_per_assembly_y, rods_per_assembly_x);

      // Now we know the assembly types, iterate again and fill the tabel
      for (const auto &[name, grid] : j.items()) {
        const unsigned int idx = assembly_name_to_index[name];

        for (unsigned int row = 0; row < rods_per_assembly_y; ++row) {
          const auto &row_data = grid[row];
          if (row_data.size() != rods_per_assembly_x)
            throw std::runtime_error("GeometryData: assembly '" + name + "' row " + std::to_string(row) + " has " + std::to_string(row_data.size()) + " entries, expected " + std::to_string(rods_per_assembly_x) + ".");

          for (unsigned int col = 0; col < rods_per_assembly_x; ++col)
            assemblies[idx][row][col] = row_data[col].get<unsigned int>();
        }
      }
    }

    void parse_core_map(const json &j) {
      core_n_assemblies_y = static_cast<unsigned int>(j.size());
      if (core_n_assemblies_y == 0)
        throw std::runtime_error("GeometryData: 'core_map' is empty.");
      core_n_assemblies_x = static_cast<unsigned int>(j[0].size());
      if (core_n_assemblies_x == 0)
        throw std::runtime_error("GeometryData: 'core_map' is empty.");
      core_n_assemblies_z = static_cast<unsigned int>(j[0][0].size());;

      core_map.reinit(core_n_assemblies_y, core_n_assemblies_x, core_n_assemblies_z);
 
      // Fill core_map table
      for (unsigned int row = 0; row < core_n_assemblies_y; ++row) {
        const auto &row_data = j[row];
 
        if (row_data.size() != core_n_assemblies_x)
          throw std::runtime_error("GeometryData: core_map row " + std::to_string(row) + " has " + std::to_string(row_data.size()) + " entries, expected " + std::to_string(core_n_assemblies_x) + ".");
 
        for (unsigned int col = 0; col < core_n_assemblies_x; ++col) {
          const auto &col_data = row_data[col];
  
          if (col_data.size() != core_n_assemblies_z)
            throw std::runtime_error("GeometryData: core_map col " + std::to_string(col) + " has " + std::to_string(col_data.size()) + " entries, expected " + std::to_string(core_n_assemblies_z) + ".");
  
          for (unsigned int z = 0; z < core_n_assemblies_z; ++z)
            core_map[row][col][z] = get_assembly_index(col_data[z].get<std::string>());
        }
      }
    }

    void parse_bc(const json &j) {
      if (dimension*2 != static_cast<unsigned int>(j.size()))
        throw std::runtime_error("GeometryData: boundary conditions array dimension " + std::to_string(static_cast<unsigned int>(j.size())) + " , expected " + std::to_string(dimension * 2) + ".");
      
      boundary_conditions.reserve(dimension*2);
      for (unsigned int side = 0; side < dimension*2; ++side) {
        const std::string cond_str = j[side].at("type").get<std::string>();
        
        if (cond_str == "albedo")
          boundary_conditions.emplace_back(BoundaryConditions::BoundaryConditionType::Albedo, j[side].at("value").get<double>());
        else if (cond_str == "dirichlet")
          boundary_conditions.emplace_back(BoundaryConditions::BoundaryConditionType::Dirichlet, j[side].at("value").get<double>());
        else
          throw std::runtime_error("GeometryData: unknown boundary condition '" + cond_str + "' for side '" + std::to_string(side) + "'.");
      }
    }

  };

}