#pragma once

#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>
#include <memory>

class GlobalTimer {
public:
  static void init(MPI_Comm mpi_communicator, dealii::ConditionalOStream &pcout) {
    if (!timer) {
      timer = std::make_unique<dealii::TimerOutput>(mpi_communicator, pcout, dealii::TimerOutput::summary, dealii::TimerOutput::wall_times);
    }
  }

  static dealii::TimerOutput& get() {
    Assert(timer, dealii::ExcMessage("GlobalTimer non initialized!"));
    return *timer;
  }

  static void clear() {
    timer.reset();
  }

private:
  inline static std::unique_ptr<dealii::TimerOutput> timer = nullptr;

};