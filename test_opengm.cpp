#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include "model_opengm.h"

using namespace std; // 'using' is used only in example code
using namespace opengm;
using namespace Tagging;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  po::options_description desc("Allowed options");
    desc.add_options()
  ("help", "produce help message")
  ("T", po::value<size_t>()->default_value(4), "number of sweeps in Gibbs sampling")
  ("eta", po::value<double>()->default_value(1), "step-size for policy gradient (adagrad)")
  ("B", po::value<size_t>()->default_value(0), "number of burnin steps")
  ("Q", po::value<size_t>()->default_value(1), "number of passes")
  ("Q0", po::value<int>()->default_value(1), "number of passes for smart init")
  ("testFrequency", po::value<double>()->default_value(0.3), "frequency of testing")
  ("K", po::value<size_t>()->default_value(5), "number of samples in policy gradient")
  ("temp", po::value<string>()->default_value("scanline"), "the annealing scheme to use.")
  ("temp_decay", po::value<double>()->default_value(0.9), "decay of temperature.")
  ("temp_magnify", po::value<double>()->default_value(0.1), "magnifying factor of init temperature.");
  
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  if(vm.count("help")) {
    cout << desc << "\n";
    return 1;
  }

  // build a graphical model (other examples have more details)
  typedef SimpleDiscreteSpace<size_t, size_t> Space;
  typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> ,PottsFunction<double>), Space> GraphicalModelType;
  GraphicalModelType instance;
  
  // load the graphical model from the hdf5 dataset
  opengm::hdf5::load(instance, "../data/opengm/inpainting-n/inpainting-n4/triplepoint4-plain-ring.h5","gm");
  
  OpenGM<GraphicalModelType> sample(instance);
  ModelEnumerativeGibbs<GraphicalModelType, opengm::Minimizer> model(vm);
  objcokus rng;
  rng.seedMT(0);

  // optimize by Gibbs
  const size_t num_iteration = 1000;

  for(size_t it = 0; it < num_iteration; it++) {
    for(size_t j = 0; j < sample.size(); j++) {
      model.sampleOne(sample, rng, j);
    }
    cout << "iter = " << it << " , " << "score = " << model.score(sample) << endl;
  }

  return 0;
}
