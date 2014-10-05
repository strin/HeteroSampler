#include "corpus.h"
#include "corpus_ising.h"
#include "objcokus.h"
#include "tag.h"
#include "feature.h"
#include "model.h"
#include "model_opengm.h"
#include "utils.h"
#include "policy.h"
#include "blockpolicy.h"
#include "opengm.h"

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>


using namespace std;
using namespace Tagging;
using namespace opengm;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  try{
    /* load options */
    po::variables_map vm;
    vm.insert(std::make_pair("name", po::variable_value(string("check_opengm_chain"), false)));
    vm.insert(std::make_pair("scoring", po::variable_value(string("Lhood"), false)));
    vm.insert(std::make_pair("numThreads", po::variable_value((size_t)10, false)));
    vm.insert(std::make_pair("testCount", po::variable_value((size_t)100, false)));
    vm.insert(std::make_pair("trainCount", po::variable_value((size_t)100, false)));
    vm.insert(std::make_pair("windowL", po::variable_value((int)0, false)));
    vm.insert(std::make_pair("depthL", po::variable_value((int)0, false)));
    vm.insert(std::make_pair("factorL", po::variable_value((int)2, false)));
    vm.insert(std::make_pair("testFrequency", po::variable_value((double)0.3, false)));
    vm.insert(std::make_pair("verbose", po::variable_value((bool)true, false)));
    vm.insert(std::make_pair("temp", po::variable_value(string("scanline"), false)));
    vm.insert(std::make_pair("temp_init", po::variable_value((double)1, false)));
    vm.insert(std::make_pair("temp_decay", po::variable_value((double)0.9, false)));
    vm.insert(std::make_pair("temp_magnify", po::variable_value((double)0.1, false)));
    vm.insert(std::make_pair("feat", po::variable_value(string("bias cond-ent nb-ent"), false)));
    vm.insert(std::make_pair("verbosity", po::variable_value(string(""), false)));
    vm.insert(std::make_pair("init", po::variable_value(string("random"), false)));
    vm.insert(std::make_pair("lets_lazymax", po::variable_value((bool)true, false)));
    vm.insert(std::make_pair("inplace", po::variable_value((bool)true, false)));
    vm.insert(std::make_pair("eta", po::variable_value((double)1, false)));
    vm.insert(std::make_pair("c", po::variable_value((double)1, false)));
    vm.insert(std::make_pair("T", po::variable_value((size_t)2, true)));
    vm.insert(std::make_pair("K", po::variable_value((size_t)1, false)));
    vm.insert(std::make_pair("Q", po::variable_value((size_t)1, false)));
    vm.insert(std::make_pair("Q0", po::variable_value((int)1, false)));
    vm.insert(std::make_pair("B", po::variable_value((size_t)0, false)));
    po::notify(vm);    

    cout << vm["scoring"].as<string>() << endl;
    /* specify the model */
    typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
    typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> ,PottsFunction<double>), Space> GraphicalModelType;
    typedef CorpusOpenGM<GraphicalModelType> CorpusOpenGMType;
    auto corpus = std::make_shared<CorpusOpenGMType>();

    const size_t numLabels = 2;
    const size_t numVars = 4;
    Space space(numVars, numLabels);
    ptr<GraphicalModelType> instance = std::make_shared<GraphicalModelType>(space);
    const size_t shape[] = {numLabels};
    ExplicitFunction<double> u(shape, shape + 1), u0(shape, shape+1);
    auto u_id = instance->addFunction(u), u0_id = instance->addFunction(u0);
    u(0) = 10;
    u(1) = 10;
    u0(0) = 20;
    u0(1) = 0;
    PottsFunction<double> f(numLabels, numLabels, 10, 0);
    auto f_id = instance->addFunction(f);
    for(size_t id = 1; id < numVars; id++) {
      size_t vars123[] = {id};
      instance->addFactor(u_id, vars123, vars123+1);
    }
    size_t vars0[] = {0};
    instance->addFactor(u0_id, vars0, vars0+1);
    for(size_t id = 0; id < numVars-1; id++) {
      size_t vars_pair[] = {id, id+1};
      instance->addFactor(f_id, vars_pair, vars_pair+2);
    }
    corpus->seqs.push_back(ptr<InstanceOpenGM<GraphicalModelType> >(new InstanceOpenGM<GraphicalModelType>(corpus.get(), instance)));


    shared_ptr<Model> model = std::make_shared<ModelEnumerativeGibbs<GraphicalModelType, opengm::Minimizer> >(vm);

    

    string name = vm["name"].as<string>();
    shared_ptr<Policy> policy = std::make_shared<BlockPolicy<LockdownPolicy> >(model, vm);
    policy->train(corpus);

  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
