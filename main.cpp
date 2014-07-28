#include "corpus.h"
#include "objcokus.h"
#include "tag.h"
#include "model.h"
#include "utils.h"
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  // parse args.
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("inference", po::value<string>(), "inference method (Gibbs, TreeUA)")
      ("T", po::value<int>(), "number of transitions")
      ("B", po::value<int>(), "number of burnin steps")
      ("Q", po::value<int>(), "number of passes")
      ("eps_split", po::value<int>(), "prob of split in MarkovTree")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  if(vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }
  string inference = "Gibbs";
  if(vm.count("inference")) {
    inference = vm["inference"].as<string>();
  }
  int T = 1;
  if(vm.count("T")) 
    T = vm["T"].as<int>();
  int B = 0;
  if(vm.count("B"))
    B = vm["B"].as<int>();
  int Q = 10;
  if(vm.count("Q"))
    Q = vm["Q"].as<int>();

  // run.
  Corpus corpus;
  corpus.read("data/eng_ner/train");
  Corpus testCorpus;
  testCorpus.read("data/eng_ner/test");
  auto set_param = [&] (shared_ptr<Model> model) {
    model->T = T;
    model->Q = Q;
    model->B = B;
  };
  if(inference == "Gibbs") {
    shared_ptr<Model> model = shared_ptr<Model>(new Model(corpus));
    set_param(model);
    model->run(testCorpus);
  }else if(inference == "TreeUA") {
    shared_ptr<ModelTreeUA> model = shared_ptr<ModelTreeUA>(new ModelTreeUA(corpus));
    if(vm.count("eps_split")) {
      model->eps_split = vm["eps_split"].as<int>();
    }
    set_param(model);
    model->runSimple(testCorpus);
    model->run(testCorpus);
  }else if(inference == "GibbsIncr") { 
    shared_ptr<Model> model = shared_ptr<Model>(new ModelIncrGibbs(corpus));
    set_param(model);
    model->run(testCorpus);
  }else if(inference == "Simple") {
    shared_ptr<Model> model = shared_ptr<Model>(new Model(corpus));
    set_param(model);
    model->runSimple(testCorpus);
  }
}
