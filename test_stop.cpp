#include "corpus.h"
#include "objcokus.h"
#include "tag.h"
#include "model.h"
#include "utils.h"
#include "stop.h"
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  // parse args.
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("inference", po::value<string>()->default_value("Gibbs"), "inference method (Gibbs)")
      ("eta", po::value<double>()->default_value(1), "step size")
      ("T", po::value<int>()->default_value(10), "number of transitions")
      ("Tstar", po::value<double>()->default_value(1.5), "computational resource constraint")
      ("testCount", po::value<size_t>()->default_value(-1), "how many test data used ? default: all (-1). ")
      ("trainCount", po::value<size_t>()->default_value(-1), "how many training data used ? default: all (-1). ")
      ("B", po::value<int>()->default_value(3), "number of burnin steps")
      ("K", po::value<int>()->default_value(10), "number of threads/particles")
      ("c", po::value<double>()->default_value(0.1), "extent of time regularization")
      ("windowL", po::value<int>()->default_value(0), "window size for node-wise features")
      ("mode", po::value<string>()->default_value("POS"), "mode (POS / NER)")
      ("train", po::value<string>()->default_value("data/eng_ner/train"), "training data")
      ("test", po::value<string>()->default_value("data/eng_ner/test"), "test data")
      ("adaptive", po::value<bool>()->default_value(true), "use adaptive inference (stop) ? default: yes.")
      ("name", po::value<string>()->default_value("default"), "name of the run")
      ("numThreads", po::value<int>()->default_value(10), "number of threads to use")
      ("iter", po::value<size_t>()->default_value(2), "number of iterations for gradient")
  ;
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);    
  if(vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }
  Corpus::Mode mode = Corpus::MODE_POS;
  if(vm["mode"].as<string>() == "NER")
    mode = Corpus::MODE_NER;
  string train = vm["train"].as<string>(), test = vm["test"].as<string>();
  Corpus corpus(mode);
  corpus.read(train);
  Corpus testCorpus(mode);
  testCorpus.read(test);
  try{
    shared_ptr<Model> model = nullptr;
    if(vm["inference"].as<string>() == "Gibbs") {
      model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(&corpus, vm));
      auto file = openFile("model/gibbs.model");
      file >> *model;
      file.close();
      Stop stop(model, vm); 
      stop.run(corpus);
      stop.test(testCorpus);
    }else if(vm["inference"].as<string>() == "Simple") {
      shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(&corpus, vm));
      auto file = openFile("model/simple.model");
      file >> *model;
      file.close();
      Stop stop(model, vm); 
      stop.run(corpus);
      stop.test(testCorpus);
    }
  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
