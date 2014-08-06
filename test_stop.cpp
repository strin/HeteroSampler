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
      ("B", po::value<int>()->default_value(3), "number of burnin steps")
      ("K", po::value<int>()->default_value(10), "number of threads/particles")
      ("c", po::value<double>(), "extent of time regularization")
      ("windowL", po::value<int>()->default_value(0), "window size for node-wise features")
      ("mode", po::value<string>()->default_value("POS"), "mode (POS / NER)")
      ("train", po::value<string>(), "training data")
      ("test", po::value<string>(), "test data")
      ("name", po::value<string>()->default_value("name"), "name of the run")
      ("numThreads", po::value<int>()->default_value(10), "number of threads to use")
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
  string train = "data/eng_ner/train", test = "data/eng_ner/test";
  Corpus corpus(mode);
  corpus.read(train);
  Corpus testCorpus(mode);
  testCorpus.read(test);
  try{
    if(vm["inference"].as<string>() == "Gibbs") {
      shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(&corpus, vm["windowL"].as<int>()));
      ifstream file; 
      file.open("model/gibbs.model", ifstream::in);
      if(!file.is_open()) 
	throw "gibbs model not found.";
      file >> *model;
      file.close();
      Stop stop(model, vm); 
      stop.run(*model->corpus);
    }else if(vm["inference"].as<string>() == "Simple") {
      shared_ptr<Model> model = shared_ptr<ModelSimple>(new ModelSimple(&corpus, vm["windowL"].as<int>()));
      ifstream file; 
      file.open("model/simple.model", ifstream::in);
      if(!file.is_open()) 
	throw "gibbs model not found.";
      file >> *model;
      file.close();
      Stop stop(model, vm); 
      stop.run(*model->corpus);
    }
  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
