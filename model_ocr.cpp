#include "corpus.h"
#include "objcokus.h"
#include "tag.h"
#include "model.h"
#include "utils.h"
#include "feature.h"
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
using namespace Tagging;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  // default arguments
  const int T = 10, B = 0, Q = 10, K = 5;
  const double eta = 0.4;

  // parse arguments from command line
  po::options_description desc("Allowed options");
  desc.add_options()
      ("help", "produce help message")
      ("windowL", po::value<int>()->default_value(0), "window size for node-wise features")
      ("depthL", po::value<int>()->default_value(0), "depth size for node-wise features")
      ("factorL", po::value<int>()->default_value(2), "up to what order of gram should be used")
      ("output", po::value<string>()->default_value("model/default.model"), "output model file")
      ("eta", po::value<double>()->default_value(eta), "step size")
      ("T", po::value<size_t>()->default_value(T), "number of transitions")
      ("B", po::value<size_t>()->default_value(B), "number of burnin steps")
      ("Q", po::value<size_t>()->default_value(Q), "number of passes")
      ("K", po::value<size_t>()->default_value(K), "number of parallel trajectories (default = 5)")
      ("scoring", po::value<string>()->default_value("Acc"), "scoring (Acc, NER)")
      ("train", po::value<string>(), "training data")
      ("test", po::value<string>(), "test data")
      ("testFrequency", po::value<double>()->default_value(0.3), "frequency of testing when making one pass of the training data")
  ;

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);
  if(vm.count("help")) {
      cout << desc << "\n";
      return 1;
  }

  try{
    // load corpus.
    string train = "data/ocr/train0", test = "data/ocr/test0";
    if(vm.count("train")) train = vm["train"].as<string>();
    if(vm.count("test")) test = vm["test"].as<string>();

    ptr<CorpusOCR<16,8> > corpus = ptr<CorpusOCR<16, 8> >(new CorpusOCR<16, 8>());
    corpus->read(train);

    ptr<CorpusOCR<16,8> > testCorpus = ptr<CorpusOCR<16, 8> >(new CorpusOCR<16, 8>());
    testCorpus->read(test);

    // run
    string output = vm["output"].as<string>();
    size_t pos = output.find_last_of("/");
    if(pos == string::npos) throw "invalid model output dir.";
    int sysres = system(("mkdir -p "+output.substr(0, pos)).c_str());

    // Gibbs sampling for inference
    shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(corpus, vm));

    cast<ModelCRFGibbs>(model)->extractFeatures = extractOCR;
    cast<ModelCRFGibbs>(model)->extractFeatAll = extractOCRAll;

    model->run(testCorpus);

    // output model
    ofstream file;
    file.open(vm["output"].as<string>());
    file << *model;
    file.close();

  }catch(char const* exception) {
    cerr << "Exception: " << string(exception) << endl;
  }

  return 0;
}
