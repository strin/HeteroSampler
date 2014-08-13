#include "corpus.h"
#include "objcokus.h"
#include "tag.h"
#include "model.h"
#include "utils.h"
#include "policy.h"
#include <iostream>
#include <boost/program_options.hpp>

using namespace std;
namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  try{
    // parse args.
    po::options_description desc("Allowed options");
    desc.add_options()
	("help", "produce help message")
	("inference", po::value<string>()->default_value("Gibbs"), "inference method (Gibbs)")
	("model", po::value<string>()->default_value("model/gibbs.model"), "use saved model to do the inference")
	("policy", po::value<string>()->default_value("entropy"), "sampling policy")
	("name", po::value<string>()->default_value("default"), "name of the run")
	("train", po::value<string>()->default_value("data/eng_ner/train"), "training data")
	("test", po::value<string>()->default_value("data/eng_ner/test"), "test data")
	("numThreads", po::value<size_t>()->default_value(10), "number of threads to use")
	("threshold", po::value<double>()->default_value(0.8), "theshold for entropy policy")
	("T", po::value<size_t>()->default_value(1), "number of sweeps in Gibbs sampling")
	("K", po::value<size_t>()->default_value(5), "number of samples in policy gradient")
	("eta", po::value<double>()->default_value(1), "step-size for policy gradient (adagrad)")
	("c", po::value<double>()->default_value(0.1), "time regularization")
	("windowL", po::value<int>()->default_value(0), "window size for node-wise features")
	("testCount", po::value<size_t>()->default_value(-1), "how many test data used ? default: all (-1). ")
	("trainCount", po::value<size_t>()->default_value(-1), "how many training data used ? default: all (-1). ")
	("mode", po::value<string>()->default_value("POS"), "mode (POS / NER)")
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
    corpus.read(train, false);
    Corpus testCorpus(mode);
    testCorpus.read(test, false);
    shared_ptr<Model> model;
    if(vm["inference"].as<string>() == "Gibbs") {
      model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(&corpus, vm));
      auto file = openFile(vm["model"].as<string>());
      file >> *model;
      file.close();
    }
    shared_ptr<Policy> policy;
    if(vm["policy"].as<string>() == "entropy") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<EntropyPolicy>(new EntropyPolicy(model, vm)));   
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "gibbs") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<GibbsPolicy>(new GibbsPolicy(model, vm)));   
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "cyclic") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicPolicy(model, vm)));
      policy->train(corpus);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "cyclic_value") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicValuePolicy(model, vm)));
      policy->train(corpus);
      policy->test(testCorpus);
    }
  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
