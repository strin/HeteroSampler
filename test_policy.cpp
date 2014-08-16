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
	("Tstar", po::value<double>()->default_value(1.5), "computational resource constraint (used to compute c)")
	("B", po::value<size_t>()->default_value(0), "number of burnin steps")
	("Q", po::value<size_t>()->default_value(10), "number of passes")
	("Q0", po::value<int>()->default_value(1), "number of passes for smart init")
	("K", po::value<size_t>()->default_value(5), "number of samples in policy gradient")
	("eta", po::value<double>()->default_value(1), "step-size for policy gradient (adagrad)")
	("c", po::value<double>()->default_value(0.1), "time regularization")
	("windowL", po::value<int>()->default_value(0), "window size for node-wise features")
	("depthL", po::value<int>()->default_value(0), "depth size for node-wise features")
	("factorL", po::value<int>()->default_value(2), "up to what order of gram should be used")
	("testCount", po::value<size_t>()->default_value(-1), "how many test data used ? default: all (-1). ")
	("trainCount", po::value<size_t>()->default_value(-1), "how many training data used ? default: all (-1). ")
	("scoring", po::value<string>()->default_value("Acc"), "scoring (Acc, NER)")
	("testFrequency", po::value<double>()->default_value(0.3), "frequency of testing")
	("verbose", po::value<bool>()->default_value(false), "whether to output more debug information")
    ;
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
    if(vm.count("help")) {
	cout << desc << "\n";
	return 1;
    }
    string train = vm["train"].as<string>(), test = vm["test"].as<string>();
    Corpus corpus;
    corpus.read(train, false);
    Corpus testCorpus;
    testCorpus.read(test, false);
    shared_ptr<Model> model;
    if(vm["inference"].as<string>() == "Gibbs") {
      model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(&corpus, vm));
      std::ifstream file; 
      file.open(vm["model"].as<string>(), std::fstream::in);
      if(!file.is_open()) 
	throw (vm["model"].as<string>()+" not found.").c_str();
      file >> *model;
      file.close();
    }
    corpus.computeWordFeat();
    shared_ptr<Policy> policy;
    if(vm["policy"].as<string>() == "entropy") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<EntropyPolicy>(new EntropyPolicy(model, vm)));   
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "gibbs") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<GibbsPolicy>(new GibbsPolicy(model, vm)));   
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "gibbs_shared") {
      Policy::ResultPtr result = nullptr;
      string name = vm["name"].as<string>();
      const size_t T = vm["T"].as<size_t>();
      vector<shared_ptr<GibbsPolicy> > gibbs_policy(T);
      for(size_t t = 1; t <= T; t++) {
	string myname = name+"_T"+to_string(t);
	gibbs_policy[t-1] = shared_ptr<GibbsPolicy>(new GibbsPolicy(model, vm));
	gibbs_policy[t-1]->T = 1;  // do one sweep after another.
	system(("mkdir -p "+myname).c_str());
	gibbs_policy[t-1]->resetLog(shared_ptr<XMLlog>(new XMLlog(myname+"/policy.xml")));  
	if(t == 1) {
	  result = gibbs_policy[t-1]->test(testCorpus);
	}else{
	  gibbs_policy[t-1]->test(result);
	}
	gibbs_policy[t-1]->resetLog(nullptr);
      }
      system(("rm -r "+name).c_str());
    }else if(vm["policy"].as<string>() == "cyclic") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicPolicy(model, vm)));
      policy->train(corpus);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "cyclic_value") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicValuePolicy(model, vm)));
      policy->train(corpus);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "cyclic_value_shared") {
      string name = vm["name"].as<string>();
      const int fold = 10;
      shared_ptr<CyclicValuePolicy> policy = shared_ptr<CyclicValuePolicy>(new CyclicValuePolicy(model, vm));
      policy->lets_resp_reward = true;
      policy->train(corpus);
      vector<shared_ptr<CyclicValuePolicy> > test_policy;
      auto compare = [] (pair<double, double> a, pair<double, double> b) {
	return (a.first < b.first);
      };
      sort(policy->resp_reward.begin(), policy->resp_reward.end(), compare); 
      for(int i = 0; i <= fold; i++) {
	double c = policy->resp_reward[i * (policy->resp_reward.size()-1)/(double)fold].first;
	string myname = name+"_i"+to_string(i);
	system(("mkdir -p " + myname).c_str());
	shared_ptr<CyclicValuePolicy> ptest = shared_ptr<CyclicValuePolicy>(new CyclicValuePolicy(model, vm));
	test_policy.push_back(ptest);
	ptest->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
	ptest->param = policy->param; 
	ptest->c = c;
	ptest->test(testCorpus);
	ptest->resetLog(nullptr);
      }
      system(("rm -r "+name).c_str());
    }
  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
