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
    // parse args.
    po::options_description desc("Allowed options");
    desc.add_options()
  	("help", "produce help message")
  	("inference", po::value<string>()->default_value("Gibbs"), "inference method (Gibbs)")
  	("dataset", po::value<string>()->default_value("literal"), "type of dataset to use (literal, ocr)")
  	("model", po::value<string>()->default_value("model/gibbs.model"), "use saved model to do the inference")
  	("unigram_model", po::value<string>(), "use a unigram (if necessary)")
  	("policy", po::value<string>()->default_value("entropy"), "sampling policy")
  	("name", po::value<string>()->default_value("default"), "name of the run")
  	("train", po::value<string>()->default_value("data/eng_ner/train"), "training data")
  	("test", po::value<string>()->default_value("data/eng_ner/test"), "test data")
  	("numThreads", po::value<size_t>()->default_value(10), "number of threads to use")
  	("threshold", po::value<double>()->default_value(0.8), "theshold for entropy policy")
  	("T", po::value<size_t>()->default_value(4), "number of sweeps in Gibbs sampling")
  	("Tstar", po::value<double>()->default_value(1.5), "computational resource constraint (used to compute c)")
  	("B", po::value<size_t>()->default_value(0), "number of burnin steps")
  	("Q", po::value<size_t>()->default_value(1), "number of passes")
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
  	("lets_model", po::value<bool>()->default_value(false), "whether to update model during policy learning (default: false)")
  	("lets_notrain", po::value<bool>()->default_value(false), "do not train the policy")
    ("inplace", po::value<bool>()->default_value(false), "inplace is true, then the algorithm do not work with entire hisotry")
    ("init", po::value<string>()->default_value("random"), "initialization method: random, iid, unigram.")
    ("verbosity", po::value<string>()->default_value(""), "what kind of information to log? ")
    ("feat", po::value<std::string>()->default_value(""), "feature switches")
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
    string train = vm["train"].as<string>(), test = vm["test"].as<string>();
    string dataset = vm["dataset"].as<string>();
    ptr<Corpus> corpus, testCorpus;
    if(dataset == "literal") {
      corpus = ptr<CorpusLiteral>(new CorpusLiteral());
      testCorpus = ptr<CorpusLiteral>(new CorpusLiteral());
      cast<CorpusLiteral>(corpus)->computeWordFeat();
    }else if(dataset == "ocr") {
      corpus = std::make_shared<CorpusOCR<16, 8> >();
      testCorpus = std::make_shared<CorpusOCR<16, 8> >();
    }else if(dataset == "ising") {
      corpus = std::make_shared<CorpusIsing>();
      testCorpus = std::make_shared<CorpusIsing>();
    }else if(dataset == "opengm") {
      typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
      typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> ,PottsFunction<double>), Space> GraphicalModelType;
      typedef CorpusOpenGM<GraphicalModelType> CorpusOpenGMType;
      corpus = std::make_shared<CorpusOpenGMType>();
      testCorpus = std::make_shared<CorpusOpenGMType>();
    }
    corpus->read(train, false);
    testCorpus->read(test, false);

    shared_ptr<Model> model, model_unigram;
    if(vm["inference"].as<string>() == "Gibbs") {
      auto loadGibbsModel = [&] (string name) -> ModelPtr {
        shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(corpus, vm));
        std::ifstream file; 
        file.open(name, std::fstream::in);
        if(!file.is_open()) 
          throw (name+" not found.").c_str();
        file >> *model;
        file.close();
        // extract features based on application.
        if(dataset == "ocr") {
          cast<ModelCRFGibbs>(model)->extractFeatures = extractOCR;
          cast<ModelCRFGibbs>(model)->extractFeatAll = extractOCRAll; 
        }else if(dataset == "ising") {
          cast<ModelCRFGibbs>(model)->extractFeatures = extractIsing;
          cast<ModelCRFGibbs>(model)->extractFeatAll = extractIsingAll;
          cast<ModelCRFGibbs>(model)->extractFeaturesAtInit = extractIsingAtInit;
        }
        return model;
      };
      model = loadGibbsModel(vm["model"].as<string>());
      if(vm.count("unigram_model")) {
        model_unigram = loadGibbsModel(vm["unigram_model"].as<string>());
      }
    }else if(vm["inference"].as<string>() == "OpenGibbs") {
      typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
      typedef opengm::GraphicalModel<double, opengm::Adder, OPENGM_TYPELIST_2(ExplicitFunction<double> ,PottsFunction<double>), Space> GraphicalModelType;
      typedef CorpusOpenGM<GraphicalModelType> CorpusOpenGMType;
      model = std::make_shared<ModelEnumerativeGibbs<GraphicalModelType, opengm::Minimizer> >(vm);
    }

    shared_ptr<Policy> policy;
    bool lets_model = vm["lets_model"].as<bool>();
    auto train_func = [&] (shared_ptr<Policy> policy) {
      if(lets_model) policy->train(corpus);
      else policy->trainPolicy(corpus);
    };
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
      shared_ptr<GibbsPolicy> gibbs_policy;
      gibbs_policy = shared_ptr<GibbsPolicy>(new GibbsPolicy(model, vm));
      gibbs_policy->T = 1;  // do one sweep after another.
      for(size_t t = 1; t <= T; t++) {
      	string myname = name+"_T"+to_string(t);
      	system(("mkdir -p "+myname).c_str());
      	gibbs_policy->resetLog(shared_ptr<XMLlog>(new XMLlog(myname+"/policy.xml")));  
      	if(t == 1) {
      	  result = gibbs_policy->test(testCorpus);
      	}else{
          gibbs_policy->init_method = "";
      	  gibbs_policy->test(result);
      	}
      	gibbs_policy->resetLog(nullptr);
      }
      system(("rm -r "+name).c_str());
    }else if(vm["policy"].as<string>() == "cyclic") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicPolicy(model, vm)));
      train_func(policy);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "cyclic_value") {
      policy = dynamic_pointer_cast<Policy>(shared_ptr<CyclicPolicy>(new CyclicValuePolicy(model, vm)));
      train_func(policy);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "multi_cyclic_value") {
      policy = shared_ptr<Policy>(new MultiCyclicValuePolicy(model, vm));
      train_func(policy);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "lockdown") {
      policy = shared_ptr<Policy>(new LockdownPolicy(model, vm));
      train_func(policy);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "lockdown_shared") {
      string name = vm["name"].as<string>();
      const int fold = 20;
//      const int fold_l[fold] = {0,5,10,15,20,25,26,27,28,29};
      system(("rm -r "+name+"*").c_str());
      policy = shared_ptr<Policy>(new LockdownPolicy(model, vm));
      policy->lets_resp_reward = true;
      policy->model_unigram = model_unigram;
      system(("mkdir -p " + name + "_train").c_str());
      policy->resetLog(shared_ptr<XMLlog>(new XMLlog(name + "_train" + "/policy.xml")));
      train_func(policy);
      cast<LockdownPolicy>(policy)->c = -DBL_MAX; // sample every position.
      policy->test(testCorpus);
      policy->resetLog(nullptr);
      auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
        return (a.first < b.first);
      };
      auto compare2 = [] (std::pair<double, double> a, std::pair<double, double> b) {
        return (a.second < b.second);
      };
      sort(policy->test_resp_reward.begin(), policy->test_resp_reward.end(), compare);
      double c_max = std::max_element(policy->test_resp_reward.begin(),
                                            policy->test_resp_reward.end(), compare)->first,
                    c_min = std::min_element(policy->test_resp_reward.begin(),
                                            policy->test_resp_reward.end(), compare)->first;
      std::vector<std::pair<double, double> > acc_c;
      auto runWithC = [&] (double m_c) {
        string myname = name + "_c" + boost::lexical_cast<string>(m_c);
        system(("mkdir -p " + myname).c_str());
        shared_ptr<LockdownPolicy> ptest = std::make_shared<LockdownPolicy>(model, vm);
        ptest->model_unigram = model_unigram;
        ptest->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
        ptest->param = policy->param;
        ptest->c = m_c;
        Policy::ResultPtr result = ptest->test(testCorpus);
        ptest->resetLog(nullptr);
        acc_c.push_back(make_pair(result->score, m_c));
        sort(acc_c.rbegin(), acc_c.rend(), compare2);
      };
      auto findLargestSeg = [&] () -> double {
        double segmax = -DBL_MAX, c = 0.0;
        for(size_t t = 0; t < acc_c.size()-1; t++) {
          double seg = acc_c[t+1].first - acc_c[t].first;
          if(seg > segmax) {
            segmax = seg;
            c = (acc_c[t+1].second + acc_c[t].second) / 2.0;
          }
        }
        return c;
      };
      runWithC(c_min);
      runWithC(c_max);
      for(int i = 0; i < fold; i++) {
        double c = findLargestSeg();
        runWithC(c);
      }
//      for(int i : fold_l) {
//      	double c = policy->test_resp_reward[i * (policy->test_resp_reward.size()-1)/(double)fold_l[fold-1]].first;
//      	// string myname = name+"_i"+to_string(i);
//      }
    }else if(vm["policy"].as<string>() == "random_scan") {
      policy = shared_ptr<Policy>(new RandomScanPolicy(model, vm));
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "adaptive_random_scan") {
      policy = shared_ptr<Policy>(new RandomScanPolicy(model, vm));
      shared_ptr<Policy>(new RandomScanPolicy(model, vm));
      train_func(policy);
      policy->test(testCorpus);
    }else if(vm["policy"].as<string>() == "multi_cyclic_value_shared") {
      string name = vm["name"].as<string>();
      const int fold = 10;
      const int fold_l[fold] = {0,5,10,15,20,25,26,27,28,29};
      system(("rm -r "+name+"*").c_str());
      shared_ptr<CyclicValuePolicy> policy = shared_ptr<CyclicValuePolicy>(new MultiCyclicValuePolicy(model, vm));
      cast<CyclicValuePolicy>(policy)->c = -DBL_MAX; // sample every position.
      policy->lets_resp_reward = true;
      system(("mkdir -p " + name + "_train").c_str());
      policy->resetLog(shared_ptr<XMLlog>(new XMLlog(name + "_train" + "/policy.xml")));
      train_func(policy);
      if(vm["lets_notrain"].as<bool>()) mapReset(*policy->param, 1);
      policy->test(testCorpus);
      policy->resetLog(nullptr);
      shared_ptr<MultiCyclicValuePolicy> ptest;
      auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
	return (a.first < b.first);
      };
      sort(policy->test_resp_reward.begin(), policy->test_resp_reward.end(), compare); 
      for(int i : fold_l) {
        double c = policy->test_resp_reward[i * (policy->test_resp_reward.size()-1)/(double)fold_l[fold-1]].first;
        // string myname = name+"_i"+to_string(i);
        string myname = name + "_c" + boost::lexical_cast<string>(c);
        system(("mkdir -p " + myname).c_str());
        ptest = shared_ptr<MultiCyclicValuePolicy>(new MultiCyclicValuePolicy(model, vm));
        ptest->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
        ptest->param = policy->param; 
        ptest->c = c;
        ptest->test(testCorpus);
        ptest->resetLog(nullptr);
      }
    }else if(vm["policy"].as<string>() == "multi_cyclic_value_unigram_shared") {
      shared_ptr<ModelCRFGibbs> model_unigram = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(corpus, vm));
      std::ifstream file; 
      file.open(vm["unigram_model"].as<string>(), std::fstream::in);
      if(!file.is_open()) 
        throw (vm["unigram_model"].as<string>()+" not found.").c_str();
      file >> *model_unigram;
      file.close();
      if(dataset == "ocr") {
        cast<ModelCRFGibbs>(model_unigram)->extractFeatures = extractOCR;
        cast<ModelCRFGibbs>(model_unigram)->extractFeatAll = extractOCRAll; 
      }
      string name = vm["name"].as<string>();
      const int fold = 10;
      const int fold_l[fold] = {0,5,10,15,20,25,26,27,28,29};
      system(("rm -r "+name+"*").c_str());
      shared_ptr<CyclicValuePolicy> policy = shared_ptr<CyclicValuePolicy>(new MultiCyclicValueUnigramPolicy(model, model_unigram, vm));
      policy->lets_resp_reward = true;
      system(("mkdir -p " + name + "_train").c_str());
      policy->resetLog(shared_ptr<XMLlog>(new XMLlog(name + "_train" + "/policy.xml")));
      train_func(policy);
      if(vm["lets_notrain"].as<bool>()) mapReset(*policy->param, 1);
      policy->test(testCorpus);
      policy->resetLog(nullptr);
      shared_ptr<MultiCyclicValueUnigramPolicy> ptest;
      auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
        return (a.first < b.first);
      };
      sort(policy->test_resp_reward.begin(), policy->test_resp_reward.end(), compare); 
      for(int i : fold_l) {
        double c = policy->test_resp_reward[i * (policy->test_resp_reward.size()-1)/(double)fold_l[fold-1]].first;
	// string myname = name+"_i"+to_string(i);
        string myname = name + "_c" + boost::lexical_cast<string>(c);
        system(("mkdir -p " + myname).c_str());
        ptest = shared_ptr<MultiCyclicValueUnigramPolicy>(new MultiCyclicValueUnigramPolicy(model, model_unigram, vm));
        ptest->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
        ptest->param = policy->param; 
        ptest->c = c;
        ptest->test(testCorpus);
        ptest->resetLog(nullptr);
      }
      system(("rm -r "+name).c_str());
    }else if(vm["policy"].as<string>() == "cyclic_value_shared") {
      string name = vm["name"].as<string>();
      const int fold = 10;
      shared_ptr<CyclicValuePolicy> policy = shared_ptr<CyclicValuePolicy>(new CyclicValuePolicy(model, vm));
      policy->lets_resp_reward = true;
      train_func(policy);
      shared_ptr<CyclicValuePolicy> ptest;
      auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
	return (a.first < b.first);
      };
      sort(policy->resp_reward.begin(), policy->resp_reward.end(), compare); 
      for(int i = 0; i <= fold; i++) {
        double c = policy->resp_reward[i * (policy->resp_reward.size()-1)/(double)fold].first;
        string myname = name+"_i"+to_string(i);
        system(("mkdir -p " + myname).c_str());
        ptest = shared_ptr<CyclicValuePolicy>(new CyclicValuePolicy(model, vm));
        ptest->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
        ptest->param = policy->param; 
        ptest->c = c;
        ptest->test(testCorpus);
        ptest->resetLog(nullptr);
      }
      system(("rm -r "+name).c_str());
    }else if(vm["policy"].as<string>() == "blockpolicy") {
      string name = vm["name"].as<string>();
      const int fold = 20;
      system(("rm -r "+name+"*").c_str());
      auto policy = std::make_shared<BlockPolicy<LockdownPolicy> >(model, vm);
      policy->lets_resp_reward = true;
      policy->model_unigram = model_unigram;
      system(("mkdir -p " + name + "_train").c_str());
      policy->resetLog(shared_ptr<XMLlog>(new XMLlog(name + "_train" + "/policy.xml")));
      train_func(policy);
      int testCount = vm["testCount"].as<size_t>();
      int count = testCorpus->count(testCount);
      size_t T = vm["T"].as<size_t>();
      ptr<BlockPolicy<LockdownPolicy>::Result> result = policy->test(testCorpus, 0);
      policy->resetLog(nullptr);
      auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
        return (a.first < b.first);
      };
      auto compare2 = [] (std::pair<double, double> a, std::pair<double, double> b) {
        return (a.second < b.second);
      };
      std::vector<std::pair<double, double> > budget_acc;
      auto runWithBudget = [&] (double budget) {
        string myname = name + "_b" + boost::lexical_cast<string>(budget);
        system(("mkdir -p " + myname).c_str());
        policy->resetLog(shared_ptr<XMLlog>(new XMLlog(myname + "/policy.xml")));
        policy->test(result, budget); 
        policy->resetLog(nullptr);
        budget_acc.push_back(make_pair(budget, result->score));
        sort(budget_acc.begin(), budget_acc.end(), compare);
      };
      auto findLargestSeg = [&] () -> double {
        double segmax = -DBL_MAX, budget = 0.0;
        for(size_t t = 0; t < budget_acc.size()-1; t++) {
          double seg = budget_acc[t+1].second - budget_acc[t].second;
          if(seg > segmax) {
            segmax = seg;
            budget = (budget_acc[t+1].first + budget_acc[t].first) / 2.0;
          }
        }
        return budget;
      };
      runWithBudget(count);
      for(int t = 0; t < T; t++) {
        if(t == 0) {
          const int segs = 10;
          for(int i = 0; i < segs; i++) {
              runWithBudget(count/(double)segs);
          }
        }else{
          const int segs = 3;
          for(int i = 0; i < segs; i++) {
            runWithBudget(count / (double)segs);
          }
        }
      }
    }
  }catch(char const* ee) {
    cout << "error: " << ee << endl;
  }
}
