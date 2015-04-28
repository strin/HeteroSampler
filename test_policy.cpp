#include "corpus.h"
#include "corpus_ising.h"
#include "objcokus.h"
#include "tag.h"
#include "feature.h"
#include "model.h"
#include "model_opengm.h"
#include "utils.h"
#include "policy.h"
#include "opengm.h"

#include <boost/format.hpp>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/space/simplediscretespace.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>

using namespace std;
using namespace HeteroSampler;
using namespace opengm;

namespace po = boost::program_options;

int main(int argc, char* argv[]) {
  for (int i = 0; i < argc; i++) {
    cout << argv[i] << " ";
  }
  cout << endl;
  try {
    // parse args
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help", "produce help message")
    // model and data
    ("inference", po::value<string>()->default_value("Gibbs"), "inference engine (CRF / OpenGM)")
    ("type", po::value<string>()->default_value("tagging"), "type of the problem (tagging / ocr / ising / opengm)")
    ("model", po::value<string>()->default_value("model/gibbs.model"), "file for the pre-trained model")
    ("unigram_model", po::value<string>(), "file for a unigram model (option, dependency for unigram entropy meta-feature)")
    ("train", po::value<string>()->default_value("data/eng_ner/train"), "training data")
    ("test", po::value<string>()->default_value("data/eng_ner/test"), "test data")
    // learning
    ("policy", po::value<string>()->default_value("gibbs"), "the policy used for sampling (gibbs / adaptive)")
    ("learning", po::value<string>()->default_value("logistic"), "learning strategy (logistic / nn)")
    ("T", po::value<size_t>()->default_value(4), "number of sweeps by the policy")
    ("K", po::value<size_t>()->default_value(1), "number of trajectories")
    ("eta", po::value<double>()->default_value(1), "step-size for policy gradient (adagrad)")
    ("testCount", po::value<size_t>()->default_value(-1), "how many test data used ? default: all (-1). ")
    ("trainCount", po::value<size_t>()->default_value(-1), "how many training data used ? default: all (-1). ")
    ("Q", po::value<size_t>()->default_value(1), "number of passes")
    ("numThreads", po::value<size_t>()->default_value(10), "number of threads to use")
    ("inplace", po::value<bool>()->default_value(true), "set inplace = false causes the sampler to represent entire trajectory")
    ("lets_lazymax", po::value<bool>()->default_value(false), "lazymax is true, the algorithm takes max sample only after each sweep.")
    ("init", po::value<string>()->default_value("random"), "initialization method: random, iid, unigram.")
    ("feat", po::value<std::string>()->default_value(""), "list of meta-features to use, separated with space")
    // simulated annealing
    ("temp", po::value<string>()->default_value(""), "the annealing scheme to use (\"scanline\" or \"\")")
    ("temp_init", po::value<double>()->default_value(1), "initial temperature")
    ("temp_decay", po::value<double>()->default_value(0.9), "decay of temperature.")
    ("temp_magnify", po::value<double>()->default_value(0.1), "magnifying factor of init temperature.")
    // ouput
    ("output", po::value<string>()->default_value("result/default"), "output path for this run")
    ("log", po::value<string>()->default_value("log/latest.txt"), "log file for the model")
    // reward
    ("reward", po::value<int>()->default_value(0), "what is the depth of simulation to compute reward.")
    ("oracle", po::value<int>()->default_value(0), "what is the depth of simulation to compute reward for oracle.")
    ("rewardK", po::value<int>()->default_value(5), "the number of trajectories used to approximate the reward")
    // other options
    ("verbose", po::value<bool>()->default_value(false), "whether to output more debug information")
    ("verbosity", po::value<string>()->default_value(""), "what kind of information to log? ")
    ("lets_notrain", po::value<bool>()->default_value(false), "do not train the policy")
    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }

    // load data
    string train = vm["train"].as<string>(), test = vm["test"].as<string>();
    ptr<Corpus> corpus, test_corpus;
    string type = vm["type"].as<string>();

    if (type == "tagging") {
      corpus = ptr<CorpusLiteral>(new CorpusLiteral());
      test_corpus = ptr<CorpusLiteral>(new CorpusLiteral());
      cast<CorpusLiteral>(corpus)->computeWordFeat();
    } else if (type == "ocr") {
      corpus = std::make_shared<CorpusOCR<16, 8> >();
      test_corpus = std::make_shared<CorpusOCR<16, 8> >();
    } else if (type == "ising") {
      corpus = std::make_shared<CorpusIsing>();
      test_corpus = std::make_shared<CorpusIsing>();
    } else if (type == "opengm") {
      typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
      typedef opengm::GraphicalModel<double, opengm::Adder,
              OPENGM_TYPELIST_2(ExplicitFunction<double> ,
                                PottsFunction<double>),
              Space>
              GraphicalModelType;
      typedef CorpusOpenGM<GraphicalModelType> CorpusOpenGMType;
      corpus = std::make_shared<CorpusOpenGMType>();
      test_corpus = std::make_shared<CorpusOpenGMType>();
    }

    corpus->read(train, false);
    corpus->test_count = vm["trainCount"].as<size_t>();
    test_corpus->read(test, false);
    test_corpus->test_count = vm["testCount"].as<size_t>();

    // load pre-trained model
    shared_ptr<Model> model, model_unigram;
    if (type == "ocr" || type == "ising" || type == "tagging") {
      auto loadGibbsModel = [&] (string name) -> ModelPtr {
        shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(corpus, vm));
        std::ifstream file;
        file.open(name, std::fstream::in);
        if (!file.is_open())
          throw (name + " not found.").c_str();
        file >> *model;
        file.close();
        // extract features based on application.
        if (type == "ocr") {
          cast<ModelCRFGibbs>(model)->extractFeatures = extractOCR;
          cast<ModelCRFGibbs>(model)->extractFeatAll = extractOCRAll;
        } else if (type == "ising") {
          cast<ModelCRFGibbs>(model)->extractFeatures = extractIsing;
          cast<ModelCRFGibbs>(model)->extractFeatAll = extractIsingAll;
          cast<ModelCRFGibbs>(model)->extractFeaturesAtInit = extractIsingAtInit;
          cast<ModelCRFGibbs>(model)->getMarkovBlanket = getIsingMarkovBlanket;
          cast<ModelCRFGibbs>(model)->getInvMarkovBlanket = getIsingMarkovBlanket;
        }
        return model;
      };
      model = loadGibbsModel(vm["model"].as<string>());
      if (vm.count("unigram_model")) {
        model_unigram = loadGibbsModel(vm["unigram_model"].as<string>());
      }
    } else if (type == "opengm") {
      typedef opengm::SimpleDiscreteSpace<size_t, size_t> Space;
      typedef opengm::GraphicalModel<double, opengm::Adder,
              OPENGM_TYPELIST_2(ExplicitFunction<double> ,
                                PottsFunction<double>),
              Space> GraphicalModelType;
      typedef CorpusOpenGM<GraphicalModelType> CorpusOpenGMType;
      model = std::make_shared<ModelEnumerativeGibbs<GraphicalModelType, opengm::Minimizer> >(vm);
    }

    shared_ptr<Policy> policy;
    auto train_func = [&] (shared_ptr<Policy> policy) {
      policy->train_policy(corpus);
    };

    int sysres = 0;
    string name = vm["output"].as<string>();
    const size_t T = vm["T"].as<size_t>();
    removeFile(name);
    makeDirs(name + "/");

    if (vm["policy"].as<string>() == "gibbs")
    {
      Policy::ResultPtr result = nullptr;
      shared_ptr<GibbsPolicy> gibbs_policy;
      gibbs_policy = shared_ptr<GibbsPolicy>(new GibbsPolicy(model, vm));
      gibbs_policy->T = 1;  // do one sweep after another.
      for (size_t t = 1; t <= T; t++) {
        string myname = name + "/T" + to_string(t) + ".xml";
        gibbs_policy->resetLog(std::make_shared<XMLlog>(myname));
        if (t == 1) {
          result = gibbs_policy->test(test_corpus);
        } else {
          gibbs_policy->init_method = "";
          gibbs_policy->test(result);
        }
      }
    }
    else if (vm["policy"].as<string>() == "adaptive")
    {
      const int fold = 20;
      auto policy = std::make_shared<BlockPolicy>(model, vm);
      policy->model_unigram = model_unigram;

      // training
      makeDirs(name + "/train");
      policy->resetLog(shared_ptr<XMLlog>(new XMLlog(name + "_train" + ".xml")));
      policy->train(corpus);
      int testCount = vm["testCount"].as<size_t>();
      int count = test_corpus->count(testCount);
      auto result = policy->test(test_corpus, 0);
      policy->resetLog(nullptr);

      // run with different budgets
      double budget = 0;
      auto runWithBudget = [&] (double b) {
        budget += b;
        string myname = name + "/b" + boost::str(boost::format("%.2f") % budget) + ".xml";
        policy->resetLog(shared_ptr<XMLlog>(new XMLlog(myname)));
        policy->test(result, b);
        policy->resetLog(nullptr);
      };
      runWithBudget(1);
      for (int t = 0; t < T; t++) {
        if (t == 0) {
          const int segs = 10;
          for (int i = 0; i < segs; i++) {
            runWithBudget(1 / (double)segs);
          }
        } else {
          const int segs = 3;
          for (int i = 0; i < segs; i++) {
            runWithBudget(1 / (double)segs);
          }
        }
      }
    }
  } catch (char const* ee) {
    cout << "error: " << ee << endl;
  }

  return 0;
}
