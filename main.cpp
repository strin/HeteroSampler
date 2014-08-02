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
      ("eta", po::value<double>(), "step size")
      ("etaT", po::value<double>(), "step size for time adaptation")
      ("T", po::value<int>(), "number of transitions")
      ("B", po::value<int>(), "number of burnin steps")
      ("Q", po::value<int>(), "number of passes")
      ("Q0", po::value<int>(), "number of passes for smart init")
      ("K", po::value<int>(), "number of threads/particles")
      ("c", po::value<double>(), "extent of time regularization")
      ("windowL", po::value<int>(), "window size for node-wise features")
      ("Tstar", po::value<double>(), "time resource constraints")
      ("eps_split", po::value<int>(), "prob of split in MarkovTree")
      ("mode", po::value<string>(), "mode (POS / NER)")
      ("train", po::value<string>(), "training data")
      ("test", po::value<string>(), "test data")
      ("testFrequency", po::value<double>(), "frequency of testing")
      ("stopDataSize", po::value<int>(), "stopDataSize")
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
  int Q0 = 1;
  if(vm.count("Q0"))
    Q0 = vm["Q0"].as<int>();
  int K = 5;
  if(vm.count("K"))
    K = vm["K"].as<int>();
  double eta = 0.4;
  if(vm.count("eta"))
    eta = vm["eta"].as<double>();
  double etaT = 0.4;
  if(vm.count("etaT"))
    etaT = vm["etaT"].as<double>();
  double testFrequency = 0.3;
  if(vm.count("testFrequency"))
    testFrequency = vm["testFrequency"].as<double>();

  // run.
  Corpus::Mode mode = Corpus::MODE_POS;
  if(vm.count("mode") && vm["mode"].as<string>() == "NER")
    mode = Corpus::MODE_NER;
  string train = "data/eng_ner/train", test = "data/eng_ner/test";
  if(vm.count("train")) train = vm["train"].as<string>();
  if(vm.count("test")) test = vm["test"].as<string>();
  Corpus corpus(mode);
  corpus.read(train);
  Corpus testCorpus(mode);
  testCorpus.read(test);
  auto set_param = [&] (shared_ptr<Model> model) {
    model->T = T;
    model->Q = Q;
    model->B = B;
    model->eta = eta;
    model->testFrequency = testFrequency;
  };
  int windowL = 0;
  if(vm.count("windowL"))
    windowL = vm["windowL"].as<int>();
  try{
    if(inference == "Gibbs") {
      shared_ptr<Model> model = shared_ptr<ModelCRFGibbs>(new ModelCRFGibbs(corpus, windowL));
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "Simple") {
      shared_ptr<Model> model = shared_ptr<Model>(new ModelSimple(corpus, windowL));
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "FwBw") {
      shared_ptr<Model> model = shared_ptr<Model>(new ModelFwBw(corpus, windowL));
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "TreeUA") {
      shared_ptr<ModelTreeUA> model = shared_ptr<ModelTreeUA>(new ModelTreeUA(corpus, windowL, K, T, B, Q, Q0, eta));
      if(vm.count("eps_split")) {
	model->eps_split = vm["eps_split"].as<int>();
      }
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "AdaTree") {
      double m_c = 1.0;
      if(vm.count("c")) m_c = vm["c"].as<double>();
      double Tstar = T;
      shared_ptr<ModelAdaTree> model = shared_ptr<ModelAdaTree>(new ModelAdaTree(corpus, windowL,
						  K, m_c, Tstar, T, B, Q, Q0));
      set_param(model);
      if(vm.count("eps_split")) {
	model->eps_split = vm["eps_split"].as<int>();
      }
      if(vm.count("etaT")) {
	model->etaT = vm["etaT"].as<double>();
      }
      model->run(testCorpus);
    }else if(inference == "Prune") {
      double m_c = 1.0;
      if(vm.count("c")) m_c = vm["c"].as<double>();
      int stop_data_size = 100;
      if(vm.count("stopDataSize")) stop_data_size = vm["stopDataSize"].as<int>();
      double Tstar = T;
      shared_ptr<ModelPrune> model = shared_ptr<ModelPrune>(new ModelPrune(corpus, windowL, K, stop_data_size,
						    m_c, Tstar, etaT, T, B, Q, Q0));
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "GibbsIncr") { 
      shared_ptr<Model> model = shared_ptr<Model>(new ModelIncrGibbs(corpus, windowL));
      set_param(model);
      model->run(testCorpus);
    }else if(inference == "TagEntropySimple") {
      shared_ptr<Model> model = shared_ptr<Model>(new ModelSimple(corpus, windowL));
      set_param(model);
      model->run(testCorpus);
      FeaturePointer feat = model->tagEntropySimple();
      XMLlog log_entropy("tag_entropy.xml"); 
      log_entropy << *feat;
      feat = model->wordFrequencies();
      XMLlog log_freq("word_freq.xml");
      log_freq << *feat;
      XMLlog log_tagbigram("tag_bigram.xml");
      auto mat_vec = model->tagBigram();
      Vector2d mat = mat_vec.first;
      vector<double> vec = mat_vec.second;
      log_tagbigram.begin("vector");
      for(size_t i = 0; i < corpus.tags.size(); i++) 
	log_tagbigram << vec[i] << " ";
      log_tagbigram << endl;
      log_tagbigram.end();
      log_tagbigram.begin("matrix");
      for(size_t i = 0; i < corpus.tags.size(); i++) {
	for(size_t j = 0; j < corpus.tags.size(); j++) {
	  log_tagbigram << mat[i][j] << " ";
	}
	log_tagbigram << endl;
      }
      log_tagbigram.end();
    }
  }catch(char const* exception) {
    cerr << "Exception: " << string(exception) << endl;
  }
}
