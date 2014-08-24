#include "policy.h"
#include "feature.h"
#include <boost/lexical_cast.hpp>

#define USE_FEAT_ENTROPY 1
#define USE_FEAT_ALL 0
#define USE_FEAT_BIAS 1

namespace po = boost::program_options;

using namespace std;

namespace Tagging {
  Policy::Policy(ModelPtr model, const po::variables_map& vm) 
  :model(model), test_thread_pool(vm["numThreads"].as<size_t>(), 
		      [&] (int tid, MarkovTreeNodePtr node) {
			this->sampleTest(tid, node);
		      }), 
		 thread_pool(vm["numThreads"].as<size_t>(), 
		     [&] (int tid, MarkovTreeNodePtr node) {
		      this->sample(tid, node);
		     }),
   name(vm["name"].as<string>()),
   K(vm["K"].as<size_t>()),
   eta(vm["eta"].as<double>()),
   train_count(vm["trainCount"].as<size_t>()), 
   test_count(vm["testCount"].as<size_t>()),
   verbose(vm["verbose"].as<bool>()), 
   Q(vm["Q"].as<size_t>()),
   param(makeParamPointer()), G2(makeParamPointer()) {
    // init stats.
    ptr<CorpusLiteral> corpus = cast<CorpusLiteral>(model->corpus);
    auto wordent_meanent = corpus->tagEntropySimple();
    wordent = get<0>(wordent_meanent);
    wordent_mean = get<1>(wordent_meanent);
    auto wordfreq_meanfreq = corpus->wordFrequencies();
    wordfreq = get<0>(wordfreq_meanfreq);
    wordfreq_mean = get<1>(wordfreq_meanfreq);
    auto tag_bigram_unigram = corpus->tagBigram();
    tag_bigram = tag_bigram_unigram.first;
    tag_unigram_start = tag_bigram_unigram.second;
    system(("mkdir -p "+name).c_str());
    lg = shared_ptr<XMLlog>(new XMLlog(name+"/policy.xml"));  
    lg->begin("args");
      lg->begin("corpus");
	*lg << vm["train"].as<string>() << endl;
      lg->end();
      lg->begin("wordent_mean");
	*lg << wordent_mean << endl;
      lg->end();
      lg->begin("wordfreq_mean");
	*lg << wordfreq_mean << endl;
      lg->end();
    lg->end();
  }

  Policy::~Policy() {
    while(lg != nullptr and lg->depth() > 0) 
      lg->end();
  }

  double Policy::reward(MarkovTreeNodePtr node) {
    Tag& tag = *node->tag;
    const Sentence* seq = tag.seq;
    double score = 0.0;
    for(int i = 0; i < tag.size(); i++) {
      score -= (tag.tag[i] != seq->tag[i]);
    }
    return score;
  }

  void Policy::sample(int tid, MarkovTreeNodePtr node) {
    this->sampleTest(tid, node);
  }

  void Policy::sampleTest(int tid, MarkovTreeNodePtr node) {
    node->depth = 0;
    try{
      while(true) {
	if(node->depth >= POLICY_MARKOV_CHAIN_MAXDEPTH) 
	  throw "Policy Chain reaches maximum depth.";
	node->tag->rng = &test_thread_pool.rngs[tid];
	node->choice = this->policy(node);
	if(node->choice == -1) {
	  node->log_weight = this->reward(node); 
	  node->gradient = makeParamPointer();
	  break;
	}else{
	  node->log_weight = -DBL_MAX; 
	  node->gradient = model->sampleOne(*node->tag, node->choice);
	}
	node = addChild(node, *node->tag);
      }
    }catch(const char* ee) {
      cout << "error: " << ee << endl;
    }
  }

  void Policy::gradientPolicy(MarkovTree& tree) {
    double log_sum_w = tree.logSumWeights(tree.root); // norm grad, avoid overflow.
    pair<ParamPointer, double> grad_lgweight = tree.aggregateGradient(tree.root, log_sum_w);
    ParamPointer gradient = grad_lgweight.first;
    if(verbose) {
      lg->begin("gradient_agg");
      *lg << *gradient << endl;
      lg->end();
    }
    Tagging::adagrad(param, G2, gradient, eta);
  }

  void Policy::gradientKernel(MarkovTree& tree) {
    double log_sum_w = tree.logSumWeights(tree.root);
    pair<ParamPointer, double> grad_lgweight = tree.aggregateGradient(tree.root, log_sum_w);
    ParamPointer gradient = grad_lgweight.first;
    if(verbose) {
      lg->begin("gradient_agg");
      *lg << *gradient << endl;
      lg->end();
    }
    Tagging::adagrad(model->param, model->G2, gradient, eta);
  }

  void Policy::train(ptr<Corpus> corpus) {
    cout << "> train " << endl;
    lg->begin("train");
      for(size_t q = 0; q < Q; q++) {
	cout << "\t epoch " << q << endl;
	cout << "\t update policy " <<  endl;
	this->trainPolicy(corpus);
	cout << "\t update kernel " << endl;
	this->trainKernel(corpus);
      }
      lg->begin("param");
      *lg << *param;
      lg->end();
    lg->end(); // </train>
  }

  void Policy::trainPolicy(ptr<Corpus> corpus) {
    corpus->retag(*model->corpus);
    size_t count = 0;
    for(const SentencePtr seq : corpus->seqs) {
      if(count >= train_count) break;
      if(count % 1000 == 0) 
	cout << "\t\t " << (double)count/corpus->seqs.size()*100 << " %" << endl;
      if(verbose)
	lg->begin("example_"+to_string(count));
      MarkovTree tree;
      Tag tag(seq.get(), model->corpus, &rng, model->param);
      tree.root->log_weight = -DBL_MAX;
      for(size_t k = 0; k < K; k++) {
	MarkovTreeNodePtr node = addChild(tree.root, tag);
	this->thread_pool.addWork(node); 
      }
      thread_pool.waitFinish();
      /*
      for(size_t k = 0; k < K; k++) {
	MarkovTreeNodePtr node = tree.root->children[k];
	ParamPointer g = makeParamPointer();
	if(node->gradient != nullptr)
	  mapUpdate(*g, *node->gradient);
	while(node->children.size() > 0) {
	  node = node->children[0]; // take final sample.
	  if(node->gradient != nullptr) {
	    mapUpdate(*g, *node->gradient);
	  }
	}
	lg->begin("gradient");
	*lg << *g << endl;
	*lg << node->log_weight << endl;
	lg->end(); 
      }*/
      this->gradientPolicy(tree);
      if(verbose) {
	for(size_t k = 0; k < K; k++) {
	  MarkovTreeNodePtr node = tree.root->children[k];
	  while(node->children.size() > 0) node = node->children[0]; // take final sample.
	  lg->begin("node");
	  this->logNode(node);
	  lg->end(); // </node>
	}
      }
      if(verbose) {
	lg->begin("param");
	*lg << *param;
	lg->end(); // </param>
	lg->end(); // </example>
      }
      count++;
    }
  }

  void Policy::trainKernel(ptr<Corpus> corpus) {
    corpus->retag(*model->corpus);
    size_t count = 0;
    for(const SentencePtr seq : corpus->seqs) {
      if(count >= train_count) break;
      if(count % 1000 == 0) 
	cout << "\t\t " << (double)count/corpus->seqs.size()*100 << " %" << endl;
      MarkovTree tree;
      Tag tag(seq.get(), model->corpus, &rng, model->param);
      tree.root->log_weight = -DBL_MAX;
      for(size_t k = 0; k < K; k++) {
	MarkovTreeNodePtr node = addChild(tree.root, tag);
	this->thread_pool.addWork(node); 
      }
      thread_pool.waitFinish();
      this->gradientKernel(tree);
      count += 1;
    }
  }

  Policy::Result::Result(ptr<Corpus> corpus) 
  :corpus(corpus) {
    nodes.clear();
  }

  Policy::ResultPtr Policy::test(ptr<Corpus> testCorpus) {
    Policy::ResultPtr result = makeResultPtr(testCorpus); 
    result->corpus->retag(*model->corpus);
    result->nodes.resize(min(test_count, testCorpus->seqs.size()), nullptr);
    result->time = 0;
    test(result);
    return result;
  }

  void Policy::test(Policy::ResultPtr result) {
    cout << "> test " << endl;
    lg->begin("test");
    lg->begin("param");
    *lg << *param;
    lg->end(); // </param>
    assert(result != nullptr);
    size_t count = 0;
    lg->begin("example");
    count = 0;
    size_t hit_count = 0, pred_count = 0, truth_count = 0; 
    size_t ave_time = 0;
    vector<MarkovTreeNodePtr> stack;
    vector<int> id;
    for(const SentencePtr seq : result->corpus->seqs) {
      if(count >= test_count) break;
      MarkovTreeNodePtr node;
      if(result->nodes[count] == nullptr) {
	node = makeMarkovTreeNode(nullptr);
	node->tag = makeTagPtr(seq.get(), model->corpus, &rng, model->param);
      }else{
	node = result->nodes[count];
      }
      stack.push_back(node);
      id.push_back(count);
      test_thread_pool.addWork(node);
      count++;
      if(count % thread_pool.numThreads() == 0 || count == test_count 
	  || count == result->corpus->seqs.size()) {
	test_thread_pool.waitFinish();
	for(size_t i = 0; i < id.size(); i++) {
	  MarkovTreeNodePtr node = stack[i];
	  lg->begin("example_"+to_string(id[i]));
	  this->logNode(node);
	  while(node->children.size() > 0) node = node->children[0]; // take final sample.
	  result->nodes[id[i]] = node;
	  ave_time += node->depth+1;
	  if(model->scoring == Model::SCORING_ACCURACY) {
	    tuple<int, int> hit_pred = model->evalPOS(*node->tag);
	    hit_count += get<0>(hit_pred);
	    pred_count += get<1>(hit_pred);
	  }else if(model->scoring == Model::SCORING_NER) {
	    tuple<int, int, int> hit_pred_truth = model->evalNER(*node->tag);
	    hit_count += get<0>(hit_pred_truth);
	    pred_count += get<1>(hit_pred_truth);
	    truth_count += get<2>(hit_pred_truth);
	  }
	  lg->end(); // </example_i>
	}
	stack.clear();
	id.clear();
      }
    }
    lg->end(); // </example>
    double accuracy = (double)hit_count/pred_count;
    double recall = (double)hit_count/truth_count;
    result->time += (double)ave_time/count;
    cout << "time: " << result->time << endl;
    if(model->scoring == Model::SCORING_ACCURACY) {
      lg->begin("accuracy");
      *lg << accuracy << endl;
      cout << "acc: " << accuracy << endl;
      lg->end(); // </accuracy>
      lg->begin("time");
      *lg << result->time << endl;
      lg->end(); // </time>
      lg->end(); // </test>
      result->score = accuracy;
    }else if(model->scoring == Model::SCORING_NER) {
      double f1 = 2 * accuracy * recall / (accuracy + recall);
      lg->begin("accuracy");
      *lg << f1 << endl;
      cout << "f1: " << f1 << endl;
      lg->end(); // </accuracy>
      lg->begin("time");
      *lg << result->time << endl;
      lg->end(); // </time>
      lg->end(); // </test>
      result->score = f1;
    }
    result->score = -1;
  }

  FeaturePointer Policy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = makeFeaturePointer();
    Tag& tag = *node->tag;
    const Sentence& seq = *tag.seq;
    size_t seqlen = tag.size();
    size_t taglen = model->corpus->tags.size();
    string word = cast<TokenLiteral>(seq.seq[pos])->word;
    // bias.
#if USE_FEAT_BIAS == 1
    insertFeature(feat, "b");
#endif
#if USE_FEAT_ENTROPY == 1
    // feat: entropy and frequency.
    if(wordent->find(word) == wordent->end())
      insertFeature(feat, "ent", log(taglen)-wordent_mean);
    else
      insertFeature(feat, "ent", (*wordent)[word]);
#endif
#if USE_FEAT_ALL == 1
    if(wordfreq->find(word) == wordfreq->end())
      insertFeature(feat, "freq", log(model->corpus->total_words)-wordfreq_mean);
    else
      insertFeature(feat, "freq", (*wordfreq)[word]);
    StringVector nlp = NLPfunc(word);
    for(const string wordfeat : *nlp) {
      if(wordfeat == word) continue; 
      string lowercase = word;
      transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
      if(wordfeat == lowercase) continue;
      if(wordfeat[0] == 'p' or wordfeat[0] == 's') continue;
      insertFeature(feat, wordfeat);
    }
#endif
    return feat;
  }

  void Policy::logNode(MarkovTreeNodePtr node) {
    while(node->children.size() > 0) node = node->children[0]; // take final sample.
    lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
    lg->begin("truth"); *lg << node->tag->seq->str() << endl; lg->end();
    lg->begin("tag"); *lg << node->tag->str() << endl; lg->end();
    int hits = 0;
    for(size_t i = 0; i < node->tag->size(); i++) {
      lg->begin("feat"); 
      *lg << *this->extractFeatures(node, i);
      lg->end(); // </feat> */
      if(node->tag->tag[i] == node->tag->seq->tag[i]) {
	hits++;
      }
    }
    lg->begin("resp");
    for(size_t i = 0; i < node->tag->size(); i++) {
      *lg << node->tag->resp[i] << "\t";            
    }
    *lg << endl;
    lg->end();
    lg->begin("mask");
    for(size_t i = 0; i < node->tag->size(); i++) {
      *lg << node->tag->mask[i] << "\t";            
    }
    *lg << endl;
    lg->end();
    lg->begin("dist"); *lg << node->tag->size()-hits << endl; lg->end();
  }

  void Policy::resetLog(std::shared_ptr<XMLlog> new_lg) {
    while(lg->depth() > 0) lg->end();
    lg = new_lg;
  }
  /////////////////////////////////////////////////////////////////////////////
  ////// Gibbs Policy   ///////////////////////////////////////////

  GibbsPolicy::GibbsPolicy(ModelPtr model, const po::variables_map& vm)
  :Policy(model, vm), T(vm["T"].as<size_t>()) 
  {
  }

  int GibbsPolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = 0;
    if(node->depth < T * node->tag->size()) {
      node->time_stamp++;
      return node->depth % node->tag->size();
    }
    return -1; // stop.
  }


  /////////////////////////////////////////////////////////////////////////////
  ////// Entropy Policy ///////////////////////////////////////////

  EntropyPolicy::EntropyPolicy(ModelPtr model, const po::variables_map& vm)
  :Policy(model, vm), 
   threshold(vm["threshold"].as<double>())
  {
  }

  int EntropyPolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = 0;
    if(node->depth < node->tag->size()) { // first pass.
      node->time_stamp++;
      return node->depth;
    }else{
      assert(!node->parent.expired());
      size_t i = node->time_stamp;
      for(; i < 2 * node->tag->size(); i++) {
	size_t pos = i % node->tag->size();
	const string& word = cast<TokenLiteral>(node->tag->seq->seq[pos])->word;  
	double ent = 0;
	if(wordent->find(word) == wordent->end()) 
	  ent = log(model->corpus->tags.size());
	else
	  ent = (*wordent)[word]+wordent_mean;
	if(ent > log(threshold)) {
	  node->tag->mask[pos] = 1;
	  node->time_stamp = i+1;
	  return pos;
	}else{
	  node->tag->mask[pos] = 0;
	}
      }
      node->time_stamp = i;
      return -1; // stop. 
    }
  }



  /////////////////////////////////////////////////////////////////////////////
  ////// Cyclic Policy ///////////////////////////////////////////
  CyclicPolicy::CyclicPolicy(ModelPtr model, const po::variables_map& vm)
  :Policy(model, vm), 
   c(vm["c"].as<double>()) {
  }

  FeaturePointer CyclicPolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = Policy::extractFeatures(node, pos);
#if USE_FEAT_ENTROPY == 1
    insertFeature(feat, "model-ent", node->tag->entropy[pos]);
#endif
#if USE_FEAT_ALL == 1
    if(model->scoring == Model::SCORING_NER) { // tag inconsistency, such as B-PER I-LOC
      string tg = model->corpus->invtags.find(node->tag->tag[pos])->second;
      if(pos >= 1) {
	string prev_tg = model->corpus->invtags.find(node->tag->tag[pos-1])->second;
	if(prev_tg[0] == 'B' and tg[0] == 'I' and tg.substr(1) != prev_tg.substr(1)) 
	  insertFeature(feat, "bad");
      }
      if(pos < node->tag->size()-1) {
	string next_tg = model->corpus->invtags.find(node->tag->tag[pos+1])->second;
	if(next_tg[0] == 'I' and tg[0] == 'B' and tg.substr(1) != next_tg.substr(1)) 
	  insertFeature(feat, "bad");
      }
    }
#endif
    return feat;
  }

  int CyclicPolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = 0;
    if(node->depth < node->tag->size()) {
      node->time_stamp++;
      return node->depth;
    }else{
      objcokus* rng = node->tag->rng;
      assert(!node->parent.expired());
      size_t i = node->time_stamp;
      node->gradient = makeParamPointer();
      for( ; i < 2 * node->tag->size(); i++) {      
	size_t pos = i % node->tag->size();
	FeaturePointer feat = this->extractFeatures(node, pos);
	double resp = logisticFunc(Tagging::score(param, feat));
	node->tag->resp[pos] = resp;
	if(rng->random01() < resp) {
	  node->tag->mask[pos] = 1;
	  mapUpdate(*node->gradient, *feat, (1-resp));
	  node->time_stamp = i+1;
	  return pos;
	}else{
	  node->tag->mask[pos] = 0;
	  mapUpdate(*node->gradient, *feat, -resp);	
	}
      }
      node->time_stamp = i;
      return -1;
    }
  }

  double CyclicPolicy::reward(MarkovTreeNodePtr node) {
    return Policy::reward(node) - this->c * (node->depth + 1);
  }


  /////////////////////////////////////////////////////////////////////////////////////
  //////// Cyclic Value Policy /////////////////////////////////////////////////////////////
  CyclicValuePolicy::CyclicValuePolicy(ModelPtr model, const po::variables_map& vm)
  :CyclicPolicy(model, vm), lets_resp_reward(false) { 
  }

  // will update gradient of policy.
  void CyclicValuePolicy::sample(int tid, MarkovTreeNodePtr node) {
    node->depth = 0;
    node->choice = -1;
    try{
      node->tag->rng = &thread_pool.rngs[tid];
      for(size_t i = 0; i < node->tag->size(); i++) {
	model->sampleOne(*node->tag, i);
      }
      node->gradient = makeParamPointer();
      Tag old_tag(*node->tag);
      for(size_t i = 0; i < node->tag->size(); i++) {
	auto is_equal = [&] () {
	  return (double)(node->tag->tag[i] == node->tag->seq->tag[i]); 
	};
	double reward_baseline = is_equal();
	model->sampleOne(*node->tag, i);
	double reward = is_equal();
	double logR = reward - reward_baseline; 
	FeaturePointer feat = this->extractFeatures(node, i);   
	double resp = Tagging::score(param, feat);
	if(lets_resp_reward) {
	  resp_reward.push_back(make_pair(resp, logR));
	}
  //      cout << "logR: " << logR << ", resp: " << resp << endl;
	mapUpdate(*node->gradient, *feat, 2 * (logR - resp)); 
      }
      node->log_weight = 0;
    }catch(const char* ee) {
      cout << "error: " << ee << endl;
    }
  }

  void CyclicValuePolicy::trainPolicy(ptr<Corpus> corpus) {
    if(lets_resp_reward and K > 1 and thread_pool.numThreads() > 1) 
      throw "multithread environment cannot record reward.";
    if(lets_resp_reward) 
      resp_reward.clear();
    Policy::trainPolicy(corpus);
    if(lets_resp_reward) {
      lg->begin("resp_reward");
      for(const pair<double, double>& p : resp_reward) {
	*lg << p.first << " " << p.second << endl;
      }
      lg->end(); // </resp_reward>
    }
  }

  // will update gradient of transition.
  int CyclicValuePolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = 0;
    if(node->depth < node->tag->size()) {
      node->time_stamp++;
      return node->depth;
    }else{
      objcokus* rng = node->tag->rng;
      assert(!node->parent.expired());
      size_t i = node->time_stamp;
      for(; i < 2 * node->tag->size(); i++) {      
	size_t pos = i % node->tag->size();
	FeaturePointer feat = this->extractFeatures(node, pos);
	double resp = Tagging::score(param, feat);
	node->tag->resp[pos] = resp;
	// if(rng->random01() < resp) { // strategy 1. randomized test.
	if(resp > c) { // strategy 2. deterministic test.
	  node->tag->mask[pos] = 1;
	  node->time_stamp = i+1;
	  return pos;
	}else{
	  node->tag->mask[pos] = 0;
	}
      }
      node->time_stamp = i;
      return -1;
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////
  //////// Multi Cyclic Value Policy /////////////////////////////////////////////////
  MultiCyclicValuePolicy::MultiCyclicValuePolicy(ModelPtr model, const po::variables_map& vm)
  :CyclicValuePolicy(model, vm), 
   T(vm["T"].as<size_t>()) { 
  }

  int MultiCyclicValuePolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = -1;
    if(node->depth < node->tag->size()) {
      node->time_stamp++;
      return node->depth;
    }else{
      objcokus* rng = node->tag->rng;
      assert(!node->parent.expired());
      size_t i = node->time_stamp + 1;
      node->gradient = makeParamPointer();
      for(; i < T * node->tag->size(); i++) {      
	size_t pos = i % node->tag->size();
	node->time_stamp = i;
	FeaturePointer feat = this->extractFeatures(node, pos);
	double resp = Tagging::score(param, feat);
	node->tag->resp[pos] = resp;
	if(resp > c) { 
	  node->tag->mask[pos] = 1;
	  return pos;
	}else{
	  node->tag->mask[pos] = 0;
	}
      }
      node->time_stamp = i;
      return -1;
    }
  }

  void MultiCyclicValuePolicy::sample(int tid, MarkovTreeNodePtr node) {
    node->depth = 0;
    node->choice = -1;
    try{
      node->tag->rng = &thread_pool.rngs[tid];
      for(size_t i = 0; i < node->tag->size(); i++) {
	model->sampleOne(*node->tag, i);
      }
      node->gradient = makeParamPointer();
      for(size_t t = 1; t < T; t++) {
	for(size_t i = 0; i < node->tag->size(); i++) {
	  node->time_stamp = t * node->tag->size() + i;
	  auto is_equal = [&] () {
	    return (double)(node->tag->tag[i] == node->tag->seq->tag[i]); 
	  };
	  double reward_baseline = is_equal();
	  model->sampleOne(*node->tag, i);
	  double reward = is_equal();
	  double logR = reward - reward_baseline; 
	  FeaturePointer feat = this->extractFeatures(node, i);   
	  double resp = Tagging::score(param, feat);
	  if(lets_resp_reward) {
	    thread_pool.lock();
	    resp_reward.push_back(make_pair(resp, logR));
	    thread_pool.unlock();
	  }
	  mapUpdate(*node->gradient, *feat, 2 * (logR - resp)); 
	}
      }
      node->log_weight = 0;
    }catch(const char* ee) {
      cout << "error: " << ee << endl;
    }
  }

  FeaturePointer MultiCyclicValuePolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = CyclicValuePolicy::extractFeatures(node, pos);
    int pass = node->time_stamp / node->tag->size();
    for(pair<string, double>& p : *feat) {
      p.first = boost::lexical_cast<string>(pass) + "-" + p.first;
    }
    return feat; 
  }


  void MultiCyclicValuePolicy::logNode(MarkovTreeNodePtr node) {
    size_t pass = 1;
    while(true) {
      if(node->time_stamp >= pass * node->tag->size()) {
	pass += 1;
	lg->begin("pass_"+boost::lexical_cast<string>(pass));
	  lg->begin("tag"); *lg << node->tag->str() << endl; lg->end();
	  for(size_t i = 0; i < node->tag->size(); i++) {
	    lg->begin("feat"); 
	    *lg << *this->extractFeatures(node, i);
	    lg->end(); // </feat> */
	  }
	  lg->begin("resp");
	  for(size_t i = 0; i < node->tag->size(); i++) {
	    *lg << node->tag->resp[i] << "\t";            
	  }
	  *lg << endl;
	  lg->end(); // </resp>
	  lg->begin("mask");
	  for(size_t i = 0; i < node->tag->size(); i++) {
	    *lg << node->tag->mask[i] << "\t";            
	  }
	  *lg << endl;
	  int hits = 0;
	  for(size_t i = 0; i < node->tag->size(); i++) {
	    if(node->tag->tag[i] == node->tag->seq->tag[i]) {
	      hits++;
	    }
	  }
	  lg->begin("dist"); *lg << node->tag->size()-hits << endl; lg->end();
	  lg->end(); // </mask>
	lg->end(); // </pass>
      }
      if(node->children.size() > 0)
	node = node->children[0]; // take final sample.
      else
	break;
    }
    lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
    lg->begin("truth"); *lg << node->tag->seq->str() << endl; lg->end();
  }
}
