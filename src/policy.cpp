#include "policy.h"

namespace po = boost::program_options;

using namespace std;

Policy::Policy(ModelPtr model, const po::variables_map& vm) 
:model(model), test_thread_pool(vm["numThreads"].as<size_t>(), 
		    [&] (int tid, MarkovTreeNodePtr node) {
		      this->sampleTest(tid, node);
		    }), 
 name(vm["name"].as<string>()),
 K(vm["K"].as<size_t>()),
 eta(vm["eta"].as<double>()),
 train_count(vm["trainCount"].as<size_t>()), 
 test_count(vm["testCount"].as<size_t>()),
 param(makeParamPointer()), G2(makeParamPointer()) {
  // init stats.
  auto wordent_meanent = model->tagEntropySimple();
  wordent = get<0>(wordent_meanent);
  wordent_mean = get<1>(wordent_meanent);
  auto wordfreq_meanfreq = model->wordFrequencies();
  wordfreq = get<0>(wordfreq_meanfreq);
  wordfreq_mean = get<1>(wordfreq_meanfreq);
  auto tag_bigram_unigram = model->tagBigram();
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
  while(lg->depth() > 0) 
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

void Policy::sampleTest(int tid, MarkovTreeNodePtr node) {
  node->depth = 0;
  try{
    while(true) {
      if(node->depth >= POLICY_MARKOV_CHAIN_MAXDEPTH) 
	throw "Policy Chain reaches maximum depth.";
      node->tag->rng = &test_thread_pool.rngs[tid];
      node->choice = this->policy(node);
      node->log_weight = -DBL_MAX; 
      if(node->choice == -1) {
	node->log_weight = this->reward(node); 
	break;
      }
      model->sampleOne(*node->tag, node->choice);
      node = addChild(node, *node->tag);
    }
  }catch(const char* ee) {
    cout << "error: " << ee << endl;
  }
}

void Policy::train(const Corpus& corpus) {
  cout << "> train " << endl;
  Corpus retagged(corpus);
  retagged.retag(*model->corpus);
  lg->begin("train");
  size_t count = 0;
  for(const Sentence& seq : retagged.seqs) {
    if(count >= train_count) break;
    lg->begin("example_"+to_string(count));
    MarkovTree tree;
    Tag tag(&seq, model->corpus, &rng, model->param);
    tree.root->log_weight = -DBL_MAX;
    for(size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = addChild(tree.root, tag);
      this->test_thread_pool.addWork(node); 
    }
    test_thread_pool.waitFinish();
    for(size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = tree.root->children[k];
      ParamPointer g = makeParamPointer();
      if(node->gradient != nullptr)
	mapUpdate(*g, *node->gradient);
      while(node->children.size() > 0) {
	node = node->children[0]; // take final sample.
	if(node->gradient != nullptr) {
	  mapUpdate(*g, *node->gradient);
	  *lg << (*g)["freq"] << " " << (*node->gradient)["freq"] << endl;
	}
      }
      lg->begin("gradient");
      *lg << *g << endl;
      *lg << node->log_weight << endl;
      lg->end();
    }
    pair<ParamPointer, double> grad_lgweight = tree.aggregateGradient(tree.root, 0);
    ParamPointer gradient = grad_lgweight.first;
    lg->begin("gradient_agg");
    *lg << *gradient << endl;
    lg->end();
    ::adagrad(param, G2, gradient, eta);
    for(size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = tree.root->children[k];
      while(node->children.size() > 0) node = node->children[0]; // take final sample.
      lg->begin("node");
      this->logNode(node);
      lg->end(); // </node>
    }
    lg->begin("param");
    *lg << *param;
    lg->end(); // </param>
    lg->end(); // </example>
    count++;
  }
  lg->begin("param");
  *lg << *param;
  lg->end();
  lg->end(); // </train>
}

double Policy::test(const Corpus& testCorpus) {
  cout << "> test " << endl;
  Corpus retagged(testCorpus);
  retagged.retag(*model->corpus);
  vector<MarkovTreeNodePtr> result;
  lg->begin("test");
  size_t count = 0;
  for(const Sentence& seq : retagged.seqs) {
    if(count >= test_count) break;
    MarkovTreeNodePtr node = makeMarkovTreeNode(nullptr);
    node->tag = makeTagPtr(&seq, model->corpus, &rng, model->param);
    test_thread_pool.addWork(node);
    result.push_back(node);
    count++;
  }
  test_thread_pool.waitFinish();
  lg->begin("example");
  count = 0;
  size_t hit_count = 0, pred_count = 0, truth_count = 0; 
  for(MarkovTreeNodePtr node : result) {
    while(node->children.size() > 0) node = node->children[0]; // take final sample.
    lg->begin("example_"+to_string(count));
    this->logNode(node);
    if(model->corpus->mode == Corpus::MODE_POS) {
      tuple<int, int> hit_pred = model->evalPOS(*node->tag);
      hit_count += get<0>(hit_pred);
      pred_count += get<1>(hit_pred);
    }else if(model->corpus->mode == Corpus::MODE_NER) {
      tuple<int, int, int> hit_pred_truth = model->evalNER(*node->tag);
      hit_count += get<0>(hit_pred_truth);
      pred_count += get<1>(hit_pred_truth);
      truth_count += get<2>(hit_pred_truth);
    }
    lg->end(); // </example_i>
    count++;
  }
  lg->end(); // </example>
  lg->begin("accuracy");
  double accuracy = (double)hit_count/pred_count;
  double recall = (double)hit_count/truth_count;
  if(model->corpus->mode == Corpus::MODE_POS) {
    *lg << accuracy << endl;
    lg->end(); // </accuracy>
    lg->end(); // </test>
    return accuracy;
  }else if(model->corpus->mode == Corpus::MODE_NER) {
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    *lg << f1 << endl;
    lg->end(); // </accuracy>
    lg->end(); // </test>
    return f1;
  }
  return -1;
}

FeaturePointer Policy::extractFeatures(MarkovTreeNodePtr node, int pos) {
  FeaturePointer feat = makeFeaturePointer();
  Tag& tag = *node->tag;
  const Sentence& seq = *tag.seq;
  size_t seqlen = tag.size();
  size_t taglen = model->corpus->tags.size();
  string word = seq.seq[pos].word;
  // bias.
  (*feat)["b"] = 1;
  // feat: entropy and frequency.
  if(wordent->find(word) == wordent->end())
    (*feat)["ent"] = log(taglen)-wordent_mean;
  else
    (*feat)["ent"] = (*wordent)[word];
  /*if(wordfreq->find(word) == wordfreq->end())
    (*feat)["freq"] = log(model->corpus->total_words)-wordfreq_mean;
  else
    (*feat)["freq"] = (*wordfreq)[word];*/
  return feat;
}

void Policy::logNode(MarkovTreeNodePtr node) {
  lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
  lg->begin("truth"); *lg << node->tag->seq->str() << endl; lg->end();
  lg->begin("tag"); *lg << node->tag->str() << endl; lg->end();
  int hits = 0;
  for(size_t i = 0; i < node->tag->size(); i++) {
    lg->begin("feat"); 
    *lg << *this->extractFeatures(node, i);
    lg->end(); // </feat>
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
/////////////////////////////////////////////////////////////////////////////
////// Gibbs Policy   ///////////////////////////////////////////

GibbsPolicy::GibbsPolicy(ModelPtr model, const po::variables_map& vm)
:Policy(model, vm),  
 T(vm["T"].as<size_t>())
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
      const string& word = node->tag->seq->seq[pos].word;  
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
 c(vm["c"].as<double>())
{
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
    for(; i < 2 * node->tag->size(); i++) {      
      size_t pos = i % node->tag->size();
      FeaturePointer feat = this->extractFeatures(node, pos);
      double resp = logisticFunc(::score(param, feat));
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
