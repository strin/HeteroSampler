#include "policy.h"

namespace po = boost::program_options;

using namespace std;

Policy::Policy(ModelPtr model, const po::variables_map& vm) 
:model(model), test_thread_pool(vm["numThreads"].as<size_t>(), 
		    [&] (int tid, MarkovTreeNodePtr node) {
		      this->sampleTest(tid, node);
		    }), 
 name(vm["name"].as<string>()),
 train_count(vm["trainCount"].as<size_t>()), 
 test_count(vm["testCount"].as<size_t>()) {
 system(("mkdir -p "+name).c_str());
  lg = shared_ptr<XMLlog>(new XMLlog(name+"/policy.xml"));  
  lg->begin("args");
  lg->end();
  // init stats.
  wordent = model->tagEntropySimple();
  wordfreq = model->wordFrequencies();
  auto tag_bigram_unigram = model->tagBigram();
  tag_bigram = tag_bigram_unigram.first;
  tag_unigram_start = tag_bigram_unigram.second;
}

Policy::~Policy() {
  while(lg->depth() > 0) 
    lg->end();
}

void Policy::sampleTest(int tid, MarkovTreeNodePtr node) {
  node->depth = 0;
  try{
    while(true) {
      if(node->depth >= POLICY_MARKOV_CHAIN_MAXDEPTH) 
	throw "Policy Chain reaches maximum depth.";
      node->tag->rng = &test_thread_pool.rngs[tid];
      node->choice = this->policy(node);
      if(node->choice == -1) break;
      model->sampleOne(*node->tag, node->choice);
      node = addChild(node, *node->tag);
    }
  }catch(const char* ee) {
    cout << "error: " << ee << endl;
  }
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
  size_t hit_count = 0, pred_count = 0; 
  for(MarkovTreeNodePtr node : result) {
    while(node->children.size() > 0) node = node->children[0]; // take final sample.
    lg->begin("example_"+to_string(count));
    this->logNode(node);
    for(size_t i = 0; i < node->tag->size(); i++) {
      if(node->tag->tag[i] == node->tag->seq->tag[i]) {
	hit_count++;
      }
      pred_count++;
    }
    lg->end(); // </example_i>
    count++;
  }
  lg->end(); // </example>
  double acc = (double)hit_count/pred_count;
  lg->begin("accuracy");
    *lg << acc << endl;
  lg->end(); // </accuracy>
  lg->end(); // </test>
  return acc;
}

FeaturePointer Policy::extractFeatures(MarkovTreeNodePtr node, int pos) {
  FeaturePointer feat = makeFeaturePointer();
  Tag& tag = *node->tag;
  const Sentence& seq = *tag.seq;
  size_t seqlen = tag.size();
  size_t taglen = model->corpus->tags.size();
  string word = seq.seq[pos].word;
  // feat: entropy and frequency.
  if(wordent->find("ent") == wordent->end())
    (*feat)["ent"] = log(taglen);
  else
    (*feat)["ent"] = (*wordent)[word];
  if(wordfreq->find(word) == wordfreq->end())
    (*feat)["freq"] = log(model->corpus->total_words);
  else
    (*feat)["freq"] = (*wordfreq)[word];
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
	ent = (*wordent)[word];
      if(ent > log(threshold)) {
	node->time_stamp = i+1;
	return pos;
      }
    }
    node->time_stamp = i;
    return -1; // stop. 
  }
}


