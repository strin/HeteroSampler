#include "model.h"
#include "utils.h"
#include "log.h"
#include "MarkovTree.h"
#include <iostream>
#include <string>
#include <map>
#include <cmath>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::placeholders;
namespace po = boost::program_options;

ModelTreeUA::ModelTreeUA(const Corpus* corpus, const po::variables_map& vm) 
:ModelCRFGibbs(corpus, vm), eps(0), 
 eps_split(vm["eps_split"].as<double>()), 
 Q0(vm["Q0"].as<int>()) {
  Model::K = K;
  this->initThreads(K);
}

void ModelTreeUA::run(const Corpus& testCorpus) {
  auto init_simple = [&] () {
    ModelSimple simple_model(this->corpus, vm);
    simple_model.run(testCorpus, true);
    copyParamFeatures(simple_model.param, "simple-", this->param, "");
  };
  if(Q0 > 0) {
    init_simple();
    ofstream file;
    file.open("simple.model");
    file << (*this);
    file.close();
  }else if(Q0 == -1) { // read model from simple.txt
    ifstream file; 
    file.open("simple.model", ifstream::in);
    if(!file.is_open()) {
      init_simple();
    }else{
      file >> (*this);
    }
  }
  Model::run(testCorpus); 
}


void ModelTreeUA::workerThreads(int tid, shared_ptr<MarkovTreeNode> node, Tag tag) {
    XMLlog& lg = *th_log[tid];
    while(true) {
      int pos = node->depth % tag.size();
      tag.rng = &rngs[tid];
      if(node->depth >= tag.size())
	pos = rngs[tid].randomMT() % tag.size();
      node->gradient = 	tag.proposeGibbs(pos, [&] (const Tag& tag) -> FeaturePointer {
					  return this->extractFeatures(tag, pos);  
					}, true);
      if(node->depth < B) node->log_weight = -DBL_MAX;
      else node->log_weight = this->score(tag); 

      if(node->depth == 0) { // multithread split.
	unique_lock<mutex> lock(th_mutex);
	active_work--;
	for(int k = 0; k < K; k++) {
	  node->children.push_back(makeMarkovTreeNode(node));
	  th_work.push_back(make_tuple(node->children.back(), tag));
	}
	th_cv.notify_all();
	lock.unlock();
	return;
      }/*else if(log(rng.random01()) < log(this->eps_split)) {
	for(int k = 0; k < K; k++) {
	  node->children.push_back(makeMarkovTreeNode(node));
	  workerThreads(tid, seed, node->children.back(), tag, rng);
	}
	return;
      }*/else if(node->depth >= B && log(rngs[tid].random01()) < log(this->eps)) { // stop.
	lg.begin("final-tag");  lg << tag.str() << endl; lg.end();
	lg.begin("weight"); lg << node->log_weight << endl; lg.end();
	lg.begin("time"); lg << node->depth << endl; lg.end();
	node->tag = shared_ptr<Tag>(new Tag(tag));
	unique_lock<mutex> lock(th_mutex);
	active_work--;
	lock.unlock();
	return;
      }else{                             // proceed as chain.
	node->children.push_back(makeMarkovTreeNode(node));
	node = node->children.back();
      }
    }
}

void ModelTreeUA::initThreads(size_t numThreads) {
  th.resize(numThreads);
  this->rngs.resize(numThreads);
  active_work = 0;
  for(size_t ni = 0; ni < numThreads; ni++) {
    this->th_stream.push_back(shared_ptr<stringstream>(new stringstream()));
    this->th_log.push_back(shared_ptr<XMLlog>(new XMLlog(*th_stream.back())));
    this->th[ni] = shared_ptr<thread>(new thread([&] (int tid) {
      unique_lock<mutex> lock(th_mutex);
      while(true) {
	if(th_work.size() > 0) {
	  tuple<shared_ptr<MarkovTreeNode>, Tag> work = th_work.front();
	  th_work.pop_front();
	  active_work++;
	  lock.unlock();
	  th_stream[tid]->str("");
	  workerThreads(tid,  get<0>(work), get<1>(work));
	  th_finished.notify_all();
	  lock.lock();
	}else{
	  th_cv.wait(lock);	
	}
      }
    }, ni));
  }
}

shared_ptr<MarkovTree> ModelTreeUA::explore(const Sentence& seq) {
  this->B = seq.size();     // at least one sweep.
  this->eps = 1.0/T;
  shared_ptr<MarkovTree> tree = shared_ptr<MarkovTree>(new MarkovTree());
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  Tag tag(&seq, corpus, &rngs[0], param);
  objcokus cokus; // cokus is not re-entrant.
  cokus.seedMT(0);
  unique_lock<mutex> lock(th_mutex);
  th_work.push_back(make_tuple(tree->root, tag));
  th_cv.notify_one();
  while(true) {
    th_finished.wait(lock);
    if(active_work + th_work.size() == 0) break;
  }
  lock.unlock();
  for(int k = 0; k < K; k++) {
    xmllog.begin("thread_"+to_string(k));
    xmllog.logRaw(th_stream[k]->str());
    xmllog << endl;
    xmllog.end();
  }
  return tree;
}

ParamPointer ModelTreeUA::gradient(const Sentence& seq) {
  return explore(seq)->expectedGradient();
}

TagVector ModelTreeUA::sample(const Sentence& seq) {
  return explore(seq)->getSamples();
}
double ModelTreeUA::score(const Tag& tag) {
  const Sentence* seq = tag.seq;
  double score = 0.0;
  for(int i = 0; i < tag.size(); i++) {
    score -= (tag.tag[i] != seq->tag[i]);
  }
  return score;
}


ModelAdaTree::ModelAdaTree(const Corpus* corpus, const po::variables_map& vm) 
:ModelTreeUA(corpus, vm), 
 m_c(vm["c"].as<double>()),m_Tstar(vm["Tstar"].as<double>()), 
 etaT(vm["etaT"].as<double>()) {
  // aggregate stats. 
  auto wordent_meanent = tagEntropySimple();
  wordent = get<0>(wordent_meanent);
  wordent_mean = get<1>(wordent_meanent);
  auto wordfreq_meanfreq = wordFrequencies();
  wordfreq = get<0>(wordfreq_meanfreq);
  wordfreq_mean = get<1>(wordfreq_meanfreq);
  auto tag_bigram_unigram = tagBigram();
  tag_bigram = tag_bigram_unigram.first;
  tag_unigram_start = tag_bigram_unigram.second;
}

void ModelAdaTree::workerThreads(int tid, shared_ptr<MarkovTreeNode> node, Tag tag) { 
    XMLlog& lg = *th_log[tid];
    while(true) {
      int pos = rngs[tid].randomMT() % tag.size();
      node->gradient = tag.proposeGibbs(pos, 
					[&] (const Tag& tag) -> FeaturePointer {
					  return this->extractFeatures(tag, pos);  
					}, true);
      node->tag = shared_ptr<Tag>(new Tag(tag));
      auto predT = this->logisticStop(node, *tag.seq, tag); 
      double prob = get<0>(predT);
      node->posgrad = get<1>(predT);
      node->neggrad = get<2>(predT);
      FeaturePointer feat = get<3>(predT); 
      th_mutex.lock();
      this->configStepsize(feat, this->etaT);
      th_mutex.unlock();

      lg.begin("tag"); 
      lg  << "[thread: " << tid << "] "
	  << "[depth: " << node->depth << "] " 
	  << "[prob: " << prob << "] "
	  << tag.str() << endl;
      lg.end();
      
      if(node->depth < B) node->log_weight = -DBL_MAX;
      else node->log_weight = this->score(node, tag)+log(prob); 

      if(node->depth == 0 || log(rngs[tid].random01()) < log(this->eps_split)) { // multithread split.
	unique_lock<mutex> lock(th_mutex);
	for(int k = 0; k < K; k++) {
	  node->children.push_back(makeMarkovTreeNode(node));
	  tag.rng = &rngs[k];
	  th_work.push_back(make_tuple(node->children.back(), tag));
	}
	active_work--;
	th_cv.notify_all();
	lock.unlock();
	node->stop_feat = feat;
	return;
      }else if(node->depth >= B && log(rngs[tid].random01()) < log(prob)) { // stop.
	unique_lock<mutex> lock(th_mutex);
	lg.begin("final-tag");  lg << tag.str() << endl; lg.end();
	lg.begin("weight"); lg << node->log_weight << endl; lg.end();
	lg.begin("time"); lg << node->depth << endl; lg.end();
	lg.begin("feat");
	for(const pair<string, double>& p : *feat) {
	  lg << p.first << " : " << p.second 
		 << " , param : " << (*param)[p.first] << endl;
	}
	lg.end();
	active_work--;
	lock.unlock();
	return;
      }else{                             // proceed as chain.
	node->children.push_back(makeMarkovTreeNode(node));
	node = node->children.back();
      }
    }
}

FeaturePointer ModelAdaTree::extractStopFeatures(MarkovTreeNodePtr node, const Sentence& seq, const Tag& tag, int pos) {
  Tag mytag(tag);
  FeaturePointer feat = makeFeaturePointer();
  size_t taglen = corpus->tags.size();
  string word = seq.seq[pos].word;
  /* dataset statistics */
  insertFeature(feat, "ent", log(taglen));
  if(wordent->find(word) != wordent->end()) 
    insertFeature(feat, "ent", (*wordent)[word]);
  insertFeature(feat, "freq", log(corpus->total_words));
  if(wordfreq->find(word) != wordfreq->end()) 
    insertFeature(feat, "freq", (*wordfreq)[word]);
  /* posterior statistics */
  insertFeature(feat, "score", tag.score(this->extractFeatures(tag, pos)));
  double sc[taglen];
  for(size_t i = 0; i < taglen; i++) {
    mytag.tag[pos] = i;
    sc[i] = mytag.score(this->extractFeatures(mytag, pos));
  }
  logNormalize(sc, taglen);
  insertFeature(feat, "ent-score", logEntropy(sc, taglen));
  return feat;
}

FeaturePointer ModelAdaTree::extractStopFeatures(MarkovTreeNodePtr node, const Sentence& seq, const Tag& tag) {
  FeaturePointer feat = makeFeaturePointer();
  size_t seqlen = tag.size();
  size_t taglen = corpus->tags.size();
  // feat: bias.
  insertFeature(feat, "bias-stopornot");
  // feat: word len.
  insertFeature(feat, "len-stopornot", seqlen);
  insertFeature(feat, "len-inv-stopornot", 1/(double)seqlen);
  // feat: entropy and frequency.
  double max_ent = -DBL_MAX, ave_ent = 0.0;
  double max_freq = -DBL_MAX, ave_freq = 0.0;
  for(size_t t = 0; t < seqlen; t++) {
    string word = seq.seq[t].word;
    double ent = 0, freq = 0;
    if(wordent->find(word) == wordent->end())
      ent = log(taglen); // no word, use maxent.
    else
      ent = (*wordent)[word];
    if(ent > max_ent) max_ent = ent;
    ave_ent += ent;

    if(wordfreq->find(word) == wordfreq->end())
      freq = log(corpus->total_words);
    else
      freq = (*wordfreq)[word];
    if(freq > max_freq) max_freq = freq;
    ave_freq += freq;
  }
  ave_ent /= seqlen;
  ave_freq /= seqlen;
  insertFeature(feat, "max-ent", max_ent);
  insertFeature(feat, "ave-ent", ave_ent);
  insertFeature(feat, "max-freq", max_freq);
  insertFeature(feat, "ave-freq", ave_freq);
  // feat: avg sample path length.
  int L = 3, l = 0;
  double dist = 0.0;
  shared_ptr<MarkovTreeNode> p = node;
  while(p->depth >= 1 && L >= 0) {
    auto pf = p->parent.lock();
    if(pf == nullptr) throw "MarkovTreeNode father is expired.";
    dist += p->tag->distance(*pf->tag);
    l++; L--;
    p = pf;
  }
  if(l > 0) dist /= l;
  insertFeature(feat, "len-sample-path", dist);
  // log probability of current sample in terms of marginal training stats.
  double logprob = tag_unigram_start[tag.tag[0]];
  for(size_t t = 1; t < seqlen; t++) 
    logprob += tag_bigram[tag.tag[t-1]][tag.tag[t]];
  insertFeature(feat, "log-prob-tag-bigram", logprob);
  return feat;
}

tuple<double, ParamPointer, ParamPointer, FeaturePointer> 
ModelAdaTree::logisticStop(shared_ptr<MarkovTreeNode> node, const Sentence& seq, const Tag& tag) {
  ParamPointer posgrad = makeParamPointer(), 
		neggrad = makeParamPointer();
  FeaturePointer feat = this->extractStopFeatures(node, seq, tag);
  double prob = logisticFunc(log(eps)-log(1-eps)+tag.score(feat)); 
  if(prob < 1e-3) prob = 1e-3;   // truncation, avoid too long transitions.
  else{
    mapUpdate<double, double>(*posgrad, *feat, (1-prob));
    mapUpdate<double, double>(*neggrad, *feat, -prob);
  }
  return make_tuple(prob, posgrad, neggrad, feat);
}

double ModelAdaTree::score(shared_ptr<MarkovTreeNode> node, const Tag& tag) {
  const Sentence* seq = tag.seq;
  double score = 0.0;
  for(int i = 0; i < tag.size(); i++) {
    score -= (tag.tag[i] != seq->tag[i]);
  }
  score -= m_c * max(0.0, node->depth-m_Tstar);
  return score;
}
