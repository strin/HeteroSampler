#include "stop.h"

namespace po = boost::program_options;
using namespace std;

Stop::Stop(ModelPtr model, const po::variables_map& vm) 
:model(model), T(vm["T"].as<int>()), B(vm["B"].as<int>()), 
 K(vm["K"].as<int>()), name(vm["name"].as<string>()), 
	thread_pool(vm["numThreads"].as<int>(), 
		      [&] (int tid, MarkovTreeNodePtr node) {
			this->sample(tid, node);
		      }),
  stop_data(makeStopDataset()), param(makeParamPointer()), 
  G2(makeParamPointer()), eta(eta) {
  rng.seedMT(time(0));
  // init const environment.
  wordent = model->tagEntropySimple();
  wordfreq = model->wordFrequencies();
  auto tag_bigram_unigram = model->tagBigram();
  tag_bigram = tag_bigram_unigram.first;
  tag_unigram_start = tag_bigram_unigram.second;
}

StopDatasetPtr Stop::explore(const Sentence& seq) {    
  Tag tag(&seq, model->corpus, &rng, param);
  MarkovTree tree;
  thread_pool.addWork(tree.root);
  thread_pool.waitFinish();
  return tree.generateStopDataset(tree.root, 0);
}

void Stop::sample(int tid, MarkovTreeNodePtr node) {
  while(true) {
    model->sample(*node->tag, 1);
    node->stop_feat = extractStopFeatures(node);
    node->log_weight = this->score(node);
    if(node->depth == T) {
      node->log_prior_weight = 1;
      return;
    }else if(node->depth <= B) { // split node, aggregate data.
      node->compute_stop = true;
      for(size_t k = 0; k < K; k++) {
	node->children.push_back(makeMarkovTreeNode(node, *node->tag));
	thread_pool.addWork(node->children.back()); 
      }
      return;
    }else{
      node->children.push_back(makeMarkovTreeNode(node, *node->tag));
      node = node->children.back();
    }
  }
}

double Stop::score(shared_ptr<MarkovTreeNode> node) {
  Tag& tag = *node->tag;
  const Sentence* seq = tag.seq;
  double score = 0.0;
  for(int i = 0; i < tag.size(); i++) {
    score -= (tag.tag[i] != seq->tag[i]);
  }
  return score;
}

FeaturePointer Stop::extractStopFeatures(MarkovTreeNodePtr node) {
  FeaturePointer feat = makeFeaturePointer();
  Tag& tag = *node->tag;
  const Sentence& seq = *tag.seq;
  size_t seqlen = tag.size();
  size_t taglen = model->corpus->tags.size();
  // feat: bias.
  (*feat)["bias-stopornot"] = 1.0;
  // feat: word len.
  (*feat)["len-stopornot"] = (double)seqlen; 
  (*feat)["len-inv-stopornot"] = 1/(double)seqlen;
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
      freq = log(model->corpus->total_words);
    else
      freq = (*wordfreq)[word];
    if(freq > max_freq) max_freq = freq;
    ave_freq += freq;
  }
  ave_ent /= seqlen;
  ave_freq /= seqlen;
  (*feat)["max-ent"] = max_ent;
  (*feat)["max-freq"] = max_freq;
  (*feat)["ave-ent"] = ave_ent;
  (*feat)["ave-freq"] = ave_freq;
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
  (*feat)["len-sample-path"] = dist;
  // log probability of current sample in terms of marginal training stats.
  double logprob = tag_unigram_start[tag.tag[0]];
  for(size_t t = 1; t < seqlen; t++) 
    logprob += tag_bigram[tag.tag[t-1]][tag.tag[t]];
  (*feat)["log-prob-tag-bigram"] = logprob;
  return feat;
}

void Stop::run(const Corpus& corpus) {
  // aggregate dataset.
  Corpus retagged(corpus);
  retagged.retag(corpus); // use training taggs. 
  StopDatasetPtr stop_data = makeStopDataset();
  for(const Sentence& seq : corpus.seqs) {
    mergeStopDataset(stop_data, this->explore(seq)); 
  }
  stop_data_log = shared_ptr<XMLlog>(new XMLlog("stopdata.xml"));
  logStopDataset(stop_data, *this->stop_data_log);
  stop_data_log->end(); 
  // train logistic regression. 
  StopDatasetKeyContainer::iterator key_iter;
  StopDatasetValueContainer::iterator R_iter;
  StopDatasetValueContainer::iterator epR_iter;
  StopDatasetSeqContainer::iterator seq_iter;
  for(key_iter = std::get<0>(*stop_data).begin(), R_iter = std::get<1>(*stop_data).begin(),
      epR_iter = std::get<2>(*stop_data).begin(), seq_iter = std::get<3>(*stop_data).begin();
      key_iter != std::get<0>(*stop_data).end() && R_iter != std::get<1>(*stop_data).end() 
      && epR_iter != std::get<2>(*stop_data).end() && seq_iter != std::get<3>(*stop_data).end(); 
      key_iter++, R_iter++, epR_iter++, seq_iter++) {
    double resp = logisticFunc(::score(param, *key_iter));
    ParamPointer gradient = makeParamPointer();
    mapUpdate<double, double>(*gradient, **key_iter, resp * (1 - resp) * (*R_iter - *epR_iter));

    for(const pair<string, double>& p : *gradient) {
      mapUpdate(*G2, p.first, p.second * p.second);
      mapUpdate(*param, p.first, eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
    }
  }
}


void Stop::test(const Corpus& testCorpus) {
}
