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
      test_thread_pool(vm["numThreads"].as<int>(), 
		    [&] (int tid, MarkovTreeNodePtr node) {
		      this->sampleTest(tid, node);
		    }),
stop_data(makeStopDataset()), param(makeParamPointer()), 
G2(makeParamPointer()), eta(vm["eta"].as<double>()),
c(vm["c"].as<double>()), train_count(vm["trainCount"].as<size_t>()),
test_count(vm["testCount"].as<size_t>()),
lets_adaptive(vm["adaptive"].as<bool>()), 
iter(vm["iter"].as<size_t>()),
Tstar(vm["Tstar"].as<double>()) {
  system(("mkdir "+name).c_str());
  lg = shared_ptr<XMLlog>(new XMLlog(name+"/stop.xml"));
  lg->begin("args");
  lg->begin("c"); *lg << c << endl; lg->end();
  lg->begin("T"); *lg << T << endl; lg->end();
  lg->begin("B"); *lg << B << endl; lg->end();
  lg->begin("K"); *lg << K << endl; lg->end();
  lg->end();
  rng.seedMT(time(0));
  // init const environment.
  wordent = model->tagEntropySimple();
  wordfreq = model->wordFrequencies();
  auto tag_bigram_unigram = model->tagBigram();
  tag_bigram = tag_bigram_unigram.first;
  tag_unigram_start = tag_bigram_unigram.second;
}

Stop::~Stop() {
  while(lg->depth() > 0)
    lg->end();
}

StopDatasetPtr Stop::explore(const Sentence& seq) {    
  MarkovTree tree;
  tree.root->tag = makeTagPtr(&seq, model->corpus, &rng, model->param);
  thread_pool.addWork(tree.root);
  thread_pool.waitFinish();
  return tree.generateStopDataset(tree.root, 0);
}

void Stop::sample(int tid, MarkovTreeNodePtr node) {
  while(true) {
    node->tag->rng = &thread_pool.rngs[tid];
    model->sample(*node->tag, 1);
    node->stop_feat = extractStopFeatures(node);
    node->log_weight = this->score(node) - c * node->depth;
    node->log_prior_weight = -DBL_MAX;
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

void Stop::sampleTest(int tid, MarkovTreeNodePtr node) {
  node->depth = 0;
  while(true) { // maximum level of inference = T.
    node->tag->rng = &test_thread_pool.rngs[tid];
    model->sample(*node->tag, 1);
    if(lets_adaptive) {
      node->stop_feat = extractStopFeatures(node);
      mapUpdate(*node->stop_feat, *mean_feat, -1);
      for(const pair<string, double>& p : *node->stop_feat) {
	(*node->stop_feat)[p.first] /= (*std_feat)[p.first];
      }
      node->resp = logisticFunc(::score(param, node->stop_feat));
    }
    if(node->depth == T || (lets_adaptive &&  
      test_thread_pool.rngs[tid].random01() < node->resp)) { // stop.
      return;
    }else{
      node->children.push_back(makeMarkovTreeNode(node, *node->tag));
      node = node->children.back();
    }
  }
}

double Stop::score(shared_ptr<MarkovTreeNode> node) {
  // strategy 1. using distance w.r.t target.
  Tag& tag = *node->tag;
  const Sentence* seq = tag.seq;
  double score = 0.0;
  for(int i = 0; i < tag.size(); i++) {
    score -= (tag.tag[i] != seq->tag[i]);
  }
  // strategy 2. use log-score.
  //double score = model->score(*node->tag);
  return score;
}

FeaturePointer Stop::extractStopFeatures(MarkovTreeNodePtr node) {
  FeaturePointer feat = makeFeaturePointer();
  Tag& tag = *node->tag;
  const Sentence& seq = *tag.seq;
  size_t seqlen = tag.size();
  size_t taglen = model->corpus->tags.size();
  // feat: bias.
  (*feat)["bias-stop"] = 1.0; 
  // feat: # special symbols. (like CD , ;)
  size_t num_cd = 0, num_sym = 0;
  for(size_t t = 0; t < seqlen; t++) {
    string tg_str = model->corpus->invtags.find(tag.tag[t])->second;
    if(tg_str == "CD") 
      num_cd++;
    if(tg_str == ";" || tg_str == "." || tg_str == "," || tg_str == ":"
      || tg_str == "\"") 
      num_sym++;
  }
  (*feat)["per-cd"] = num_cd/(double)seqlen;
  (*feat)["per-sym"] = num_sym/(double)seqlen;
  // feat: word len.
  (*feat)["len-stop"] = (double)seqlen; 
  (*feat)["len-inv-stop"] = 1/(double)seqlen;
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
  //(*feat)["len-sample-path"] = dist;
  // log probability of current sample in terms of marginal training stats.
  double logprob = tag_unigram_start[tag.tag[0]];
  for(size_t t = 1; t < seqlen; t++) 
    logprob += tag_bigram[tag.tag[t-1]][tag.tag[t]];
  (*feat)["log-prob-tag-bigram"] = logprob;
  return feat;
}

void Stop::run(const Corpus& corpus) {
  if(!lets_adaptive) return;
  cout << "> run " << endl;
  // aggregate dataset.
  Corpus retagged(corpus);
  retagged.retag(*model->corpus); // use training taggs. 
  StopDatasetPtr stop_data = makeStopDataset();
  int count = 0;
  lg->begin("train");
  lg->begin("example");
  for(const Sentence& seq : corpus.seqs) { 
    if(count >= train_count) break; 
    StopDatasetPtr dataset = this->explore(seq);
    mergeStopDataset(stop_data, this->explore(seq)); 
    count++;
  }
  double N = (double)get<0>(*stop_data).size();
  // remove mean.
  StopDatasetKeyContainer::iterator key_iter;
  StopDatasetValueContainer::iterator R_iter;
  StopDatasetValueContainer::iterator epR_iter;
  StopDatasetSeqContainer::iterator seq_iter;
  mean_feat = makeFeaturePointer();
  for(key_iter = get<0>(*stop_data).begin(); key_iter != get<0>(*stop_data).end(); key_iter++) {
    mapUpdate(*mean_feat, **key_iter); 
  }
  mapDivide(*mean_feat, N);
  (*mean_feat)["bias-stop"] = 0;  // do not substract mean.
  for(key_iter = get<0>(*stop_data).begin(); key_iter != get<0>(*stop_data).end(); key_iter++) {
    mapUpdate(**key_iter, *mean_feat, -1); 
  }
  std_feat = makeFeaturePointer();
  for(key_iter = get<0>(*stop_data).begin(); key_iter != get<0>(*stop_data).end(); key_iter++) {
    for(const pair<string, double>& p : **key_iter) {
      mapUpdate(*std_feat, p.first, p.second * p.second);
    }
  }
  for(const pair<string, double>& p : *std_feat) {
    (*std_feat)[p.first] = sqrt(p.second/(N-1));
  }
  (*std_feat)["bias-stop"] = 1.0;
  for(key_iter = get<0>(*stop_data).begin(); key_iter != get<0>(*stop_data).end(); key_iter++) {
    for(const pair<string, double>& p : **key_iter) {
      (**key_iter)[p.first] /= (*std_feat)[p.first];
    }
  }
  (*param)["bias-stop"] = -log(Tstar - 1);
  for(size_t it = 0; it < iter; it++) {
    count = 0;
    double sc = 0;
    double nob_sum = -DBL_MAX;
    double aveT = 0.0;
    for(key_iter = get<0>(*stop_data).begin(), R_iter = get<1>(*stop_data).begin(),
	epR_iter = get<2>(*stop_data).begin(), seq_iter = get<3>(*stop_data).begin();
	key_iter != get<0>(*stop_data).end() && R_iter != get<1>(*stop_data).end() 
	&& epR_iter != get<2>(*stop_data).end() && seq_iter != get<3>(*stop_data).end(); 
	key_iter++, R_iter++, epR_iter++, seq_iter++) {
      // train logistic regression. 
      lg->begin("example_"+to_string(count));
      *lg << **key_iter << endl;
      double resp = logisticFunc(::score(param, *key_iter));
      cout << "logit " << resp << endl;
      aveT += 1/resp/N;

      (**key_iter)["bias-stop"] = 0;
      double resp_nob = logisticFunc(::score(param, *key_iter));
      (**key_iter)["bias-stop"] = 1;
      nob_sum = logAdd(nob_sum, -resp_nob);

      double R_max = max(*R_iter, *epR_iter);
      // double sc = log(resp * exp(*R_iter - R_max) + (1 - resp) * exp(*epR_iter - R_max)) + R_max;
      sc += resp * (*R_iter) + (1 - resp) * (*epR_iter);
      *lg << "R: " << *R_iter << endl;
      *lg << "epR: " << *epR_iter << endl;
      *lg << "resp: " << ::score(param, *key_iter) << endl;
      *lg << "score: " << sc << endl;
      // mapUpdate<double, double>(*gradient, **key_iter, resp * (1 - resp) * (exp(*R_iter - R_max) - exp(*epR_iter - R_max)) / (exp(sc - R_max)));
      ParamPointer gradient = makeParamPointer();
      (*gradient)["bias-stop"] = 0.0;
      mapUpdate<double, double>(*gradient, **key_iter, resp * (1 - resp) * (*R_iter - *epR_iter));
      for(const pair<string, double>& p : *gradient) {
	mapUpdate(*G2, p.first, p.second * p.second);
	mapUpdate(*param, p.first, eta * p.second/sqrt(1e-4 + (*G2)[p.first]));
      }
      lg->end();
      count++;
    }
    (*param)["bias-stop"] = -log(N * (Tstar - 1)) + nob_sum; 
    cout << "score = " << sc << " , " << aveT << endl;
    /*for(const pair<string, double>& p : *gradient) {
      mapUpdate(*param, p.first, eta * p.second);
    }*/
  }
  lg->end(); //</example> 
  lg->end(); //</train>
  stop_data_log = shared_ptr<XMLlog>(new XMLlog(name+"/stopdata.xml"));
  logStopDataset(stop_data, *this->stop_data_log);
  stop_data_log->end(); 
}


double Stop::test(const Corpus& testCorpus) {
  cout << "> test " << endl;
  Corpus retagged(testCorpus);
  retagged.retag(*model->corpus); // use training taggs. 
  vector<MarkovTreeNodePtr> result;
  int count = 0;
  lg->begin("test");
  for(const Sentence& seq : retagged.seqs) {
    if(count >= test_count) 
       break;
    MarkovTreeNodePtr node = makeMarkovTreeNode(nullptr);
    node->tag = makeTagPtr(&seq, model->corpus, &rng, model->param);
    test_thread_pool.addWork(node);
    result.push_back(node);
    count++;
  }
  test_thread_pool.waitFinish();
  int hit_count = 0, pred_count = 0;
  count = 0;
  lg->begin("param");
    *lg << *param << endl;
  lg->end();
  lg->begin("example");
  for(MarkovTreeNodePtr node : result) {
    while(node->children.size() > 0) node = node->children[0];
    lg->begin("example_"+to_string(count));
      lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
      lg->begin("truth"); *lg << node->tag->seq->str() << endl; lg->end();
      lg->begin("tag"); *lg << node->tag->str() << endl; lg->end();
      if(lets_adaptive) {
	lg->begin("resp"); *lg << node->resp << endl; lg->end();
	lg->begin("feat"); *lg << *node->stop_feat << endl; lg->end();
      }
      int hits = 0;
      for(size_t i = 0; i < node->tag->size(); i++) {
	if(node->tag->tag[i] == node->tag->seq->tag[i]) {
	  hits++;
	}
      }
      lg->begin("dist"); *lg << node->tag->size()-hits << endl; lg->end();
      hit_count += hits;
      pred_count += node->tag->size();
    lg->end();
    count++;
  }
  lg->end(); //</example>
  double acc = (double)hit_count/pred_count;
  lg->begin("accuracy");
    *lg << acc << endl;
  lg->end();
  lg->end(); // </test>
  return acc;
}
