#include "policy.h"
#include "feature.h"
#include <boost/lexical_cast.hpp>

#define USE_FEAT_ENTROPY 0
#define USE_FEAT_CONDENT 1
#define USE_FEAT_ALL 0
#define USE_FEAT_BIAS 1
#define USE_ORACLE 0
#define USE_SELFAVOID 0 

#define USE_WINDOW 0

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
   lets_resp_reward(false),
   lets_inplace(vm["inplace"].as<bool>()), 
   model_unigram(nullptr), 
   param(makeParamPointer()), G2(makeParamPointer()) {
    // feature switch.
    split(featopt, vm["feat"].as<string>(), boost::is_any_of(" "));
    // init stats.
    if(isinstance<CorpusLiteral>(model->corpus)) {
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
    }
    system(("mkdir -p "+name).c_str());
    lg = shared_ptr<XMLlog>(new XMLlog(name+"/policy.xml"));  
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
          int pos = node->choice;
          node->gradient = model->sampleOne(*node->tag, pos);
          FeaturePointer feat = this->extractFeatures(node, pos);
          node->tag->mask[pos] += 1;
          node->tag->feat[pos] = feat;
          node->tag->resp[pos] = Tagging::score(this->param, feat);
//          node->tag->checksum[pos] = this->checksum(node, pos);    // compute checksum after sample.
          node->tag->checksum[pos] = 0;
         /* if(lets_resp_reward) {
            double resp = node->tag->resp[pos];
            test_thread_pool.lock();
            Tag tag(*node->tag);
            auto is_equal = [&] () {
              return (double)(tag.tag[pos] == tag.seq->tag[pos]); 
            };
            double reward_baseline = is_equal();
            model->sampleOne(tag, pos);
            double reward = is_equal();
            // double logR = reward - reward_baseline; 
            double logR = tag.reward[pos];
            test_resp_reward.push_back(make_pair(resp, logR));
            test_resp_RH.push_back(make_pair(resp, 1-reward_baseline));
            test_resp_RL.push_back(make_pair(resp, logR));
            if(isinstance<CorpusLiteral>(model->corpus)) {
              test_word_tag.push_back(make_tuple(resp, 1-reward_baseline, cast<TokenLiteral>(tag.seq->seq[pos])->word, tag.tag[pos]));
            }
            test_thread_pool.unlock();
          }*/
        }
        if(lets_inplace) {
          node->depth++;
        }else {
          node = addChild(node, *node->tag);
        }
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
    lg->begin("commit"); *lg << getGitHash() << endl; lg->end();  
    lg->begin("args");
      if(isinstance<CorpusLiteral>(corpus)) {
	lg->begin("wordent_mean");
	  *lg << wordent_mean << endl;
	lg->end();
	lg->begin("wordfreq_mean");
	  *lg << wordfreq_mean << endl;
	lg->end();
      }
    lg->end();
    corpus->retag(model->corpus);
    size_t count = 0;
    for(const SentencePtr seq : corpus->seqs) {
      if(count >= train_count) break;
      if(count % int(0.1 * min(train_count, corpus->seqs.size())) == 0) 
	cout << "\t\t " << (double)count/corpus->seqs.size()*100 << " %" << endl;
      if(verbose)
	lg->begin("example_"+to_string(count));
      MarkovTree tree;
      Tag tag(seq.get(), corpus, &rng, model->param);
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
      lg->begin("param");
      *lg << *param;
      lg->end(); // </param>
      if(verbose) {
	lg->end(); // </example>
      }
      count++;
    }
  }

  void Policy::trainKernel(ptr<Corpus> corpus) {
    corpus->retag(model->corpus);
    size_t count = 0;
    for(const SentencePtr seq : corpus->seqs) {
      if(count >= train_count) break;
      if(count % 1000 == 0) 
	cout << "\t\t " << (double)count/corpus->seqs.size()*100 << " %" << endl;
      MarkovTree tree;
      Tag tag(seq.get(), corpus, &rng, model->param);
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
    result->corpus->retag(model->corpus);
    result->nodes.resize(min(test_count, testCorpus->seqs.size()), nullptr);
    result->time = 0;
    result->wallclock = 0;
    test(result);
    return result;
  }
  
  void Policy::test(Policy::ResultPtr result) {
    cout << "> test " << endl;
    lg->begin("test");
    lg->begin("param");
    *lg << *param;
    lg->end(); // </param>
    this->testPolicy(result);
    lg->end(); // </test>
  }

  void Policy::testPolicy(Policy::ResultPtr result) {
    assert(result != nullptr);
    size_t count = 0;
    clock_t time_start = clock(), time_end;

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
    time_end = clock();
    double accuracy = (double)hit_count / pred_count;
    double recall = (double)hit_count / truth_count;
    result->time += (double)ave_time / count;
    result->wallclock += (double)(time_end - time_start) / CLOCKS_PER_SEC;
    lg->begin("time");
      cout << "time: " << result->time << endl;
      *lg << result->time << endl;
    lg->end(); // </time>
    lg->begin("wallclock");
      cout << "wallclock: " << result->wallclock << endl;
      *lg << result->wallclock << endl;
    lg->end(); // </wallclock>
    if(model->scoring == Model::SCORING_ACCURACY) {
      lg->begin("accuracy");
      *lg << accuracy << endl;
      cout << "acc: " << accuracy << endl;
      lg->end(); // </accuracy>
      result->score = accuracy;
    }else if(model->scoring == Model::SCORING_NER) {
      double f1 = 2 * accuracy * recall / (accuracy + recall);
      lg->begin("accuracy");
      *lg << f1 << endl;
      cout << "f1: " << f1 << endl;
      lg->end(); // </accuracy>
      result->score = f1;
    }
    // result->score = -1;
  }

  FeaturePointer Policy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = makeFeaturePointer();
    Tag& tag = *node->tag;
    const Sentence& seq = *tag.seq;
    size_t seqlen = tag.size();
    size_t taglen = model->corpus->tags.size();
    // bias.
    if(featoptFind("bias") || featoptFind("all")) 
      insertFeature(feat, "b");
    if(featoptFind("word-ent") || featoptFind("all")) {
      string word = cast<TokenLiteral>(seq.seq[pos])->word;
      // feat: entropy and frequency.
      if(wordent->find(word) == wordent->end())
        insertFeature(feat, "word-ent", log(taglen)-wordent_mean);
      else
        insertFeature(feat, "word-ent", (*wordent)[word]);
    }
    if(featoptFind("word-freq") || featoptFind("all")) {
      if(isinstance<TokenLiteral>(seq.seq[pos])) {
        string word = cast<TokenLiteral>(seq.seq[pos])->word;
        ptr<CorpusLiteral> corpus = cast<CorpusLiteral>(this->model->corpus);
        if(wordfreq->find(word) == wordfreq->end())
          insertFeature(feat, "word-freq", log(corpus->total_words)-wordfreq_mean);
        else
          insertFeature(feat, "word-freq", (*wordfreq)[word]);
        StringVector nlp = corpus->getWordFeat(word);
        for(const string wordfeat : *nlp) {
          if(wordfeat == word) continue; 
          string lowercase = word;
          transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
          if(wordfeat == lowercase) continue;
          if(wordfeat[0] == 'p' or wordfeat[0] == 's') continue;
          insertFeature(feat, wordfeat);
        }
      }
    }
    if(featoptFind("cond-ent") || featoptFind("all")) {
      insertFeature(feat, "cond-ent", node->tag->entropy[pos]);
    }
    if(featoptFind("cond-lhood") || featoptFind("all")) {
      insertFeature(feat, "cond-lhood", node->tag->sc[node->tag->tag[pos]]);
    }
    if(featoptFind("unigram-ent")) {
      if(model_unigram) {
        if(std::isnan(node->tag->entropy_unigram[pos])) {
          Tag tag(*node->tag);
          model_unigram->sampleOne(tag, pos);
          node->tag->entropy_unigram[pos] = tag.entropy[pos];
        }
        insertFeature(feat, "unigram-ent", node->tag->entropy_unigram[pos]);
      }
    }
    if(featoptFind("nb-modify")) {      // if a neighbor has been modified.
      if(node->tag->checksum[pos] != this->checksum(node, pos)) {
        insertFeature(feat, "nb-modify", 1.0);
      }
    }
    if(featoptFind("unigram-lhood")) {
      if(model_unigram) {
        if(node->tag->sc_unigram[pos].size() == 0) {
          Tag tag(*node->tag);
          model_unigram->sampleOne(tag, pos);
          node->tag->sc_unigram[pos] = tag.sc;
        }
        insertFeature(feat, "unigram-lhood", node->tag->sc_unigram[pos][node->tag->tag[pos]]);
      }
    }
    if(featoptFind("word-sig") || featoptFind("all")) {
      if(model->scoring == Model::SCORING_NER) { // tag inconsistency, such as B-PER I-LOC
        ptr<Corpus> corpus = model->corpus;
        string tg = corpus->invtags[node->tag->tag[pos]];
        if(pos >= 1) {
  	      string prev_tg = corpus->invtags[node->tag->tag[pos-1]];
  	      if(prev_tg[0] == 'B' and tg[0] == 'I' and tg.substr(1) != prev_tg.substr(1)) 
  	       insertFeature(feat, "bad");
        }
        if(pos < node->tag->size()-1) {
          string next_tg = corpus->invtags[node->tag->tag[pos+1]];
  	      if(next_tg[0] == 'I' and tg[0] == 'B' and tg.substr(1) != next_tg.substr(1)) 
  	        insertFeature(feat, "bad");
        }
      }
    }
    if(featoptFind("oracle")) {
      int oldval = node->tag->tag[pos];
      Tag temp(tag);
      model->sampleOne(temp, pos);          
      // insertFeature(feat, "oracle", -temp.sc[oldval]);
      insertFeature(feat, "oracle", temp.entropy[pos]);
    }
    if(featoptFind("self-avoid")) {
      insertFeature(feat, "self-avoid", node->tag->mask[pos]);
    }
    return feat;
  }

  double Policy::checksum(MarkovTreeNodePtr node, int pos) {
    size_t checksum = 0;
    int factorL = cast<ModelCRFGibbs>(model)->factorL;
    for(int p = max(0, pos-factorL); p <= min(pos+factorL, int(node->tag->size())); p++) {
      checksum = checksum * 13 + node->tag->timestamp[p];
    }
    return (double)checksum;
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
        node->tag->feat[pos] = feat;
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
  :CyclicPolicy(model, vm) { 
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
    cout << "cyclic value train (lets_resp_reward = " << lets_resp_reward << ")"  << endl;
    if(lets_resp_reward and K > 1 and thread_pool.numThreads() > 1) 
      throw "multithread environment cannot record reward.";
    if(lets_resp_reward) 
      resp_reward.clear();
    Policy::trainPolicy(corpus);
    if(lets_resp_reward) {
      const int fold = 10;
      const int fold_l[fold] = {0,5,10,15,20,25,26,27,28,29};
      auto logRespReward = [&] (vec<pair<double, double> > p) {
	for(const ROC& roc : getROC(fold_l, fold, p)) {
	  cout << roc.str() << endl;
	  *lg << roc.str() << endl;
	}
      };
      lg->begin("roc_R");
	logRespReward(resp_reward);
      lg->end(); // </prec_recall>
      lg->begin("roc_RL");
	logRespReward(resp_RL);
      lg->end(); // </prec_recall>
      lg->begin("roc_RH");
	logRespReward(resp_RH);
      lg->end(); // </prec_recall>
      if(verbose) {
	lg->begin("resp_reward");
	for(const pair<double, double>& p : resp_reward) {
	  *lg << p.first << " " << p.second << endl; 
	}
	lg->end();
      }
    }
  }

  void CyclicValuePolicy::testPolicy(Policy::ResultPtr result) {
    Policy::testPolicy(result);
    if(lets_resp_reward) {
      const int fold = 10;
      const int fold_l[fold] = {0,5,10,15,20,25,26,27,28,29};
      auto logRespReward = [&] (vec<pair<double, double> >& p) {
	cout << endl;
	for(const ROC& roc : getROC(fold_l, fold, p)) {
	  cout << roc.str() << endl;
	  *lg << roc.str() << endl;
	}
      };
      lg->begin("test_roc_R");
	logRespReward(test_resp_reward);
      lg->end(); // </prec_recall>
      lg->begin("test_roc_RL");
	logRespReward(test_resp_RL);
      lg->end(); // </prec_recall>
      lg->begin("test_roc_RH");
	logRespReward(test_resp_RH);
      lg->end(); // </prec_recall>
      lg->begin("test_word_tag");
      if(verbose) {
	assert(test_resp_reward.size() == test_resp_RL.size() 
	    and test_resp_reward.size() == test_resp_RH.size()
	    and test_resp_reward.size() == test_word_tag.size());
	auto compare = [] (std::tuple<double, double, string, int> a, std::tuple<double, double, string, int> b) {
	  return (get<0>(a) < get<0>(b));
	};
	sort(test_word_tag.begin(), test_word_tag.end(), compare); 
	lg->begin("resp");
	for(size_t t = 0; t < test_resp_reward.size(); t++) {
	  *lg << "resp " << get<0>(test_word_tag[t]) << " " 
	    << "reward " << test_resp_reward[t].second << " " 
	    << "RH "  
	    << get<1>(test_word_tag[t]) << " "
	    << "RL "  
	    << test_resp_RL[t].second << " "
	    << get<2>(test_word_tag[t]) << " "
	    << model->corpus->invtags[get<3>(test_word_tag[t])] << endl;
	}
	lg->end(); // </resp>
      }
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
	node->tag->feat[pos] = feat;
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

  // get precision-recall curve. 
  vec<Policy::ROC> Policy::getROC(const int fold_l[], const int fold, 
	std::vector<std::pair<double, double> >& resp_reward) {
    auto compare = [] (std::pair<double, double> a, std::pair<double, double> b) {
      return (a.first < b.first);
    };
    sort(resp_reward.begin(), resp_reward.end(), compare); 
    int truth = 0, total = resp_reward.size();
    for(const pair<double, double>& p : resp_reward) {
      truth += p.second > 0;
    }
    double cmax = resp_reward.back().first, cmin = resp_reward[0].first;
    vec<ROC> roc_list;
    int tr_count = 0, count = 0;
    for(const pair<double, double>& p : resp_reward) {
      tr_count += p.second > 0;
      count += 1;
      if(count > 1 && p.first-roc_list.back().threshold < (cmax-cmin) * 1e-4) continue;
      ROC roc; 
      roc.threshold = p.first;
      roc.TP = truth-tr_count;
      roc.TN = count-tr_count;
      roc.FP = total-count-roc.TP;
      roc.FN = count-roc.TN;
      roc.prec_sample = roc.TP / (roc.TP + roc.FP);
      roc.prec_stop = roc.TN / (roc.TN + roc.FN);
      roc.recall_sample = roc.TP / (roc.TP + roc.FN);
      roc.recall_stop = roc.TN / (roc.TN + roc.FP);
      roc_list.push_back(roc);
    }
    return roc_list;
  }

  ///////// MultiCyclicValueUnigramPolicy ////////////////////////////////
  MultiCyclicValueUnigramPolicy::MultiCyclicValueUnigramPolicy(ModelPtr model, 
      ModelPtr model_unigram, const po::variables_map& vm)
  : MultiCyclicValuePolicy(model, vm), model_unigram(model_unigram) { 
  }

  FeaturePointer MultiCyclicValueUnigramPolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = MultiCyclicValuePolicy::extractFeatures(node, pos);
    Tag tag(*node->tag);
    int oldval = tag.tag[pos];
    model_unigram->sampleOne(tag, pos);
    int pass = node->time_stamp / node->tag->size();
    // insertFeature(feat, boost::lexical_cast<string>(pass) + "-unigram", -tag.sc[oldval]);
    insertFeature(feat, boost::lexical_cast<string>(pass) + "-unigram", tag.entropy[pos]);
    return feat;
  }

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
	node->time_stamp = i;
	size_t pos = i % node->tag->size();
	FeaturePointer feat = this->extractFeatures(node, pos);
	double resp = Tagging::score(param, feat);
	node->tag->resp[pos] = resp;
	node->tag->feat[pos] = feat;
	/*if(lets_resp_reward) {
	  test_thread_pool.lock();
	  Tag tag(*node->tag);
	  auto is_equal = [&] () {
	    return (double)(tag.tag[pos] == tag.seq->tag[pos]); 
	  };
	  double reward_baseline = is_equal();
	  model->sampleOne(tag, pos);
	  double reward = is_equal();
	  // double logR = reward - reward_baseline; 
	  double logR = tag.reward[pos];
	  test_resp_reward.push_back(make_pair(resp, logR));
	  test_resp_RH.push_back(make_pair(resp, 1-reward_baseline));
	  test_resp_RL.push_back(make_pair(resp, logR));
	  if(isinstance<CorpusLiteral>(model->corpus)) {
	    test_word_tag.push_back(make_tuple(resp, 1-reward_baseline, cast<TokenLiteral>(tag.seq->seq[pos])->word, tag.tag[pos]));
	  }
	  test_thread_pool.unlock();
	}*/
	if(resp > c) { 
	  node->tag->mask[pos] = 1;
	  return pos;
	}else{
	  node->tag->mask[pos] = 0;
	}
      }
      // node->time_stamp = i;
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
	  // double logR = reward - reward_baseline; 
	  double logR = node->tag->reward[i];
	  FeaturePointer feat = this->extractFeatures(node, i);   
	  double resp = Tagging::score(param, feat);
	  if(lets_resp_reward) {
	    thread_pool.lock();
	    resp_reward.push_back(make_pair(resp, logR));
	    resp_RH.push_back(make_pair(resp, 1-reward_baseline));
	    resp_RL.push_back(make_pair(resp, logR));
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
    size_t pass = 0;
    while(true) {
      if(node->time_stamp >= (pass+1) * node->tag->size()-1 || node->children.size() == 0) {
	lg->begin("pass_"+boost::lexical_cast<string>(node->time_stamp / node->tag->size()));
	  lg->begin("tag"); *lg << node->tag->str() << endl; lg->end();
	  for(size_t i = 0; i < node->tag->size(); i++) {
	    lg->begin("feat"); 
	    if(node->tag->feat.size() > i and node->tag->feat[i]) {
	      *lg << *node->tag->feat[i] << endl;
	    }else{
	      *lg << *this->extractFeatures(node, i) << endl;
	    }
	    lg->end(); // </feat> 
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
	pass = int(node->time_stamp / node->tag->size()) + 1;
      }
      if(node->children.size() > 0)
	node = node->children[0]; // take final sample.
      else
	break;
    }
    lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
    lg->begin("truth"); *lg << node->tag->seq->str() << endl; lg->end();
  }

  //////// RandomScanPolicy ////////////////////////////////////////////
  RandomScanPolicy::RandomScanPolicy(ModelPtr model, const boost::program_options::variables_map& vm) 
   :Policy(model, vm), 
    Tstar(vm["Tstar"].as<double>()), 
    windowL(vm["windowL"].as<int>()) {

  }

  int RandomScanPolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = -1;
    int pos = 0;
    if(node->depth < node->tag->size()) {
      pos = node->depth;
      node->tag->mask[pos] = 1;
    }else{
      if(node->depth > Tstar * node->tag->size()) 
        return -1;
      vec<double> resp = node->tag->resp;
      logNormalize(&resp[0], resp.size());
      objcokus* rng = node->tag->rng;
      pos = rng->sampleCategorical(&resp[0], resp.size());
      node->tag->mask[pos] += 1;
    }
    FeaturePointer feat = this->extractFeatures(node, pos);
    node->tag->feat[pos] = feat;
    node->tag->resp[pos] = Tagging::score(this->param, feat);
    node->time_stamp++;
    return pos;
  }

  FeaturePointer RandomScanPolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = Policy::extractFeatures(node, pos);
#if USE_WINDOW == 1
    for(int p = pos-windowL; p <= pos+windowL; p++) {
      if(p == pos || p < 0 || p >= node->tag->size()) continue;
      FeaturePointer this_feat = Policy::extractFeatures(node, p);
      for(pair<string, double>& f : *this_feat) {
	insertFeature(feat, "w"+boost::lexical_cast<string>(p-pos)+"-"
			      +f.first, f.second);
      }
    }
#endif
    return feat;
  }

  void RandomScanPolicy::sample(int tid, MarkovTreeNodePtr node) {
    node->depth = 0;
    node->choice = -1;
    try{
      node->tag->rng = &thread_pool.rngs[tid];
      for(size_t i = 0; i < node->tag->size(); i++) {
	model->sampleOne(*node->tag, i);
	node->tag->mask[i] = 1;
      }
      size_t seqlen = node->tag->size();
      node->depth = seqlen;
      node->gradient = makeParamPointer();
      while(node->depth < Tstar * seqlen) {		
	vec<double> reward(seqlen);
	vec<double> resp(seqlen);
	vec<FeaturePointer> feat(seqlen);
	auto is_equal = [] (Tag& tag, int i) {
	  return (double)(tag.tag[i] == tag.seq->tag[i]); 
	};
	double norm1 = -DBL_MAX, norm2 = -DBL_MAX;
	for(size_t i = 0; i < seqlen; i++) {
	  double reward_baseline = is_equal(*node->tag, i);
	  Tag tag(*node->tag);
	  model->sampleOne(tag, i);
	  reward[i] = is_equal(tag, i) - reward_baseline;
	  feat[i] = extractFeatures(node, i);
	  resp[i] = Tagging::score(param, feat[i]);
	  norm1 = logAdd(norm1, reward[i] + resp[i]);
	  norm2 = logAdd(norm2, resp[i]);
	}
	for(size_t i = 0; i < seqlen; i++) {
	  mapUpdate(*node->gradient, *feat[i], exp(reward[i] + resp[i] - norm1));
	  mapUpdate(*node->gradient, *feat[i], -exp(resp[i] - norm2));
	}
	logNormalize(&resp[0], seqlen);
	int pos = node->tag->rng->sampleCategorical(&resp[0], seqlen);
	model->sampleOne(*node->tag, pos);
	node->tag->mask[pos] += 1;
	node->depth += 1;
      }
      node->log_weight = 0;
    }catch(const char* ee) {
      cout << "error: " << ee << endl;
    }
  }

  //////// LockdownPolicy ////////////////////////////
  LockdownPolicy::LockdownPolicy(ModelPtr model, const boost::program_options::variables_map& vm)
   :Policy(model, vm), 
    T(vm["T"].as<size_t>()),
    c(vm["c"].as<double>())
  {
      
  }
  
  FeaturePointer LockdownPolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
    FeaturePointer feat = Policy::extractFeatures(node, pos);
    insertFeature(feat, "#sp", node->tag->mask[pos]);
    return feat;
  }

  int LockdownPolicy::policy(MarkovTreeNodePtr node) {
    if(node->depth == 0) node->time_stamp = 0;
    size_t count = node->time_stamp;
    int seqlen = node->tag->size();
    for(; count < node->time_stamp + seqlen; count++) {
      int pos = count % seqlen;

      /* TODO: compute feat on demand */
      if(!std::isnan(node->tag->checksum[pos])) {
        FeaturePointer feat = this->extractFeatures(node, pos);
        node->tag->feat[pos] = feat;
        node->tag->resp[pos] = Tagging::score(param, feat);
        if(lets_resp_reward) {
          double resp = node->tag->resp[pos];
          test_thread_pool.lock();
          // collect reward, slow, for debug only.
//          Tag tag(*node->tag);
//          auto is_equal = [&] () {
//            return (double)(tag.tag[pos] == tag.seq->tag[pos]); 
//          };
//          double reward_baseline = is_equal();
//          model->sampleOne(tag, pos);
//          double reward = is_equal();
//          // double logR = reward - reward_baseline; 
//          double logR = tag.reward[pos];
//          test_resp_RH.push_back(make_pair(resp, 1-reward_baseline));
//          test_resp_RL.push_back(make_pair(resp, logR));
//          if(isinstance<CorpusLiteral>(model->corpus)) {
//            test_word_tag.push_back(make_tuple(resp, 1-reward_baseline, cast<TokenLiteral>(tag.seq->seq[pos])->word, tag.tag[pos]));
//          }
          
          // do not collect reward.
          double logR = 0;
          test_resp_reward.push_back(make_pair(resp, logR));
          test_thread_pool.unlock();
        }
      }
      int mk = node->tag->mask[pos];
      if(node->tag->resp[pos] > c and 
          node->tag->mask[pos] <= T) {
        node->time_stamp = count+1;
        return pos;
      }
    }
    node->time_stamp = count;
    return -1;
  }

  void LockdownPolicy::sample(int tid, MarkovTreeNodePtr node) {
    node->depth = 0;
    node->choice = -1;
    try{
      node->tag->rng = &thread_pool.rngs[tid];
      for(size_t i = 0; i < node->tag->size(); i++) {
        model->sampleOne(*node->tag, i);
        node->tag->mask[i] = 1;
        node->tag->checksum[i] = this->checksum(node, i);    // compute checksum after sample.
      }
      node->gradient = makeParamPointer();
      for(size_t t = 1; t < T; t++) {
        for(size_t i = 0; i < node->tag->size(); i++) {
          node->time_stamp = t * node->tag->size() + i;
          /* extract features */
          FeaturePointer feat = this->extractFeatures(node, i);
          double resp = Tagging::score(param, feat);
          /* estimate reward */
          auto is_equal = [&] () {
            return (double)(node->tag->tag[i] == node->tag->seq->tag[i]); 
          };
          double reward_baseline = is_equal();
          model->sampleOne(*node->tag, i);
          node->tag->mask[i] += 1;
          node->tag->checksum[i] = this->checksum(node, i);    // compute checksum after sample.
          double reward = is_equal();
//          double logR = reward - reward_baseline; 
          double logR = node->tag->reward[i];
          // logR *= 100; // scale for convenience.
          if(lets_resp_reward) {
            resp_reward.push_back(make_pair(resp, logR));
          }
          /* update meta-model */
          // mapUpdate(*node->gradient, *feat, 2 * (logR - resp)); 
          resp = logisticFunc(resp);
          if(logR > 0) {
            mapUpdate(*node->gradient, *feat, (1-resp));
          }else{
            mapUpdate(*node->gradient, *feat, -resp);
          }
        }
      }
      node->log_weight = 0;
    }catch(const char* ee) {
      cout << "error: " << ee << endl;
    }
  }
}
