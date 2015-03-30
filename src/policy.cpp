#include "policy.h"
#include "corpus_ising.h"
#include "feature.h"
#include <boost/lexical_cast.hpp>

#define REWARD_LHOOD 1
#define REWARD_ACCURACY 2
#define REWARD_SCHEME REWARD_LHOOD      // LIKELIHOOD, ACCURACY

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
 lets_resp_reward(false),
 lazymax_lag(-1),
 model_unigram(nullptr),
 name(vm["output"].empty() ? "" : vm["output"].as<string>()),
 learning(vm["learning"].empty() ? "logistic" : vm["learning"].as<string>()),
 mode_reward(vm["reward"].empty() ? 0 : vm["reward"].as<int>()),
 mode_oracle(vm["oracle"].empty() ? 0 : vm["oracle"].as<int>()),
 rewardK(vm["rewardK"].empty() ? 1 : vm["rewardK"].as<int>()),
 K(vm["K"].empty() ? 1 : vm["K"].as<size_t>()),
 eta(vm["eta"].empty() ? 1 : vm["eta"].as<double>()),
 train_count(vm["trainCount"].empty() ? -1 : vm["trainCount"].as<size_t>()),
 test_count(vm["testCount"].empty() ? -1 : vm["testCount"].as<size_t>()),
 verbose(vm["verbose"].empty() ? false : vm["verbose"].as<bool>()),
 Q(vm["Q"].empty() ? 1 : vm["Q"].as<size_t>()),
 lets_inplace(vm["inplace"].empty() ? true : vm["inplace"].as<bool>()),
 lets_lazymax(vm["lets_lazymax"].empty() ? false : vm["lets_lazymax"].as<bool>()),
 init_method(vm["init"].empty() ? "" : vm["init"].as<string>()),
 param(makeParamPointer()), G2(makeParamPointer()) {
   // parse options
   if(!vm["log"].empty() and vm["log"].as<string>() != "") {
     try{
       lg = std::make_shared<XMLlog>(vm["log"].as<string>());
     }catch(char const* error) {
       cout << "error: " << error << endl;
       lg = std::make_shared<XMLlog>();
     }
   }else{
     lg = std::make_shared<XMLlog>();
   }

  // feature switch
  split(featopt, vm["feat"].as<string>(), boost::is_any_of(" "));
  split(verbose_opt, vm["verbosity"].as<string>(), boost::is_any_of(" "));

  // init stats
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

  int sysres = system(("mkdir -p "+name).c_str());
}

Policy::~Policy() {
  while(lg != nullptr and lg->depth() > 0)
    lg->end();
}

double Policy::reward(MarkovTreeNodePtr node) {
  throw "function deprecated";
  // auto& tag = *cast<Tag>(node->gm);
  // // Tag& tag = *node->gm;
  // const Instance* seq = tag.seq;
  // double score = 0.0;
  // for(int i = 0; i < tag.size(); i++) {
  //   score -= (tag.tag[i] != seq->tag[i]);
  // }
  // return score;
}

double Policy::delayedReward(MarkovTreeNodePtr node, int depth, int maxdepth, vec<int>& actions) {
  int id;
  if(depth < actions.size()) { // take specified action.
    id = actions[depth];
  }else{ // sample new action.
    if(depth == 0) { //sample uniformly.
      id = int(rng.random01() * (1 - 1e-8) * node->gm->size());
    }else{
      vec<int> blanket = model->markovBlanket(*node->gm, actions[depth-1]);
      if(blanket.size() == 0) {
	id = actions[depth-1];
      }else{
	id = blanket[int(rng.random01() * (1 - 1e-8) *  blanket.size())];
      }
    }
    actions.push_back(id);
  }
  int num_label = node->gm->numLabels(id);
  int oldval = node->gm->getLabel(id);
  double R = 0;
  model->sampleOne(*node->gm, rng, id, false);
  R = node->gm->sc[node->gm->getLabel(id)] - node->gm->sc[oldval];
  if(depth < maxdepth) {
    R += this->delayedReward(node, depth+1, maxdepth, actions);
  }
  node->gm->setLabel(id, oldval);
  return R;
}

double Policy::sampleDelayedReward(MarkovTreeNodePtr node, int id, int maxdepth, int rewardK) {
  vec<int> actions_u(1);
  actions_u[0] = id;
  double R_u = this->delayedReward(node, 0, maxdepth, actions_u);
  vec<int> actions_v(actions_u.begin() + 1, actions_u.end());
  double R_v = this->delayedReward(node, 0, maxdepth-1, actions_v);
  return R_u - R_v;
}

void Policy::sample(int tid, MarkovTreeNodePtr node) {
  this->sampleTest(tid, node);
}

void Policy::sampleTest(int tid, MarkovTreeNodePtr node) {
  node->depth = 0;
  objcokus& rng = test_thread_pool.rngs[tid];
  node->gm->rng = &rng;
  try{
    if(this->init_method == "iid") {
      for(size_t pos = 0; pos < node->gm->size(); pos++) {
        model->sampleOneAtInit(*node->gm, rng, pos);
        node->depth++;
        node->gm->mask[pos] += 1;
        node->gm->checksum[pos] = 0; // WARNING: a hack.
      }
      node->max_gm = model->copySample(*node->gm);
    }
    while(true) {
      if(node->depth >= POLICY_MARKOV_CHAIN_MAXDEPTH)
        throw "Policy Chain reaches maximum depth.";
      node->choice = this->policy(node);
      if(node->choice == -1) {
        /* strategy 1 */
//          node->log_weight = this->reward(node);
        /* strategy 2 */
        node->log_weight = model->score(*node->gm);
        node->gradient = makeParamPointer();
        break;
      }else{
        node->log_weight = -DBL_MAX;
        int pos = node->choice;
        this->sampleOne(node, rng, pos);
        this->updateResp(node, rng, pos, nullptr);
       /* if(lets_resp_reward) {
          double resp = node->gm->resp[pos];
          test_thread_pool.lock();
          Tag tag(*node->gm);
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
  if(Q == 0) { // notraining is needed.
    // set all feature weights to 1.
    for(const string& opt : featopt) {
      (*param)[opt] = 1;
    }
  }
  lg->end(); // </train>
  cout << "> completed." << endl;
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
  examples.clear();
  for(const SentencePtr seq : corpus->seqs) {
    if(count >= train_count) break;
    // cout << corpus->seqs.size() << endl;
    size_t display_lag = int(0.1 * min(train_count, corpus->seqs.size()));
    if(display_lag == 0 or count % display_lag == 0)
      cout << "\t\t " << (double)count/corpus->seqs.size()*100 << " %" << endl;
    if(verbose)
      lg->begin("example_"+to_string(count));
    MarkovTree tree;
//      Tag tag(seq.get(), corpus, &rng, model->param);
    ptr<GraphicalModel> gm = model->makeSample(*seq, model->corpus, &rng);
    tree.root->log_weight = -DBL_MAX;
    tree.root->model = this->model;
    for(size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = addChild(tree.root, *gm);
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

//      /* strategy 1. use Markov Tree to compute grad */
//      this->gradientPolicy(tree);

    /* strategy 2. compute gradient when sample */
    // pass

    /* log nodes */
    if(verbose) {
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
    }
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
    // Tag tag(seq.get(), corpus, &rng, model->param);
    ptr<GraphicalModel> gm = model->makeSample(*seq, corpus, &rng);
    tree.root->log_weight = -DBL_MAX;
    for(size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = addChild(tree.root, *gm);
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
  time = 0;
  wallclock = 0;
  wallclock_sample = 0;
  wallclock_policy = 0;
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
  double hit_count = 0, pred_count = 0, truth_count = 0;
  double ave_time = 0;
  vector<MarkovTreeNodePtr> stack;
  vector<int> id;
  for(const SentencePtr seq : result->corpus->seqs) {
    if(count >= test_count) break;
    MarkovTreeNodePtr node;
    if(result->nodes[count] == nullptr) {
      node = makeMarkovTreeNode(nullptr);
      node->model = model;
      node->gm = model->makeSample(*seq, model->corpus, &rng);
      node->log_prior_weight = model->score(*node->gm);
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
        ave_time += node->depth;
        if(model->scoring == Model::SCORING_ACCURACY) {
          tuple<int, int> hit_pred = model->evalPOS(*cast<Tag>(node->max_gm));
          hit_count += get<0>(hit_pred);
          pred_count += get<1>(hit_pred);
        }else if(model->scoring == Model::SCORING_NER) {
          tuple<int, int, int> hit_pred_truth = model->evalNER(*cast<Tag>(node->max_gm));
          hit_count += get<0>(hit_pred_truth);
          pred_count += get<1>(hit_pred_truth);
          truth_count += get<2>(hit_pred_truth);
        }else if(model->scoring == Model::SCORING_LHOOD) {
          hit_count += model->score(*node->max_gm);
          pred_count++;
        }
        lg->end(); // </example_i>
      }
      stack.clear();
      id.clear();
    }
  }
  lg->end(); // </example>
  /* log summary stats */
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
  }else if(model->scoring == Model::SCORING_LHOOD) {
    lg->begin("accuracy");
    *lg << accuracy << endl;
    cout << "lhood: " << accuracy << endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  }
  // result->score = -1;
}

FeaturePointer Policy::extractFeatures(MarkovTreeNodePtr node, int pos) {
  FeaturePointer feat = makeFeaturePointer();
  GraphicalModel& gm = *node->gm;
  size_t seqlen = gm.size();
  const Instance& seq = *gm.seq;
  // bias.
  if(featoptFind("bias") || featoptFind("all"))
    insertFeature(feat, "b");
  if(featoptFind("word-ent") || featoptFind("all")) {
    size_t taglen = model->corpus->tags.size();
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

  if(featoptFind(COND) || featoptFind(NB_ENT__COND) || featoptFind("all")) {
    insertFeature(feat, COND, node->gm->entropy[pos]);
  }

  if(featoptFind("log-cond-ent")) {
    insertFeature(feat, "log-cond-ent", log(1e-40 + node->gm->entropy[pos]));
  }

  if(featoptFind("01-cond-ent")) {
    insertFeature(feat, "01-cond-ent", node->gm->entropy[pos] < 1e-10 ? 1 : 0);
  }

  if(featoptFind(COND_LHOOD) || featoptFind("all")) {
    int label = node->gm->getLabel(pos);
    insertFeature(feat, COND_LHOOD, node->gm->this_sc[pos][label]);
  }

  if(featoptFind("unigram-ent")) {
    if(model_unigram) {
      if(std::isnan(node->gm->entropy_unigram[pos])) {
        ptr<GraphicalModel> gm_ptr = model->copySample(*node->gm);
        auto& gm = *gm_ptr;
        model_unigram->sampleOne(gm, *gm.rng, pos);
        node->gm->entropy_unigram[pos] = gm.entropy[pos];
      }
      insertFeature(feat, "unigram-ent", node->gm->entropy_unigram[pos]);
    }
  }

  if(featoptFind("inv-unigram-ent")) {
    if(model_unigram) {
      if(std::isnan(node->gm->entropy_unigram[pos])) {
        ptr<GraphicalModel> gm_ptr = model->copySample(*node->gm);
        auto& gm = *gm_ptr;
        model_unigram->sampleOne(gm, *gm.rng, pos);
        node->gm->entropy_unigram[pos] = gm.entropy[pos];
      }
      insertFeature(feat, "inv-unigram-ent", 1/(1e-4+node->gm->entropy_unigram[pos]));
    }
  }

  if(featoptFind("01-unigram-ent")) {
    if(model_unigram) {
      if(std::isnan(node->gm->entropy_unigram[pos])) {
        ptr<GraphicalModel> gm_ptr = model->copySample(*node->gm);
        auto& gm = *gm_ptr;
        model_unigram->sampleOne(gm, *gm.rng, pos);
        node->gm->entropy_unigram[pos] = gm.entropy[pos];
      }
      insertFeature(feat, "01-unigram-ent", node->gm->entropy_unigram[pos] < 1e-8);
    }
  }

  if(featoptFind("logistic-unigram-ent")) {
    if(model_unigram) {
      if(std::isnan(node->gm->entropy_unigram[pos])) {
        ptr<GraphicalModel> gm_ptr = model->copySample(*node->gm);
        auto& gm = *gm_ptr;
        model_unigram->sampleOne(gm, *gm.rng, pos);
        node->gm->entropy_unigram[pos] = gm.entropy[pos];
      }
      insertFeature(feat, "logistic-unigram-ent", logisticFunc(node->gm->entropy_unigram[pos]));
    }
  }

  if(featoptFind(NER_DISAGREE) || featoptFind("all")) {
    if(model->scoring == Model::SCORING_NER) { // tag inconsistency, such as B-PER I-LOC
      auto& tag = *cast<Tag>(node->gm);
      ptr<Corpus> corpus = model->corpus;
      string tg = corpus->invtags[tag.tag[pos]];
      if(pos >= 1) {
              string prev_tg = corpus->invtags[tag.tag[pos-1]];
              if((prev_tg[0] == 'B' and tg[0] == 'I' and tg.substr(1) != prev_tg.substr(1))
          or (prev_tg[0] == 'I' and tg[0] == 'I' and tg.substr(1) != prev_tg.substr(1))) {
                insertFeature(feat, NER_DISAGREE_L);
        }
      }
      if(pos < node->gm->size()-1) {
        string next_tg = corpus->invtags[tag.tag[pos+1]];
              if((next_tg[0] == 'I' and tg[0] == 'B' and tg.substr(1) != next_tg.substr(1))
          or (next_tg[0] == 'I' and tg[0] == 'I' and tg.substr(1) != next_tg.substr(1))) {
                insertFeature(feat, NER_DISAGREE_R);
        }
      }
    }
  }

  if(featoptFind("ising-disagree")) {
    auto image = (const ImageIsing*)(node->gm->seq); // WARNING: Hack, no dynamic check.
    auto& tag = *cast<Tag>(node->gm);
    if(image == NULL)
      throw "cannot use feature 'ising-disagree' on non-ising dataset.";
    size_t disagree = 0;
    ImageIsing::Pt pt = image->posToPt(pos);
    vec2d<int> steer = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    for(auto steer_it : steer) {
      ImageIsing::Pt this_pt = pt;
      this_pt.h += steer_it[0];
      this_pt.w += steer_it[1];
      if(this_pt.h >= 0 and this_pt.h < image->H
        and this_pt.w >= 0 and this_pt.w < image->W) {
        size_t pos2 = image->ptToPos(this_pt);
        if(tag.tag[pos2] != tag.tag[pos]) {
          disagree++;
        }
      }
    }
    insertFeature(feat, "ising-disagree", disagree);
  }

  /* features based on neighbors. */
  if(featoptFind(NB_VARY)) {
    insertFeature(feat, NB_VARY, 0); // to be modified dynamically.
  }

  if(featoptFind(NB_ENT) || featoptFind(NB_ENT__COND)) {
    double nb_sure = 0;
    int count = 0;
    for(auto id : model->markovBlanket(*node->gm, pos)) {
      nb_sure += node->gm->entropy[id];
      count++;
    }
    nb_sure /= (double)count;
    insertFeature(feat, NB_ENT, nb_sure);
  }

  if(featoptFind(NB_ENT__COND)) {
    insertFeature(feat, NB_ENT__COND, getFeature(feat, NB_ENT) * node->gm->entropy[pos]);
  }

  /* oracle features, that require sampling */
  if(featoptFind(ORACLE) || featoptFind(ORACLEv)) {
    insertFeature(feat, ORACLE, 0);     // computed in updateResp.
  }

  if(featoptFind(ORACLE_ENT) || featoptFind(ORACLE_ENTv)) {
    insertFeature(feat, ORACLE_ENT, 0);  // computed in updateResp.
  }

  if(featoptFind(ORACLE_STALENESS) || featoptFind(ORACLE_STALENESSv)) {
    insertFeature(feat, ORACLE_STALENESS, 0); // computed in updateResp.
  }

  if(featoptFind(NB_CONSENT)) {
    // strategy 1. use it only when neighbors change.
    {
    }
    // strategy 2. use it anyhow.
    {
      // for(auto id : model->markovBlanket(*node->gm, pos)) {
      //   insertFeature(feat,
      //                 make_NB_CONSENT(node->gm->getLabel(pos),
      //                                 node->gm->getLabel(id)),
      //                 1);
      // }
    }

  }
  return feat;
}

MarkovTreeNodePtr Policy::sampleOne(MarkovTreeNodePtr node, objcokus& rng, int pos) {
  model->sampleOne(*node->gm, rng, pos);
  node->log_prior_weight += node->gm->reward[pos];
  auto lag = (int)(node->gm->size() / lazymax_lag);
  if(!lets_lazymax
     || (lazymax_lag == -1 and pos == node->gm->size() - 1)
     || (lazymax_lag != -1 and
         ((lag != 0 and node->depth % lag == lag-1)
          || pos == node->gm->size() - 1))) {   // lazy copy
    if(node->log_prior_weight > node->max_log_prior_weight) {
      node->max_log_prior_weight = node->log_prior_weight;
      node->max_gm = model->copySample(*node->gm);
    }
  }
  node->gm->mask[pos] += 1;
  if(lets_inplace) {
    node->depth++;
  }else {
    node = addChild(node, *node->gm);
  }
  return node;
}

/* update resp has two parts: update features, compute new responses */
void Policy::updateResp(MarkovTreeNodePtr node, objcokus& rng, int pos, Heap* heap) {
  /* extract my meta-feature */
  FeaturePointer feat = this->extractFeatures(node, pos);
  node->gm->feat[pos] = feat;

  /* update neighbor stats */
  node->gm->changed[pos].clear();
  for(auto id : model->markovBlanket(*node->gm, pos)) {
    node->gm->changed[pos][id] = false;
    node->gm->vary[pos][id] = 0;
  }
  for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
    node->gm->vary[id][pos] += 1;
  }

  /* update my response */
  node->gm->resp[pos] = Tagging::score(this->param, feat);
  node->gm->checksum[pos] = 0; // WARNING: a hack.
  int val = node->gm->getLabel(pos), oldval = node->gm->oldlabels[pos];

  auto updateRespByHandle = [&] (int id) {
    if(heap == nullptr) return;
    Value& val = *node->gm->handle[id];
    val.resp = node->gm->resp[id];
    heap->update(node->gm->handle[id]);
  };
  updateRespByHandle(pos);

  /* update my friends' response */
  if(featoptFind(NB_VARY)) {
    /* update the nodes in inv Markov blanket */
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0) {
        assert(node->gm->blanket[id].contains(pos));
        double* feat_nb_vary = findFeature(node->gm->feat[id], NB_VARY);
        if(node->gm->blanket[id][pos] != val and node->gm->changed[id][pos] == false) {
          node->gm->changed[id][pos] = true;
          (*feat_nb_vary)++;
          node->gm->resp[id] += (*param)[NB_VARY];
          updateRespByHandle(id);
        }
        if(node->gm->blanket[id][pos] == val and node->gm->changed[id][pos] == true) {
          node->gm->changed[id][pos] = false;
          (*feat_nb_vary)--;
          node->gm->resp[id] -= (*param)[NB_VARY];
          updateRespByHandle(id);
        }
      }
    }
  }

  if(featoptFind(NB_CONSENT)) {
    /* update the nodes in inv Markov blanket */
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0) {
        double yourval = node->gm->getLabel(id);
        if(node->gm->vary[id][pos] > 1) { // more than the first time.
          // invalidate old feat.
          double* oldfeat = findFeature(node->gm->feat[id], make_NB_CONSENT(yourval, oldval));
          assert(oldfeat != nullptr and *oldfeat != 0);
          node->gm->resp[id] -= (*param)[make_NB_CONSENT(yourval, oldval)];
          (*oldfeat)--;
        }
        // insert new feat.
        double* newfeat = findFeature(node->gm->feat[id], make_NB_CONSENT(yourval, val));
        if(newfeat == nullptr) {
          insertFeature(node->gm->feat[id], make_NB_CONSENT(yourval, val));
        }else{
          (*newfeat)++;
        }
        node->gm->resp[id] += (*param)[make_NB_CONSENT(yourval, val)];
        updateRespByHandle(id);
      }
    }
  }

  if(featoptFind(NB_ENT) || featoptFind(NB_ENT__COND)) {
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0) {
        double* feat_nb_ent = findFeature(node->gm->feat[id], NB_ENT);
        double ent_diff = (node->gm->entropy[pos] - node->gm->prev_entropy[pos])
                              / (double)node->gm->blanket[id].size();;
        (*feat_nb_ent) += ent_diff;
        node->gm->resp[id] += (*param)[NB_ENT] * ent_diff;
        if(featoptFind(NB_ENT__COND)) {
          double* feat_nb_ent__cond = findFeature(node->gm->feat[id], NB_ENT__COND);
          double ent = getFeature(node->gm->feat[id], COND);
          (*feat_nb_ent__cond) += ent_diff * ent;
          node->gm->resp[id] += (*param)[NB_ENT__COND] * ent_diff * ent;
        }
      }
    }
  }

  if(featoptFind(NER_DISAGREE)) {
    if(getFeature(feat, NER_DISAGREE_L) and node->gm->blanket[pos-1].size() > 0
      and getFeature(node->gm->feat[pos-1], NER_DISAGREE_R) == 0) {
      insertFeature(node->gm->feat[pos-1], NER_DISAGREE_R);
      node->gm->resp[pos-1] = (*param)[NER_DISAGREE_R];
      updateRespByHandle(pos-1);
    }
    if(getFeature(feat, NER_DISAGREE_R) and node->gm->blanket[pos+1].size() > 0
      and getFeature(node->gm->feat[pos+1], NER_DISAGREE_L) == 0) {
      insertFeature(node->gm->feat[pos+1], NER_DISAGREE_L);
      node->gm->resp[pos+1] = (*param)[NER_DISAGREE_L];
      updateRespByHandle(pos+1);
    }
  }

  if(featoptFind(ORACLE) || featoptFind(ORACLEv)) {   // oracle feature is just *reward*.
    auto computeOracle = [&] (int id) {
      auto feat = findFeature(node->gm->feat[id], ORACLE);

      *feat = sampleDelayedReward(node, id, this->mode_oracle, this->rewardK);

      if(featoptFind(ORACLE)) {
        node->gm->resp[id] = Tagging::score(this->param, node->gm->feat[id]);
        updateRespByHandle(id);
      }
    };
    computeOracle(pos);
    set<int> visited;
    visited.insert(pos);
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0 and visited.count(id) == 0) { // has already been initialized.
        computeOracle(id);
        visited.insert(id);
      }
    }
  }

  if(featoptFind(ORACLE_ENT) || featoptFind(ORACLE_ENTv)) {
   auto computeOracle = [&] (double* feat, int id) {
      int oldval = node->gm->getLabel(id);
      model->sampleOne(*node->gm, rng, id, false);
      *feat = logEntropy(&node->gm->sc[0], node->gm->numLabels(id));
      node->gm->setLabel(id, oldval);
      if(featoptFind(ORACLE_ENT)) {
        node->gm->resp[id] = Tagging::score(this->param, node->gm->feat[id]);
        updateRespByHandle(id);
      }
    };
    computeOracle(findFeature(feat, ORACLE_ENT), pos);
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0) { // has already been initialized.
        computeOracle(findFeature(node->gm->feat[id], ORACLE_ENT), id);
      }
    }
  }

  if(featoptFind(ORACLE_STALENESS) || featoptFind(ORACLE_STALENESSv)) {
   auto computeOracle = [&] (double* feat, int id) {
      int oldval = node->gm->getLabel(id);
      model->sampleOne(*node->gm, rng, id, false);
      *feat = node->gm->this_sc[id][oldval] - node->gm->sc[oldval];
      node->gm->setLabel(id, oldval);
      if(featoptFind(ORACLE_STALENESS)) {
        node->gm->resp[id] = Tagging::score(this->param, node->gm->feat[id]);
        updateRespByHandle(id);
      }
    };
    computeOracle(findFeature(feat, ORACLE_STALENESS), pos);
    for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
      if(node->gm->blanket[id].size() > 0) { // has already been initialized.
        computeOracle(findFeature(node->gm->feat[id], ORACLE_STALENESS), id);
      }
    }
  }


  /* update Markov blanket */
  node->gm->blanket[pos] = node->gm->getLabels(model->markovBlanket(*node->gm, pos));
}

double Policy::checksum(MarkovTreeNodePtr node, int pos) {
  size_t checksum = 0;
  int factorL = cast<ModelCRFGibbs>(model)->factorL;
  for(int p = max(0, pos-factorL); p <= min(pos+factorL, int(node->gm->size())); p++) {
    checksum = checksum * 13 + node->gm->timestamp[p];
  }
  return (double)checksum;
}

void Policy::logNode(MarkovTreeNodePtr node) {
  while(node->children.size() > 0) node = node->children[0]; // take final sample.
  lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
  lg->begin("truth"); *lg<< node->gm->seq->str() << endl; lg->end();
  lg->begin("tag"); *lg << node->gm->str() << endl; lg->end();
  lg->begin("resp");
  auto logIndex = [&] (int index) {
    *lg << " / [" << boost::lexical_cast<string>(index) + "]";
  };

  for(size_t i = 0; i < node->gm->size(); i++) {
    *lg << node->gm->resp[i];
    if(verbose) {
      logIndex(i);
    }
    *lg << "\t";
  }
  *lg << endl;
  lg->end();
  if(node->gm->size() > 0 and verbose) {
    lg->begin("feat");
    /* take the uninon of features from all positions */
    std::set<string> feat_names;
    for(size_t t = 0; t < node->gm->size(); t++) {
      if(node->gm->feat[t] == nullptr) continue;
      for(auto& p : *node->gm->feat[t]) {
        feat_names.insert(string(p.first));
      }
    }
    /* log each position */
    for(auto& p : feat_names) {
        lg->begin(p);
        for(size_t i = 0; i < node->gm->size(); i++) {
          *lg << getFeature(node->gm->feat[i], p);
          if(verbose) {
            logIndex(i);
          }
          *lg << "\t";
        }
        *lg << endl;
        lg->end();
      }
    lg->end();
  }
  lg->begin("mask");
  for(size_t i = 0; i < node->gm->size(); i++) {
    *lg << node->gm->mask[i];
    if(verbose) {
      logIndex(i);
    }
    *lg << "\t";
  }
  *lg << endl;
  lg->end();
  lg->begin("score");
    *lg << node->log_prior_weight << endl;
  lg->end(); // <score>
}

void Policy::resetLog(std::shared_ptr<XMLlog> new_lg) {
  while(lg != nullptr and lg->depth() > 0) lg->end();
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
  if(node->depth < T * node->gm->size()) {
    node->time_stamp++;
    return node->depth % node->gm->size();
  }
  return -1; // stop.
}

//////// LockdownPolicy ////////////////////////////
LockdownPolicy::LockdownPolicy(ModelPtr model, const boost::program_options::variables_map& vm)
 :Policy(model, vm),
  T(vm["T"].as<size_t>())
{
  c = vm["c"].empty() ? 0 : vm["c"].as<double>();
}

FeaturePointer LockdownPolicy::extractFeatures(MarkovTreeNodePtr node, int pos) {
  FeaturePointer feat = Policy::extractFeatures(node, pos);
  if(featoptFind("exp-sp")) {
    insertFeature(feat, "exp-sp", exp(node->gm->mask[pos]));
  }
  if(featoptFind("log-sp")) {
    insertFeature(feat, "log-sp", log(1+node->gm->mask[pos]));
  }
  if(featoptFind("sp")) {
    insertFeature(feat, "sp", node->gm->mask[pos]);
  }
  if(featoptFind("sp-norm")) {
    insertFeature(feat, "sp-norm", node->gm->mask[pos] * 10.0 / (double)T);
  }
  if(featoptFind("sp-cond-ent")) {
    insertFeature(feat, boost::lexical_cast<string>(node->gm->mask[pos])+"-cond-ent", node->gm->entropy[pos]);
  }
  if(featoptFind("sp-unigram-ent")) {
    insertFeature(feat, boost::lexical_cast<string>(node->gm->mask[pos])+"-unigram-ent", node->gm->entropy_unigram[pos]);
  }
  return feat;
}

int LockdownPolicy::policy(MarkovTreeNodePtr node) {
  if(node->depth == 0) node->time_stamp = 0;
  size_t count = node->time_stamp;
  int seqlen = node->gm->size();
  for(; count < node->time_stamp + seqlen; count++) {
    int pos = count % seqlen;

    /* TODO: compute feat on demand */
    if(!std::isnan(node->gm->checksum[pos])) {
      FeaturePointer feat = this->extractFeatures(node, pos);
      node->gm->feat[pos] = feat;
      node->gm->resp[pos] = Tagging::score(param, feat);
      if(lets_resp_reward) {
        double resp = node->gm->resp[pos];
        test_thread_pool.lock();
        // collect reward, slow, for debug only.
//          Tag tag(*node->gm);
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
    int mk = node->gm->mask[pos];
    if(node->gm->resp[pos] > c and
        node->gm->mask[pos] < T) {
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
    objcokus& rng = thread_pool.rngs[tid];
    node->gm->rng = &rng;
    for(size_t i = 0; i < node->gm->size(); i++) {
      this->sampleOne(node, rng, i);
      thread_pool.lock();
      this->updateResp(node, rng, i, nullptr);
      thread_pool.unlock();
      if(verbose) {
        lg->begin("T_0_i_" + to_string(i));
        logNode(node);
        lg->end();
      }
    }
    for(size_t t = 1; t < T; t++) {
      for(size_t i = 0; i < node->gm->size(); i++) {
        node->time_stamp = t * node->gm->size() + i;
        /* extract features */
        FeaturePointer feat = node->gm->feat[i];

        // double resp = node->gm->resp[i];
        double log_resp = Tagging::score(param, feat); // fix: param always changes, so does resp.
        double logR = 0;
        double staleness = 0;
        PolicyExample example;

        /* estimate reward */
#if REWARD_SCHEME == REWARD_ACCURACY
        ptr<Tag> old_tag = cast<Tag>(node->gm);
        auto is_equal = [&] () {
          return (double)(old_tag->tag[i] == old_tag->seq->tag[i]);
        };
        double reward_baseline = is_equal();
        int oldval = node->gm->getLabel(i);
        for(size_t j = 0; j < J; j++) {
          this->sampleOne(node, rng, i);
          double reward = is_equal();
          logR += reward - reward_baseline;
          if(j < J-1)  // reset.
            node->gm->setLabel(i, oldval);
        }
        logR /= (double)J;

#elif REWARD_SCHEME == REWARD_LHOOD
        logR = sampleDelayedReward(node, i, this->mode_reward, this->rewardK);

        if(lets_resp_reward) {  // record training examples.
        // if(logR > 5) {  // record high-reward examples.
          example.oldstr = node->gm->str();
        }

        this->sampleOne(node, rng, i);
        // int oldval = node->gm->getLabel(i);
        // const int num_label = node->gm->numLabels(i);
        // this->sampleOne(node, rng, i);
        // logR =  - node->gm->sc[oldval];
        // for(int label = 0; label < num_label; label++) {
        //   logR += exp(node->gm->sc[label]) * node->gm->sc[label];
        // }

        // for(size_t j = 0; j < J; j++) {
        //   if(j < J-1) {
        //     model->sampleOne(*node->gm, rng, i);
        //     node->gm->setLabel(i, oldval);
        //   }else{
        //     this->sampleOne(node, rng, i);
        //   }
        //   if(j  == 0) {
        //     staleness = node->gm->staleness[i];
        //   }
        //   logR += node->gm->reward[i];
        // }
        // logR /= (double)J;

#endif
        /* use gradients to update model */

        thread_pool.lock();

        auto grad = makeParamPointer();
        /* update meta-model (strategy 1) */
        if(learning == "linear") {
          mapUpdate(*grad, *feat, (logR - log_resp));
        }

//          /* update meta-model (strategy 1.5) */
//          mapUpdate(*grad, *feat, ((logR > 0) - resp));

        /* update meta-model (strategy 1.6) */
//          if(logR > 0) {
//            mapUpdate(*grad, *feat, (1 - resp));
//          }else{
//            mapUpdate(*grad, *feat, (-1 - resp));
//          }

        /* update meta-model (strategy 2) */
        if(learning == "logistic") {
          double resp = logisticFunc(log_resp);
          if(logR > 0) {
            mapUpdate(*grad, *feat, (1-resp));
          }else{
            mapUpdate(*grad, *feat, -resp);
          }
        }

        /* update meta-model (strategy 3) neural network */
        if(learning == "nn") {
          double resp = logisticFunc(log_resp);
          if(param->count("L2-w") == 0) {
            (*param)["L2-w"] = 0;
          }
          if(param->count("L2-b") == 0) {
            (*param)["L2-b"] = 0;
          }
          double w = (*param)["L2-w"];
          double b = (*param)["L2-b"];
          double diff = (logR - w * resp - b);
          mapUpdate(*grad, *feat, diff * resp * (1 - resp) * w);
          mapUpdate(*grad, "L2-w", diff * resp);
          mapUpdate(*grad, "L2-b", diff);
        }

        adagrad(param, G2, grad, eta);   // overwrite adagrad, for fine-grain gradients. (return node->gradient empty).

        if(featoptFind(ORACLEv))
          (*param)[ORACLE] = 0;
        if(featoptFind(ORACLE_ENTv))
          (*param)[ORACLE_ENT] = 0;
        if(featoptFind(ORACLE_STALENESSv))
          (*param)[ORACLE_STALENESS] = 0;

       if(lets_resp_reward) {  // record training examples.
        // if(logR > 5) {  // record high-reward examples.
          resp_reward.push_back(make_pair(log_resp, logR));
          example.reward = logR;
          example.staleness = staleness;
          example.resp = log_resp;
          example.feat = makeFeaturePointer();
          example.param = makeParamPointer();
          example.node = node;
          example.str = node->gm->str();
          example.choice = i;
          *example.feat = *feat;
          *example.param = *param;
          this->examples.push_back(example);
        }

        if(verbose) {
          lg->begin("T_" + to_string(t) + "_i_" + to_string(i));
          logNode(node);
          lg->end();
        }

        /* update response */
        this->updateResp(node, rng, i, nullptr);

        thread_pool.unlock();
      }
    }
    node->log_weight = 0;
  }catch(const char* ee) {
    cout << "error: " << ee << endl;
  }
}



}
