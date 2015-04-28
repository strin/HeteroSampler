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

namespace HeteroSampler {

Policy::Policy(ModelPtr model, const po::variables_map& vm)
  : model(model), 
    test_thread_pool(vm["numThreads"].as<size_t>(),
                     [ & ] (int tid, MarkovTreeNodePtr node) {
                        this->sample(tid, node);
                     }),
    thread_pool(vm["numThreads"].as<size_t>(),
                    [ & ] (int tid, MarkovTreeNodePtr node) {
                        this->sample(tid, node);
                    }),
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
    init_method(vm["init"].empty() ? "" : vm["init"].as<string>()),
    param(makeParamPointer()), G2(makeParamPointer()) {

  // parse other options
  try {
    lg = std::make_shared<XMLlog>(vm["output"].as<string>());
  } catch (char const* error) {
    throw error;
  }

  if (!vm["log"].empty() and vm["log"].as<string>() != "") {
    try {
      auxlg = std::make_shared<XMLlog>(vm["log"].as<string>());
    } catch (char const* error) {
      cout << "error: " << error << endl;
      auxlg = std::make_shared<XMLlog>();
    }
  } else {
    auxlg = std::make_shared<XMLlog>();
  }

  // parsing meta-feature 
  vec<string> featopt;
  split(featopt, vm["feat"].as<string>(), boost::is_any_of(" "));

  MapInitializer<string, MetaFeature> feat_init(feat_map);
  
  feat_init("log-cond-ent", FEAT_LOG_COND_ENT) 
          ("01-cond-ent", FEAT_01_COND_ENT) 
          ("cond-ent", FEAT_COND_ENT)
          ("unigram-ent", FEAT_UNIGRAM_ENT)
          ("inv-unigram-ent", FEAT_INV_UNIGRAM_ENT)
          ("logistic-unigram-ent", FEAT_LOGISTIC_UNIGRAM_ENT)
          ("01-unigram-ent", FEAT_01_UNIGRAM_ENT)
          ("bias", FEAT_BIAS)
          ("nb-vary", FEAT_NB_VARY)
          ("nb-ent", FEAT_NB_ENT)
          ("nb-discord", FEAT_NB_DISCORD)
          ("oracle", FEAT_ORACLE)
          ("oracle-ent", FEAT_ORACLE_ENT)
          ("oracle-staleness", FEAT_ORACLE_STALENESS)
          ("cond-lhood", FEAT_COND_LHOOD)
          ("sp", FEAT_SP)
          ("exp-sp", FEAT_EXP_SP)
          ("log-sp", FEAT_LOG_SP)
          ("sp-cond-ent", FEAT_SP_COND_ENT)
          ("sp-unigram-ent", FEAT_UNIGRAM_ENT)
          ;

  MapInitializer<MetaFeature, string, MetaFeatureHash> name_init(feat_name);
  name_init(FEAT_LOG_COND_ENT, "lce")
            (FEAT_01_COND_ENT, "0ce")
            (FEAT_COND_ENT, "ce")
            (FEAT_BIAS, "b")
            (FEAT_NB_VARY, "nv")
            (FEAT_COND_LHOOD, "cl")
            (FEAT_ORACLE, "o")
            (FEAT_ORACLE_ENT, "oe")
            (FEAT_ORACLE_STALENESS, "os")
            (FEAT_UNIGRAM_ENT, "ue")
            (FEAT_INV_UNIGRAM_ENT, "iue")
            (FEAT_LOGISTIC_UNIGRAM_ENT, "lue")
            (FEAT_01_UNIGRAM_ENT, "0ue")
            (FEAT_NB_ENT, "ne")
            (FEAT_SP, "sp")
            (FEAT_EXP_SP, "esp")
            (FEAT_LOG_SP, "lsp")
            (FEAT_SP_COND_ENT, "spc")
            (FEAT_SP_UNIGRAM_ENT, "spu")
            ;

  auto add_feat = [&] (string name) {
    if(feat_map.contains(name)) {
      MetaFeature f = feat_map[name];
      this->feat.push_back(f);
      this->feat_name[f] = name;
      return true;
    }else{
      throw (string("unrecognized meta-feature") + name).c_str();
    }
  };

  for(string key : featopt) {
    if(key == "") continue;
    add_feat(key);
  }

  split(verbose_opt, vm["verbosity"].as<string>(), boost::is_any_of(" "));

  int sysres = system(("mkdir -p " + name).c_str());
}

Policy::~Policy() {
  while (lg != nullptr and lg->depth() > 0)
    lg->end();
}

double Policy::delayedReward(MarkovTreeNodePtr node, int depth, int maxdepth, vec<int>& actions) {
  int id;
  if (depth < actions.size()) { // take specified action.
    id = actions[depth];
  } else { // sample new action.
    if (depth == 0) { //sample uniformly.
      id = int(rng.random01() * (1 - 1e-8) * node->gm->size());
    } else {
      vec<int> blanket = model->markovBlanket(*node->gm, actions[depth - 1]);
      if (blanket.size() == 0) {
        id = actions[depth - 1];
      } else {
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
  if (depth < maxdepth) {
    R += this->delayedReward(node, depth + 1, maxdepth, actions);
  }
  node->gm->setLabel(id, oldval);
  return R;
}

double Policy::sampleDelayedReward(MarkovTreeNodePtr node, int id, int maxdepth, int rewardK) {
  vec<int> actions_u(1);
  actions_u[0] = id;
  double R_u = this->delayedReward(node, 0, maxdepth, actions_u);
  vec<int> actions_v(actions_u.begin() + 1, actions_u.end());
  double R_v = this->delayedReward(node, 0, maxdepth - 1, actions_v);
  return R_u - R_v;
}

void Policy::sample(int tid, MarkovTreeNodePtr node) {
  node->depth = 0;
  objcokus& rng = test_thread_pool.rngs[tid];
  node->gm->rng = &rng;
  try {
    if (this->init_method == "iid") {
      for (size_t pos = 0; pos < node->gm->size(); pos++) {
        model->sampleOneAtInit(*node->gm, rng, pos);
        node->depth++;
        node->gm->mask[pos] += 1;
      }
      node->max_gm = model->copySample(*node->gm);
    }
    while (true) {
      node->choice = this->policy(node);
      if (node->choice.type == Location::LOC_NULL) {
        node->log_weight = model->score(*node->gm);
        node->gradient = makeParamPointer();
        break;
      } else {
        node->log_weight = -DBL_MAX;
        int pos = node->choice.pos;
        this->sampleOne(node, rng, pos);
        this->updateResp(node, rng, pos, nullptr);
      }
    }
  } catch (const char* ee) {
    cout << "error: " << ee << endl;
  }
}

void Policy::train(ptr<Corpus> corpus) {
  lg->begin("train");
  /* train the model */
  if (Q > 0)
    *auxlg << "start training ... " << endl;
  for (size_t q = 0; q < Q; q++) {
    *auxlg << "\t epoch " << q << endl;
    *auxlg << "\t update policy " <<  endl;
    this->train_policy(corpus);
  }
  if (Q == 0) { // notraining is needed.
    // set all feature weights to 1.
    for (const auto& opt : feat_name) {
      (*param)[opt.second] = 1;
    }
    // overwrite.
    if (feat_name.find(FEAT_SP) != feat_name.end())
      (*param)[feat_name[FEAT_SP]] = -0.3;
  }
  /* log policy examples */
  if (lets_resp_reward) {
    lg->begin("policy_example");
    for (auto example : this->examples) {
      example.serialize(lg);
    }
    lg->end(); // </policy_example>
  }
  lg->begin("param");
  *lg << *param;
  lg->end(); // </param>
  lg->end(); // </train>
}

void Policy::train_policy(ptr<Corpus> corpus) {
  lg->begin("commit"); *lg << getGitHash() << endl; lg->end();
  corpus->retag(model->corpus);
  size_t count = 0;
  examples.clear();
  for (const SentencePtr seq : corpus->seqs) {
    if (count >= train_count) break;
    // cout << corpus->seqs.size() << endl;
    size_t display_lag = int(0.1 * min(train_count, corpus->seqs.size()));
    if (display_lag == 0 or count % display_lag == 0)
      *auxlg << "\t\t " << (double)count / corpus->seqs.size() * 100 << " %" << endl;
    if (verbose)
      lg->begin("example_" + to_string(count));
    MarkovTree tree;
    ptr<GraphicalModel> gm = model->makeSample(*seq, model->corpus, &rng);
    tree.root->log_weight = -DBL_MAX;
    tree.root->model = this->model;
    for (size_t k = 0; k < K; k++) {
      MarkovTreeNodePtr node = addChild(tree.root, *gm);
      this->thread_pool.addWork(node);
    }
    thread_pool.waitFinish();

    /* log nodes */
    if (verbose) {
      for (size_t k = 0; k < K; k++) {
        MarkovTreeNodePtr node = tree.root->children[k];
        while (node->children.size() > 0) node = node->children[0]; // take final sample.
        lg->begin("node");
        this->logNode(node);
        lg->end(); // </node>
      }
      lg->begin("param");
      *lg << *param;
      lg->end(); // </param>
    }
    if (verbose) {
      lg->end(); // </example>
    }
    count++;
  }
}


Policy::Result::Result(ptr<Corpus> corpus)
  : corpus(corpus) {
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
  *auxlg << "> test " << endl;
  lg->begin("test");
  lg->begin("param");
  *lg << *param;
  lg->end(); // </param>
  this->test_policy(result);
  lg->end(); // </test>
}

void Policy::test_policy(Policy::ResultPtr result) {
  assert(result != nullptr);
  size_t count = 0;
  clock_t time_start = clock(), time_end;

  lg->begin("example");
  count = 0;
  double hit_count = 0, pred_count = 0, truth_count = 0;
  double ave_time = 0;
  vector<MarkovTreeNodePtr> stack;
  vector<int> id;
  for (const SentencePtr seq : result->corpus->seqs) {
    if (count >= test_count) break;
    MarkovTreeNodePtr node;
    if (result->nodes[count] == nullptr) {
      node = makeMarkovTreeNode(nullptr);
      node->model = model;
      node->gm = model->makeSample(*seq, model->corpus, &rng);
      node->log_prior_weight = model->score(*node->gm);
    } else {
      node = result->nodes[count];
    }
    stack.push_back(node);
    id.push_back(count);
    test_thread_pool.addWork(node);
    count++;
    if (count % thread_pool.numThreads() == 0 || count == test_count
        || count == result->corpus->seqs.size()) {
      test_thread_pool.waitFinish();
      for (size_t i = 0; i < id.size(); i++) {
        MarkovTreeNodePtr node = stack[i];
        lg->begin("example_" + to_string(id[i]));
        this->logNode(node);
        while (node->children.size() > 0) node = node->children[0]; // take final sample.
        result->nodes[id[i]] = node;
        ave_time += node->depth;
        if (model->scoring == Model::SCORING_ACCURACY) {
          tuple<int, int> hit_pred = model->evalPOS(*cast<Tag>(node->max_gm));
          hit_count += get<0>(hit_pred);
          pred_count += get<1>(hit_pred);
        } else if (model->scoring == Model::SCORING_NER) {
          tuple<int, int, int> hit_pred_truth = model->evalNER(*cast<Tag>(node->max_gm));
          hit_count += get<0>(hit_pred_truth);
          pred_count += get<1>(hit_pred_truth);
          truth_count += get<2>(hit_pred_truth);
        } else if (model->scoring == Model::SCORING_LHOOD) {
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
  *auxlg << "time: " << result->time << endl;
  *lg << result->time << endl;
  lg->end(); // </time>
  lg->begin("wallclock");
  *auxlg << "wallclock: " << result->wallclock << endl;
  *lg << result->wallclock << endl;
  lg->end(); // </wallclock>
  if (model->scoring == Model::SCORING_ACCURACY) {
    lg->begin("accuracy");
    *lg << accuracy << endl;
    *auxlg << "acc: " << accuracy << endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  } else if (model->scoring == Model::SCORING_NER) {
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    lg->begin("accuracy");
    *lg << f1 << endl;
    *auxlg << "f1: " << f1 << endl;
    lg->end(); // </accuracy>
    result->score = f1;
  } else if (model->scoring == Model::SCORING_LHOOD) {
    lg->begin("accuracy");
    *lg << accuracy << endl;
    *auxlg << "lhood: " << accuracy << endl;
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
  
  for(MetaFeature f : this->feat) {

    string name = feat_name[f];
    auto add_feat = [&] (double value) {
      insertFeature(feat, name, value);
    };

    auto unigram_ent = [&] () {
      if (model_unigram) {
        if (std::isnan(node->gm->entropy_unigram[pos])) {
          ptr<GraphicalModel> gm_ptr = model->copySample(*node->gm);
          auto& gm = *gm_ptr;
          model_unigram->sampleOne(gm, *gm.rng, pos);
          node->gm->entropy_unigram[pos] = gm.entropy[pos];
          return node->gm->entropy_unigram[pos];
        }
      }
      return 0.0;
    };

    double nb_sure;
    int count;
    int label;

    switch(f) {
      case FEAT_BIAS:
        add_feat(1.0);
        break;
      case FEAT_COND_ENT:
        add_feat(node->gm->entropy[pos]);
        break;
      case FEAT_LOG_COND_ENT:
        add_feat(log(1e-40 + node->gm->entropy[pos]));
        break;
      case FEAT_01_COND_ENT:
        add_feat(node->gm->entropy[pos] < 1e-10 ? 1.0 : 0.0);
        break;
      case FEAT_COND_LHOOD:
        label = node->gm->getLabel(pos);
        add_feat(node->gm->this_sc[pos][label]);
        break;
      case FEAT_UNIGRAM_ENT:
        add_feat(unigram_ent());
        break;
      case FEAT_INV_UNIGRAM_ENT:
        add_feat(1 / (1e-4 + unigram_ent()));
        break;
      case FEAT_01_UNIGRAM_ENT:
        add_feat(double(unigram_ent() < 1e-8));
        break;
      case FEAT_LOGISTIC_UNIGRAM_ENT:
        add_feat(logisticFunc(unigram_ent()));
        break;
      case FEAT_NB_ENT:
        nb_sure = 0;
        count = 0;
        for (auto id : model->markovBlanket(*node->gm, pos)) {
          nb_sure += node->gm->entropy[id];
          count++;
        }
        nb_sure /= (double)count;
        add_feat(nb_sure);
        break;
      case FEAT_SP:
        add_feat(node->gm->mask[pos]);
        break;
      case FEAT_EXP_SP:
        add_feat(exp(node->gm->mask[pos]));
        break;
      case FEAT_LOG_SP:
        add_feat(log(1 + node->gm->mask[pos]));
        break;
      case FEAT_SP_COND_ENT:
        insertFeature(feat, 
                      boost::lexical_cast<string>(node->gm->mask[pos]) + "-cond-ent", 
                      node->gm->entropy[pos]);
        break;
      case FEAT_SP_UNIGRAM_ENT:
        insertFeature(feat, 
                      boost::lexical_cast<string>(node->gm->mask[pos]) + "-unigram-ent", 
                      node->gm->entropy_unigram[pos]);
        break;
      case FEAT_NB_VARY:
      case FEAT_NB_DISCORD:
      case FEAT_ORACLE:
      case FEAT_ORACLE_ENT:
      case FEAT_ORACLE_STALENESS:
        // to be added dynamically.
        add_feat(0);
        break;
      default:
        throw "unrecognized meta-feature";
    }
  }
  return feat;
}


MarkovTreeNodePtr Policy::sampleOne(MarkovTreeNodePtr node, objcokus& rng, int pos) {
  model->sampleOne(*node->gm, rng, pos);
  node->log_prior_weight += node->gm->reward[pos];

  if (node->log_prior_weight > node->max_log_prior_weight) {
    node->max_log_prior_weight = node->log_prior_weight;
    node->max_gm = model->copySample(*node->gm);
  }

  node->gm->mask[pos] += 1;

  if (lets_inplace) {
    node->depth++;
  } else {
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
  for (auto id : model->markovBlanket(*node->gm, pos)) {
    node->gm->changed[pos][id] = false;
    node->gm->vary[pos][id] = 0;
  }
  for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
    node->gm->vary[id][pos] += 1;
  }

  /* update my response */
  node->gm->resp[pos] = HeteroSampler::score(this->param, feat);
  int val = node->gm->getLabel(pos), oldval = node->gm->oldlabels[pos];

  set<int> visited;

  auto updateRespByHandle = [&] (int id) {
    if (heap == nullptr) return;
    Value& val = *node->gm->handle[id];
    val.resp = node->gm->resp[id];
    heap->update(node->gm->handle[id]);
  };
  updateRespByHandle(pos);

  auto computeOracle = [&] (int id) {
    auto feat = findFeature(node->gm->feat[id], feat_name[FEAT_ORACLE]);
    *feat = sampleDelayedReward(node, id, this->mode_oracle, this->rewardK);
    node->gm->resp[id] = HeteroSampler::score(this->param, node->gm->feat[id]);
    updateRespByHandle(id);
  };

  auto computeOracleEnt = [&] (double * feat, int id) {
    int oldval = node->gm->getLabel(id);
    model->sampleOne(*node->gm, rng, id, false);
    *feat = logEntropy(&node->gm->sc[0], node->gm->numLabels(id));
    node->gm->setLabel(id, oldval);
    node->gm->resp[id] = HeteroSampler::score(this->param, node->gm->feat[id]);
    updateRespByHandle(id);
  };

  auto computeStaleness = [&] (double * feat, int id) {
    int oldval = node->gm->getLabel(id);
    model->sampleOne(*node->gm, rng, id, false);
    *feat = node->gm->this_sc[id][oldval] - node->gm->sc[oldval];
    node->gm->setLabel(id, oldval);
    node->gm->resp[id] = HeteroSampler::score(this->param, node->gm->feat[id]);
    updateRespByHandle(id);
  };
  
  auto make_nb_discord = [] (int val, int your_val) {
    return "c-" + tostr(val) + "-" + tostr(your_val);
  };
  
  /* update my friends' response */
  for(MetaFeature f : this->feat) {
    string name = this->feat_name[f];
    switch(f) {
      case FEAT_NB_VARY:
        /* update the nodes in inv Markov blanket */
        for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if (node->gm->blanket[id].size() > 0) {
            assert(node->gm->blanket[id].contains(pos));
            double* feat_nb_vary = findFeature(node->gm->feat[id], name);
            if (node->gm->blanket[id][pos] != val and node->gm->changed[id][pos] == false) {
              node->gm->changed[id][pos] = true;
              (*feat_nb_vary)++;
              node->gm->resp[id] += (*param)[name];
              updateRespByHandle(id);
            }
            if (node->gm->blanket[id][pos] == val and node->gm->changed[id][pos] == true) {
              node->gm->changed[id][pos] = false;
              (*feat_nb_vary)--;
              node->gm->resp[id] -= (*param)[name];
              updateRespByHandle(id);
            }
          }
        }
        break;
      case FEAT_NB_DISCORD:
        for(auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if(node->gm->blanket[id].size() > 0) {
            double yourval = node->gm->getLabel(id);
            if(node->gm->vary[id][pos] > 1) { // more than the first time.
              // invalidate old feat.
              double* oldfeat = findFeature(node->gm->feat[id], make_nb_discord(yourval, oldval));
              assert(oldfeat != nullptr and *oldfeat != 0);
              node->gm->resp[id] -= (*param)[make_nb_discord(yourval, oldval)];
              (*oldfeat)--;
            }
            // insert new feat.
            double* newfeat = findFeature(node->gm->feat[id], make_nb_discord(yourval, val));
            if(newfeat == nullptr) {
              insertFeature(node->gm->feat[id], make_nb_discord(yourval, val));
            }else{
              (*newfeat)++;
            }
            node->gm->resp[id] += (*param)[make_nb_discord(yourval, val)];
            updateRespByHandle(id);
          }
        }
        break;
      case FEAT_NB_ENT:
        for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if (node->gm->blanket[id].size() > 0) {
            double* feat_nb_ent = findFeature(node->gm->feat[id], name);
            double ent_diff = (node->gm->entropy[pos] - node->gm->prev_entropy[pos])
                              / (double)node->gm->blanket[id].size();;
            (*feat_nb_ent) += ent_diff;
            node->gm->resp[id] += (*param)[name] * ent_diff;
          }
        }
        break;
      case FEAT_ORACLE:
        computeOracle(pos);
        visited.insert(pos);
        for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if (node->gm->blanket[id].size() > 0 and visited.count(id) == 0) { // has already been initialized.
            computeOracle(id);
            visited.insert(id);
          }
        }    
        break;
      case FEAT_ORACLE_ENT:
        computeOracleEnt(findFeature(feat, name), pos);
        for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if (node->gm->blanket[id].size() > 0) { // has already been initialized.
            computeOracleEnt(findFeature(node->gm->feat[id], name), id);
          }
        }
        break;
      case FEAT_ORACLE_STALENESS:
        computeStaleness(findFeature(feat, name), pos);
        for (auto id : model->invMarkovBlanket(*node->gm, pos)) {
          if (node->gm->blanket[id].size() > 0) { // has already been initialized.
            computeStaleness(findFeature(node->gm->feat[id], name), id);
          }
        }
        break;
      default:
        break;
    }
  }

  /* update Markov blanket */
  node->gm->blanket[pos] = node->gm->getLabels(model->markovBlanket(*node->gm, pos));
}


void Policy::logNode(MarkovTreeNodePtr node) {
  while (node->children.size() > 0) node = node->children[0]; // take final sample.
  lg->begin("time"); *lg << node->depth + 1 << endl; lg->end();
  lg->begin("truth"); *lg << node->gm->seq->str() << endl; lg->end();
  lg->begin("tag"); *lg << node->gm->str() << endl; lg->end();
  lg->begin("resp");
  auto logIndex = [&] (int index) {
    *lg << " / [" << boost::lexical_cast<string>(index) + "]";
  };

  for (size_t i = 0; i < node->gm->size(); i++) {
    *lg << node->gm->resp[i];
    if (verbose) {
      logIndex(i);
    }
    *lg << "\t";
  }
  *lg << endl;
  lg->end();
  if (node->gm->size() > 0 and verbose) {
    lg->begin("feat");
    /* take the uninon of features from all positions */
    std::set<string> feat_names;
    for (size_t t = 0; t < node->gm->size(); t++) {
      if (node->gm->feat[t] == nullptr) continue;
      for (auto& p : *node->gm->feat[t]) {
        feat_names.insert(string(p.first));
      }
    }
    /* log each position */
    for (auto& p : feat_names) {
      lg->begin(p);
      for (size_t i = 0; i < node->gm->size(); i++) {
        *lg << getFeature(node->gm->feat[i], p);
        if (verbose) {
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
  for (size_t i = 0; i < node->gm->size(); i++) {
    *lg << node->gm->mask[i];
    if (verbose) {
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
  while (lg != nullptr and lg->depth() > 0) lg->end();
  lg = new_lg;
}

/////////////////////////// Gibbs Policy ///////////////////////////////
GibbsPolicy::GibbsPolicy(ModelPtr model, const po::variables_map& vm)
  : Policy(model, vm), T(vm["T"].as<size_t>())
{
}

Location GibbsPolicy::policy(MarkovTreeNodePtr node) {
  if (node->depth == 0) node->time_stamp = 0;
  if (node->depth < T * node->gm->size()) {
    node->time_stamp++;

    return Location(node->depth % node->gm->size());
  }
  return Location(); // stop.
}

/////////////////////////// Block Policy ///////////////////////////////
BlockPolicy::BlockPolicy(ModelPtr model, const variables_map& vm)
  : GibbsPolicy(model, vm) {

}


BlockPolicy::~BlockPolicy() {

}


BlockPolicy::Result::Result(ptr<Corpus> corpus)
  : Policy::Result::Result(corpus) {

}


ptr<Policy::Result> BlockPolicy::test(ptr<Corpus> corpus) {
  return this->test(corpus, 1.0);
}


ptr<BlockPolicy::Result> BlockPolicy::test(ptr<Corpus> corpus, double budget) {
  auto result = std::make_shared<BlockPolicy::Result>(corpus);
  result->corpus->retag(model->corpus);
  result->nodes.resize(fmin((size_t)test_count, (size_t)corpus->seqs.size()), nullptr);

  for (size_t i = 0; i < result->size(); i++) {
    auto node = makeMarkovTreeNode(nullptr);
    node->model = this->model;
    node->gm = model->makeSample(*corpus->seqs[i], model->corpus, &rng);
    node->log_prior_weight = model->score(*node->gm);
    result->nodes[i] = node;
    for (int t = 0; t < node->gm->size(); t++) {
      node->gm->resp[t] = 1e8 - t; // this is a hack.
      Heap::handle_type handle = result->heap.push(Value(Location(i, t), node->gm->resp[t]));
      node->gm->handle[t] = handle;
    }
  }

  result->time = 0;
  result->wallclock = 0;
  test(result, budget);
  return result;
}


void BlockPolicy::test(Policy::ResultPtr result) {
  auto block_result = static_pointer_cast<BlockPolicy::Result>(result);
  this->test(block_result, 1.0);
}


void BlockPolicy::test(BlockPolicy::ResultPtr result, double budget) {
  *auxlg << "> test " << std::endl;
  lg->begin("test");
  lg->begin("param");
  *lg << *param;
  lg->end(); // </param>
  if (budget > 0) {
    this->test_policy(result, budget);
  }
  lg->end(); // </test>
}



void BlockPolicy::test_policy(ptr<BlockPolicy::Result> result, double budget) {
  clock_t time_start = clock(), time_end;
  assert(result != nullptr);
  double total_budget = result->corpus->count(test_count) * budget;
  for (size_t b = 0; b < total_budget; b++) {
    auto p = policy(result);
    result->setNode(p.index, this->sampleOne(result, this->rng, p));
  }
  double hit_count = 0, pred_count = 0, truth_count = 0;
  this->lg->begin("example");
  for (size_t i = 0; i < result->size(); i++) {
    MarkovTreeNodePtr node = result->getNode(i);
    lg->begin("example_" + std::to_string(i));
    this->logNode(node);
    while (node->children.size() > 0) node = node->children[0]; // take final sample.
    if (this->model->scoring == Model::SCORING_ACCURACY) {
      tuple<int, int> hit_pred = this->model->evalPOS(*cast<Tag>(node->max_gm));
      hit_count += std::get<0>(hit_pred);
      pred_count += std::get<1>(hit_pred);
    } else if (this->model->scoring == Model::SCORING_NER) {
      tuple<int, int, int> hit_pred_truth = this->model->evalNER(*cast<Tag>(node->max_gm));
      hit_count += std::get<0>(hit_pred_truth);
      pred_count += std::get<1>(hit_pred_truth);
      truth_count += std::get<2>(hit_pred_truth);
    } else if (this->model->scoring == Model::SCORING_LHOOD) {
      hit_count += this->model->score(*node->max_gm);
      pred_count++;
    }
    lg->end(); // </example_i>
  }
  lg->end(); // </example>

  /* log summary stats */
  time_end = clock();
  double accuracy = (double)hit_count / pred_count;
  double recall = (double)hit_count / truth_count;
  result->time += total_budget / result->size();
  result->wallclock += (double)(time_end - time_start) / CLOCKS_PER_SEC;
  lg->begin("time");
  *auxlg << "time: " << result->time << std::endl;
  *lg << result->time << std::endl;
  lg->end(); // </time>
  lg->begin("wallclock");
  *auxlg << "wallclock: " << result->wallclock << std::endl;
  *lg << result->wallclock << std::endl;
  lg->end(); // </wallclock>
  lg->begin("wallclock_sample");
  *auxlg << "wallclock_sample: " << result->wallclock_sample << std::endl;
  *lg << result->wallclock_sample << std::endl;
  lg->end();
  lg->begin("wallclock_policy");
  *auxlg << "wallclock_policy: " << result->wallclock_policy << std::endl;
  *lg << result->wallclock_policy << std::endl;
  lg->end();
  if (this->model->scoring == Model::SCORING_ACCURACY) {
    lg->begin("accuracy");
    *lg << accuracy << std::endl;
    *auxlg << "acc: " << accuracy << std::endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  } else if (this->model->scoring == Model::SCORING_NER) {
    double f1 = 2 * accuracy * recall / (accuracy + recall);
    lg->begin("accuracy");
    *lg << f1 << std::endl;
    *auxlg << "f1: " << f1 << std::endl;
    lg->end(); // </accuracy>
    result->score = f1;
  } else if (this->model->scoring == Model::SCORING_LHOOD) {
    lg->begin("accuracy");
    *lg << accuracy << std::endl;
    *auxlg << "lhood: " << accuracy << std::endl;
    lg->end(); // </accuracy>
    result->score = accuracy;
  }
}


void BlockPolicy::test_policy(Policy::ResultPtr result) {
  auto block_result = static_pointer_cast<BlockPolicy::Result>(result);
  this->test_policy(block_result, 1.0);
}

void BlockPolicy::sample(int tid, MarkovTreeNodePtr node) {
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
        double log_resp = HeteroSampler::score(param, feat); // fix: param always changes, so does resp.
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

        
       if(lets_resp_reward) {  // record training examples.
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

MarkovTreeNodePtr BlockPolicy::sampleOne(ptr<BlockPolicy::Result> result,
    objcokus& rng,
    const Location& loc) {
  clock_t clock_start = clock(), clock_end;
  int index = loc.index, pos = loc.pos;
  MarkovTreeNodePtr node = result->getNode(index);
  node->gm->rng = &rng;
  node = Policy::sampleOne(node, rng, pos);
  clock_end = clock();
  result->wallclock_sample += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  clock_start = clock();
  Policy::updateResp(node, rng, pos, &result->heap);
  clock_end = clock();
  result->wallclock_policy += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  return node;
}


Location BlockPolicy::policy(BlockPolicy::ResultPtr result) {
  clock_t clock_start = clock(), clock_end;
  Location loc;
  Value val = result->heap.top();
  clock_end = clock();
  result->wallclock_policy += (double)(clock_end - clock_start) / CLOCKS_PER_SEC;
  return val.loc;
}



}