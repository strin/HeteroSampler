#include "MarkovTree.h"

using namespace std;

namespace Tagging {
  ptr<MarkovTreeNode> makeMarkovTreeNode(std::shared_ptr<MarkovTreeNode> parent, const GraphicalModel& gm) {
    auto node = MarkovTreeNodePtr(new MarkovTreeNode(parent));
    assert(node->model != nullptr);
    node->gm = node->model->copySample(gm);
    return node;
  }


  MarkovTreeNode::MarkovTreeNode(shared_ptr<MarkovTreeNode> parent)
  :parent(parent), log_weight(-DBL_MAX) {
    if(parent == nullptr) {
      depth = 0;
      time_stamp = 0;
      model = nullptr;
      log_prior_weight = 0;
      max_log_prior_weight = -DBL_MAX;
      gm = nullptr;
      max_gm = nullptr;
    }else{
      depth = parent->depth+1;
      time_stamp = parent->time_stamp;
      model = parent->model;
      log_prior_weight = parent->log_prior_weight;
      max_log_prior_weight = parent->max_log_prior_weight;
      max_gm = parent->max_gm;
      gm = parent->gm;
    }
    gradient = posgrad = neggrad = nullptr;
    compute_stop = false;
  }

  bool MarkovTreeNode::is_split() {
    return this->children.size() >= 2;
  }

  bool MarkovTreeNode::is_leaf() {
    return this->children.size() == 0;
  }

  MarkovTree::MarkovTree() 
  :root(new MarkovTreeNode(nullptr)) {
  }

  /* compute gradients */
  double MarkovTree::logSumWeights(shared_ptr<MarkovTreeNode> node) {
    double weight = node->log_weight;
    for(shared_ptr<MarkovTreeNode> child : node->children) 
      weight = logAdd(weight, logSumWeights(child));
    return weight;
  }

  pair<ParamPointer, double> MarkovTree::aggregateGradient(shared_ptr<MarkovTreeNode> node, double normalize) {
    ParamPointer gradient = makeParamPointer();
    double weight = node->log_weight-normalize, weight_descendant = -DBL_MAX;
    for(shared_ptr<MarkovTreeNode> child : node->children) {
      pair<ParamPointer, double> result = aggregateGradient(child, normalize); 
      weight_descendant = logAdd(weight_descendant, result.second);
      mapUpdate(*gradient, *result.first);
    }
    if(node->posgrad != nullptr)
      mapUpdate(*gradient, *node->posgrad, exp(weight));
    if(node->neggrad != nullptr)
      mapUpdate(*gradient, *node->neggrad, exp(weight_descendant));
    weight = logAdd(weight, weight_descendant);
    if(node->gradient != nullptr) 
      mapUpdate(*gradient, *node->gradient, exp(weight));
    return pair<ParamPointer, double>(gradient, weight);
  }

  ParamPointer MarkovTree::expectedGradient() {
    return aggregateGradient(root, logSumWeights(root)).first;
  }

  double MarkovTree::logSumPriorWeights(shared_ptr<MarkovTreeNode> node, size_t max_level) {
    double weight = node->log_prior_weight;
    if(node->depth < max_level) {
      for(shared_ptr<MarkovTreeNode> child : node->children) 
        weight = logAdd(weight, logSumPriorWeights(child, max_level));
    }
    return weight;
  }

  double MarkovTree::aggregateReward(shared_ptr<MarkovTreeNode> node, double normalize, size_t max_level) {
    double reward = node->log_prior_weight - normalize + node->log_weight;
    if(node->depth < max_level) {
      for(shared_ptr<MarkovTreeNode> child : node->children) {
        reward = logAdd(reward, aggregateReward(child, normalize, max_level));
      }
    }
    return reward;
  }

  ptrs<GraphicalModel> MarkovTree::aggregateSample(MarkovTreeNodePtr node, size_t max_level) {
    ptrs<GraphicalModel> ret;
    if(node->is_leaf() || node->depth == max_level) ret.push_back(node->gm);
    else if(node->depth < max_level) {
      for(MarkovTreeNodePtr child : node->children) {
        auto this_ret = aggregateSample(child, max_level);
        ret.insert(ret.end(), this_ret.begin(), this_ret.end());
      }
    }
    return ret;
  }


  ptrs<GraphicalModel> MarkovTree::getSamples(shared_ptr<MarkovTreeNode> node) { 
    ptrs<GraphicalModel> vec;
    if(node->children.size() > 0) {
      for(auto child : node->children) {
        auto res = getSamples(child); 
        vec.insert(vec.begin(), res.begin(), res.end()); 
      }
      return vec;
    }
    vec.push_back(node->gm);
    return vec;
  }

  ptrs<GraphicalModel> MarkovTree::getSamples() {
    return this->getSamples(this->root);
  }
}
