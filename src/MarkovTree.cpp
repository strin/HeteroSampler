#include "MarkovTree.h"

using namespace std;

MarkovTreeNode::MarkovTreeNode(shared_ptr<MarkovTreeNode> parent)
:parent(parent), log_weight(-DBL_MAX) {
  if(parent == nullptr) {
    depth = 0;
    time_stamp = 0;
  }else{
    depth = parent->depth+1;
    time_stamp = parent->time_stamp;
  }
  gradient = posgrad = neggrad = nullptr;
  tag = nullptr;
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

/* Generate stop datset */
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

vector<shared_ptr<Tag> > MarkovTree::aggregateTag(MarkovTreeNodePtr node, size_t max_level) {
  vector<shared_ptr<Tag> > ret;
  if(node->is_leaf() || node->depth == max_level) ret.push_back(node->tag);
  else if(node->depth < max_level) {
    for(MarkovTreeNodePtr child : node->children) {
      vector<shared_ptr<Tag> > this_ret = aggregateTag(child, max_level);
      ret.insert(ret.end(), this_ret.begin(), this_ret.end());
    }
  }
  return ret;
}

StopDatasetPtr MarkovTree::generateStopDataset(MarkovTreeNodePtr node, int mode) {
  StopDatasetPtr stop_data = makeStopDataset();
  auto getFutureReward = [&] (MarkovTreeNodePtr node, size_t max_level) {
    double reward = -DBL_MAX;
    for(MarkovTreeNodePtr child : node->children) { 
      double weight = logSumPriorWeights(child, max_level);
      reward = logAdd(reward, aggregateReward(child, weight, max_level));
    }
    reward = reward-log(node->children.size());
    return reward;
  };
  TagVector child_vec, grandson_vec;
  if(node->compute_stop) {
    double reward_now, reward_future;
    TagVector vec;
    switch(mode) {
    case -1:
      throw "not implemented";
      break;
    case 0:
      reward_now = node->log_weight;
      reward_future = getFutureReward(node, -1);
      vec.push_back(node->tag);
      vec.push_back(nullptr);
      child_vec = aggregateTag(node);
      vec.insert(vec.end(), child_vec.begin(), child_vec.end());
      break;
    case 1:
      reward_now = getFutureReward(node, node->depth+1);
      reward_future = getFutureReward(node, -1); 
      child_vec = aggregateTag(node, node->depth+1); 
      vec.insert(vec.end(), child_vec.begin(), child_vec.end());
      vec.push_back(nullptr); // sentinel.
      grandson_vec = aggregateTag(node, -1);
      vec.insert(vec.end(), grandson_vec.begin(), grandson_vec.end());
      break;
    }
    incrStopDataset(stop_data, node->stop_feat, reward_now, reward_future, vec); 
    return stop_data;
  }
  for(MarkovTreeNodePtr child : node->children) {
    mergeStopDataset(stop_data, generateStopDataset(child, mode));
  }
  return stop_data;
}


TagVector MarkovTree::getSamples(shared_ptr<MarkovTreeNode> node) { 
  if(node->children.size() > 0) {
    TagVector vec;
    for(auto child : node->children) {
      TagVector res = getSamples(child); 
      vec.insert(vec.begin(), res.begin(), res.end()); 
    }
    return vec;
  }
  TagVector vec;
  vec.push_back(node->tag);
  return vec;
}

TagVector MarkovTree::getSamples() {
  return this->getSamples(this->root);
}
