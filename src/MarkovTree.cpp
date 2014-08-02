#include "MarkovTree.h"

using namespace std;

MarkovTreeNode::MarkovTreeNode(shared_ptr<MarkovTreeNode> parent)
:parent(parent), log_weight(-DBL_MAX) {
  if(parent == nullptr) depth = 0;
  else depth = parent->depth+1;
  gradient = posgrad = neggrad = nullptr;
  tag = nullptr;
}

bool MarkovTreeNode::is_split() {
  return this->children.size() >= 2;
}

MarkovTree::MarkovTree() 
:root(new MarkovTreeNode(nullptr)) {
}

double MarkovTree::logSumWeights(shared_ptr<MarkovTreeNode> node) {
  double weight = node->log_weight;
  for(shared_ptr<MarkovTreeNode> child : node->children) 
    weight = logAdd(weight, logSumWeights(child));
  return weight;
}

double MarkovTree::logSumPriorWeights(shared_ptr<MarkovTreeNode> node) {
  double weight = node->log_prior_weight;
  for(shared_ptr<MarkovTreeNode> child : node->children) 
    weight = logAdd(weight, logSumPriorWeights(child));
  return weight;
}

double MarkovTree::aggregateReward(shared_ptr<MarkovTreeNode> node, double normalize) {
  double reward = node->log_prior_weight - normalize + node->log_weight;
  for(shared_ptr<MarkovTreeNode> child : node->children) {
    reward = logAdd(reward, aggregateReward(child, normalize));
  }
  return reward;
}

StopDatasetPtr MarkovTree::generateStopDataset(MarkovTreeNodePtr node) {
  StopDatasetPtr stop_data = makeStopDataset();
  if(node->is_split()) {
    double reward = 0;
    for(MarkovTreeNodePtr child : node->children) { 
      double weight = logSumPriorWeights(child);
      reward += aggregateReward(child, weight);
    }
    reward /= (double)node->children.size();
    incrStopDataset(stop_data, node->stop_feat, node->log_weight-reward); 
    return stop_data;
  }
  for(MarkovTreeNodePtr child : node->children) {
    mergeStopDataset(stop_data, generateStopDataset(child));
  }
  return stop_data;
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
  mapUpdate(*gradient, *node->gradient, exp(logAdd(weight, weight_descendant)));
  return pair<ParamPointer, double>(gradient, weight);
}

ParamPointer MarkovTree::expectedGradient() {
  return aggregateGradient(root, logSumWeights(root)).first;
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
