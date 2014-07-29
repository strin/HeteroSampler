#include "tag.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;

Tag::Tag(const Sentence* seq, const Corpus& corpus, 
	objcokus* rng, ParamPointer param) 
:seq(seq), corpus(corpus), rng(rng), param(param) {
  this->randomInit();
}

void Tag::randomInit() {
  int taglen = corpus.tags.size();
  int seqlen = seq->seq.size();
  tag.resize(seqlen);
  for(int& t : tag) {
    t = rng->randomMT() % taglen;
  }
}

FeaturePointer Tag::extractSimpleFeatures(const vector<int>& tag, int pos) {
  FeaturePointer features = makeFeaturePointer();
  const vector<Token>& sen = seq->seq;
  // extract word features only.
  stringstream ss;
  ss << "simple-" << sen[pos].word << "-" << tag[pos];
  (*features)[ss.str()] = 1;
  return features;
}

FeaturePointer Tag::extractFeatures(const vector<int>& tag) {
  FeaturePointer features = makeFeaturePointer();
  const vector<Token>& sen = seq->seq;
  int seqlen = sen.size();
  // extract word features. 
  for(int si = 0; si < seqlen; si++) {
    stringstream ss;
    ss << sen[si].word << "-" << tag[si];
    (*features)[ss.str()] = 1;
  }
  // extract bigram features.
  for(int si = 1; si < seqlen; si++) {
    stringstream ss;
    ss << "p-" << tag[si-1] << "-" << tag[si];
    (*features)[ss.str()] = 1;
  }
  return features;
}

string str(FeaturePointer features) {
  stringstream ss;
  ss << "[";
  for(const pair<string, double>& p : *features) {
    ss << p.first << "\t";
  }
  ss << "]";
  return ss.str();
}

ParamPointer Tag::proposeSimple(int pos, bool withgrad) {
  const vector<Token>& sen = seq->seq;
  size_t seqlen = sen.size();
  if(pos > seqlen)
    throw "Simple model proposal out of boundary";
  size_t taglen = corpus.tags.size();
  vector<FeaturePointer> featvec;
  double sc[taglen];
  for(size_t t = 0; t < taglen; t++) {
    tag[pos] = t;
    FeaturePointer features = this->extractSimpleFeatures(this->tag, pos);
    featvec.push_back(features);
    sc[t] = this->score(features);
  }
  logNormalize(sc, taglen);
  int val = rng->sampleCategorical(sc, taglen);
  tag[pos] = val;
  this->features = this->extractSimpleFeatures(this->tag, pos);
  ParamPointer gradient = makeParamPointer();
  mapUpdate<double, double>(*gradient, *this->features);
  if(withgrad) {
    for(size_t t = 0; t < taglen; t++) {
      mapUpdate<double, double>(*gradient, *featvec[t], -exp(sc[t]));
    }
  }
  return gradient;
}

ParamPointer Tag::proposeGibbs(int pos, bool withgrad) {
  const vector<Token>& sen = seq->seq;
  int seqlen = sen.size();
  if(pos >= seqlen) 
    throw "Gibbs sampling proposal out of boundary.";
  int taglen = corpus.tags.size();
  double sc[taglen];
  vector<FeaturePointer> featvec;
  for(int t = 0; t < taglen; t++) {
    tag[pos] = t;
    FeaturePointer features = this->extractFeatures(this->tag);
    featvec.push_back(features);
    sc[t] = this->score(features);
  }
  logNormalize(sc, taglen);
  int val = rng->sampleCategorical(sc, taglen);
  tag[pos] = val;
  this->features = this->extractFeatures(this->tag);
  ParamPointer gradient(new map<string, double>());
  mapUpdate<double, double>(*gradient, *this->features);
  if(withgrad) {
    for(int t = 0; t < taglen; t++) {
      mapUpdate<double, double>(*gradient, *featvec[t], -exp(sc[t]));
    }
  }
  return gradient;
}

double Tag::distance(const Tag& tag) {
  if(this->size() != tag.size())
    throw "should not compare tags with different length.";
  double dist = 0;
  for(size_t t = 0; t < this->size(); t++) {
    dist += (this->tag[t] != tag.tag[t]);
  }
  return dist;
}

double Tag::score(FeaturePointer features) const {
  double score = 0;
  for(const pair<string, double>& feat : *features) {
    if(this->param->find(feat.first) != this->param->end()) 
      score += feat.second * (*this->param)[feat.first];
  }
  return score;
}

string Tag::str() {
  stringstream ss;
  size_t seqlen = seq->seq.size();
  for(size_t i = 0; i < seqlen; i++) {
    ss << seq->seq[i].word << "/" << corpus.invtags.find(tag[i])->second << "\t";
  }
  return ss.str();
}
