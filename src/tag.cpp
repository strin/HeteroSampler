#include "tag.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <sstream>

using namespace std;
namespace Tagging {
  Tag::Tag(const Sentence* seq, ptr<Corpus> corpus, 
	  objcokus* rng, ParamPointer param) 
  :seq(seq), corpus(corpus), rng(rng), param(param) {
    this->randomInit();
  }

  Tag::Tag(const Sentence& seq, ptr<Corpus> corpus, 
	  objcokus* rng, ParamPointer param)
  :seq(&seq), corpus(corpus), rng(rng), param(param) {
    this->tag = seq.tag;

  }

  void Tag::randomInit() {
    int taglen = corpus->tags.size();
    int seqlen = seq->seq.size();
    tag.resize(seqlen);
    resp.resize(seqlen);
    mask.resize(seqlen);
    entropy.resize(seqlen);
    for(int& t : tag) {
      t = rng->randomMT() % taglen;
    }
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

  ParamPointer Tag::proposeGibbs(int pos, function<FeaturePointer(const Tag& tag)>
  featExtract, bool grad_expect, bool grad_sample, bool argmax) {
    const vector<TokenPtr>& sen = seq->seq;
    int seqlen = sen.size();
    if(pos >= seqlen) 
      throw "Gibbs sampling proposal out of bound.";
    int taglen = corpus->tags.size();
    double sc[taglen];
    vector<FeaturePointer> featvec;
    for(int t = 0; t < taglen; t++) {
      tag[pos] = t;
      FeaturePointer features = featExtract(*this);
      featvec.push_back(features);
      sc[t] = this->score(features);
    }
    logNormalize(sc, taglen);
    // gather stats.
    this->entropy[pos] = logEntropy(sc, taglen);
    this->sc.clear();
    for(int t = 0; t < taglen; t++) { 
      if(std::isnan(sc[t])) 
	  cout << "nan " << endl;
      this->sc.push_back(sc[t]);
    }

    int val;
    /*for(int t = 0; t < taglen; t++) {
      cout << "t = " << t << " , " << exp(sc[t]) << endl;
    }*/
    if(argmax) {
      double max_sc = -DBL_MAX;
      for(int t = 0; t < taglen; t++) {
	if(sc[t] > max_sc) {
	  max_sc = sc[t];
	  val = t;
	}
      }
    }else
      val = rng->sampleCategorical(sc, taglen);
    if(val == taglen) throw "Gibbs sample out of bound.";
    tag[pos] = val;
    this->features = featExtract(*this);
    ParamPointer gradient = makeParamPointer();
    if(grad_sample)
      mapUpdate<double, double>(*gradient, *this->features);
    if(grad_expect) {
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

  string Tag::getTag(size_t pos) const {
    return this->corpus->invtags[tag[pos]];
  }

  double Tag::score(FeaturePointer features) const {
    double score = 0;
    for(const pair<string, double>& feat : *features) {
      if(this->param->find(feat.first) != this->param->end()) { 
	score += feat.second * (*this->param)[feat.first];
      }
    }
    return score;
  }

  string Tag::str() {
    string ss;
    size_t seqlen = seq->seq.size();
    for(size_t i = 0; i < seqlen; i++) {
      ss += seq->seq[i]->str();
      ss += " / ";
      ss += this->getTag(i);
      ss += "\t";
    }
    return ss;
  }
}
