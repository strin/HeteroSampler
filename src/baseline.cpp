/* implementation of baseline sequence tagging models, including 
 * > independent logistic regression.
 * > CRF with Gibbs sampling. 
 */
#include "model.h"
#include "feature.h"
#include <boost/program_options.hpp>

using namespace std;
using namespace std::placeholders;
namespace po = boost::program_options;

////////// Simple Model (Independent Logit) ////////////
ModelSimple::ModelSimple(const Corpus* corpus, const po::variables_map& vm) 
:Model(corpus, vm), windowL(vm["windowL"].as<int>()),
 depthL(vm["depthL"].as<int>()) {
  xmllog.begin("windowL"); xmllog << windowL << endl; xmllog.end();
}

void ModelSimple::sample(Tag& tag, int time) {
  for(int t = 0; t < time; t++) {
    for(size_t i = 0; i < tag.size(); i++) {
      auto featExtract = [&] (const Tag& tag) -> FeaturePointer {
			    return this->extractFeatures(tag, i); 
			  };
      tag.proposeGibbs(i, featExtract, false, false);
    }
  }
}

TagVector ModelSimple::sample(const Sentence& seq) {
  TagVector vec;
  gradient(seq, &vec, false);
  return vec;
}

FeaturePointer ModelSimple::extractFeatures(const Tag& tag, int pos) {
  FeaturePointer features = makeFeaturePointer();
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  // extract word features only.
  for(int l = max(0, pos - windowL); l <= min(pos + windowL, seqlen-1); l++) {
    StringVector nlp = NLPfunc(sen[l].word);
    for(const string& token : *nlp) {
      stringstream ss;
      ss << "simple-w-" << to_string(l-pos) 
	 << "-" << token << "-" << corpus->invtag(tag.tag[pos]);
      (*features)[ss.str()] = 1;
    }
  }
  return features;
}

ParamPointer ModelSimple::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

ParamPointer ModelSimple::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag truth(seq, corpus, &rngs[0], param);
  ParamPointer gradient = makeParamPointer();
  for(size_t i = 0; i < tag.size(); i++) {
    auto featExtract = [&] (const Tag& tag) -> FeaturePointer {
			  return this->extractFeatures(tag, i); 
			};
    ParamPointer g = tag.proposeGibbs(i, featExtract, true, false);
    if(update_grad) {
      mapUpdate<double, double>(*gradient, *g);
      mapUpdate<double, double>(*gradient, *featExtract(truth));
    }
  }
  if(samples)
    samples->push_back(shared_ptr<Tag>(new Tag(tag)));
  else{
    xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
    xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  }
  return gradient;
}

//////// Model CRF Gibbs ///////////////////////////////
ModelCRFGibbs::ModelCRFGibbs(const Corpus* corpus, const po::variables_map& vm)
:ModelSimple(corpus, vm), 
 extractFeatures([&] (const Tag& tag, int pos) {
   const vector<Token>& sen = tag.seq->seq;
   int seqlen = tag.size();
   // extract word features. 
   FeaturePointer features = makeFeaturePointer();
   extractUnigramFeature(tag, pos, windowL, depthL, features);
   if(pos >= 1) {
     extractBigramFeature(tag, pos, features);
   }
   if(pos < seqlen-1) {
     extractBigramFeature(tag, pos+1, features);
   }
   return features;
 }) {}

void ModelCRFGibbs::sample(Tag& tag, int time) {
  for(int t = 0; t < time; t++) {
    this->sampleOneSweep(tag);  
  }
}

void ModelCRFGibbs::sampleOne(Tag& tag, int choice) {
  if(choice >= tag.size())
    throw "kernel choice invalid (>= tag size)";
  tag.proposeGibbs(choice, [&] (const Tag& tag) -> FeaturePointer {
			return this->extractFeatures(tag, choice); 
		      }, false, false);
}

TagVector ModelCRFGibbs::sample(const Sentence& seq) { 
  TagVector vec;
  TagPtr tag = makeTagPtr(&seq, corpus, &rngs[0], param);
  this->sample(*tag, T);
  vec.push_back(tag);
  return vec;
}

double ModelCRFGibbs::score(const Tag& tag) {
  FeaturePointer feat = this->extractFeaturesAll(tag);
  return ::score(this->param, feat);
}

void ModelCRFGibbs::addUnigramFeatures(const Tag& tag, int pos, FeaturePointer features) {
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  // word-tag potential.
  for(int l = max(0, pos - windowL); l <= min(pos + windowL, seqlen-1); l++) {
    StringVector nlp = NLPfunc(sen[l].word);
    for(const string& token : *nlp) {
      stringstream ss;
      ss << "w-" << to_string(l-pos) 
	 << "-" << token << "-" << corpus->invtag(tag.tag[pos]);
      if(token == sen[l].word)
	(*features)[ss.str()] = 1;
      else
	(*features)[ss.str()] = 1;
    }
    /*if(corpus->mode == Corpus::MODE_NER) { // add pos tags for NER task.
      stringstream ss;
      ss << "t-" << to_string(l-pos)
         << "-" << sen[l].pos << "-" << tag.getTag(pos);
      (*features)[ss.str()] = 1;
      ss.clear();
      ss << "t2-" << to_string(l-pos)
         << "-" << sen[l].pos2 << "-" << tag.getTag(pos);
      (*features)[ss.str()] = 1;
    }*/
  }
}

void ModelCRFGibbs::addBigramFeatures(const Tag& tag, int pos, FeaturePointer features) {
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  stringstream ss;
  ss << "p-" << tag.getTag(pos-1) << "-" 
  	<< tag.getTag(pos);
  // ss << "p-" << tag.tag[pos-1] << "-" << tag.tag[pos];
  /* StringVector nlp = NLPfunc(sen[pos].word);
  for(const string& token : *nlp) {
    ss << "p2-" << tag.tag[pos-1] << "-" << tag.tag[pos] << "-" << token;
  }*/
  (*features)[ss.str()] = 1;
}

/* FeaturePointer ModelCRFGibbs::extractFeatures(const Tag& tag, int pos) {
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  // extract word features. 
  FeaturePointer features = makeFeaturePointer();
  
  // this->addUnigramFeatures(tag, pos, features);
  // extract bigram features.
  // if(pos >= 1) {
  //   addBigramFeatures(tag, pos, features); 
  // }
  // if(pos < seqlen-1) {
  //   addBigramFeatures(tag, pos+1, features);
  // }
  int depth = 0;
  if(corpus->mode == Corpus::MODE_NER) 
    depth = 2;
  extractUnigramFeature(tag, pos, windowL, depth, features);
  if(pos >= 1) {
    extractBigramFeature(tag, pos, features);
  }
  if(pos < seqlen-1) {
    extractBigramFeature(tag, pos+1, features);
  }
  return features;
}*/

FeaturePointer ModelCRFGibbs::extractFeaturesAll(const Tag& tag) {
  FeaturePointer features = makeFeaturePointer();
  size_t seqlen = tag.size();
  for(size_t t = 0; t < seqlen; t++) {
    FeaturePointer this_feat = extractFeatures(tag, t);
    mapCopy(*features, *this_feat);
  }
  return features;
}

void ModelCRFGibbs::sampleOneSweep(Tag& tag) {
  for(int i = 0; i < tag.tag.size(); i++) { 
    tag.proposeGibbs(i, [&] (const Tag& tag) -> FeaturePointer {
			  return this->extractFeatures(tag, i); 
			}, false, false);
  }
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag truth(seq, corpus, &rngs[0], param);
  FeaturePointer feat = this->extractFeaturesAll(truth);
  ParamPointer gradient = makeFeaturePointer();
  for(int t = 0; t < T; t++) {
    this->sampleOneSweep(tag);
    if(t < B) continue;
    if(update_grad)
      mapUpdate<double, double>(*gradient, *this->extractFeaturesAll(tag));
  }
  if(samples)
    samples->push_back(shared_ptr<Tag>(new Tag(tag)));
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  if(update_grad) {
    mapDivide<double>(*gradient, -(double)(T-B));
    mapUpdate<double, double>(*gradient, *feat);
  }
  return gradient;
}

////////// Incremental Gibbs Sampling /////////////////////////
ModelIncrGibbs::ModelIncrGibbs(const Corpus* corpus, const po::variables_map& vm)
:ModelCRFGibbs(corpus, vm) {
}

TagVector ModelIncrGibbs::sample(const Sentence& seq) {
  TagVector samples;
  gradient(seq, &samples, false);
  return samples;
}

ParamPointer ModelIncrGibbs::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, false);
}

ParamPointer ModelIncrGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag mytag(tag);
  Tag truth(seq, corpus, &rngs[0], param);
  FeaturePointer feat = this->extractFeaturesAll(truth);
  ParamPointer gradient = makeFeaturePointer();
  for(int i = 0; i < seq.tag.size(); i++) {
    ParamPointer g = tag.proposeGibbs(i, [&] (const Tag& tag) -> FeaturePointer {
				      return this->extractFeatures(tag, i);  
				    }, true, false);
    mytag.tag[i] = tag.tag[i];
    if(update_grad) {
      mapUpdate<double, double>(*gradient, *g);
      tag.tag[i] = seq.tag[i];
      mapUpdate<double, double>(*gradient, *this->extractFeatures(tag, i)); 
    }
  }
  if(samples) {
    samples->push_back(shared_ptr<Tag>(new Tag(mytag)));
  }else{
    xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
    xmllog.begin("tag"); xmllog << mytag.str() << endl; xmllog.end();
  }
  return gradient;
}

///////// Forward-Backward Algorithm ////////////////////////////
ModelFwBw::ModelFwBw(const Corpus* corpus, const po::variables_map& vm) 
:ModelCRFGibbs(corpus, vm) { 
}

ParamPointer ModelFwBw::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag truth(seq, corpus, &rngs[0], param);
  size_t seqlen = tag.size(), taglen = corpus->tags.size();
  // dp[i][j] is the multiplicative factor for the marginal 
  // of S_{i+1:end} starting with S_i being character j. (use log-space)
  double a[seqlen][taglen], b[seqlen][taglen], phi[seqlen][taglen][taglen];
  // compute phi.
  FeaturePointer feat = makeFeaturePointer(), bifeat = makeFeaturePointer();
  for(int i = 0; i < seqlen; i++) {
    for(size_t c = 0; c < taglen; c++) {
      tag.tag[i] = c;
      feat->clear();
      this->addUnigramFeatures(tag, i, feat);
      for(size_t s = 0; s < taglen; s++) {
	bifeat->clear();
	if(i >= 1) {
	  tag.tag[i-1] = s;
	  this->addBigramFeatures(tag, i, bifeat); 
	}
	mapCopy(*feat, *bifeat);
	phi[i][c][s] = tag.score(feat);	
	mapRemove(*feat, *bifeat);
      }
    }
  }
  // forward.
  for(size_t c = 0; c < taglen; c++) a[0][c] = phi[0][c][0];
  for(int i = 1; i < seqlen; i++) {
    for(size_t c = 0; c < taglen; c++) {
      a[i][c] = -DBL_MAX;
      for(size_t s = 0; s < taglen; s++) {
	a[i][c] = logAdd(a[i][c], a[i-1][s] + phi[i][c][s]);
      }
    }
  }
  // backward. 
  for(size_t c = 0; c < taglen; c++) b[seqlen-1][c] = 0.0;
  for(int i = seqlen-2; i >= 0; i--) {
    for(size_t c = 0; c < taglen; c++) {
      b[i][c] = -DBL_MAX;
      for(size_t s = 0; s < taglen; s++) {
	b[i][c] = logAdd(b[i][c], b[i+1][s] + phi[i+1][s][c]);
      }
    }
  }
  double Z = -DBL_MAX;
  for(size_t c = 0; c < taglen; c++) 
    Z = logAdd(Z, a[seqlen-1][c]);
  // compute gradient.
  ParamPointer gradient = makeParamPointer(); 
  if(update_grad) {
    for(int i = 0; i < seqlen; i++) {
      for(size_t c = 0; c < taglen; c++) {
	tag.tag[i] = c;
	feat->clear();
	this->addUnigramFeatures(tag, i, feat);
	mapUpdate(*gradient, *feat, - exp(a[i][c] + b[i][c] - Z));
	if(i >= 1) {
	  for(size_t s = 0; s < taglen; s++) {
	    tag.tag[i-1] = s;
	    bifeat->clear();
	    this->addBigramFeatures(tag, i, bifeat);
	    mapUpdate(*gradient, *bifeat, - exp(a[i-1][s] + phi[i][c][s] + b[i][c] - Z));
	  }
	}
      }
      // this->addUnigramFeatures(tag, pos, features);
    }
    FeaturePointer truth_feat = makeFeaturePointer();
    mapUpdate(*gradient, *this->extractFeaturesAll(truth));
    mapUpdate(*gradient, *truth_feat);
  }
  // sample backward (DO NOT sample from marginal!).
  double sc[taglen];
  for(int i = seqlen-1; i >= 0; i--) { 
    for(size_t c = 0; c < taglen; c++) {
      if(i == seqlen-1)
	sc[c] = a[i][c];
      else{
	tag.tag[i] = c;
	bifeat->clear();
	this->addBigramFeatures(tag, i+1, bifeat);
	sc[c] = a[i][c] + tag.score(bifeat); 
      }
    }
    logNormalize(sc, taglen);
    tag.tag[i] = rngs[0].sampleCategorical(sc, taglen);
  }
  if(samples) {
    samples->push_back(shared_ptr<Tag>(new Tag(tag)));
  }else{
    xmllog.begin("truth"); xmllog << truth.str() << endl; xmllog.end();
    xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  }
  return gradient;
}

ParamPointer ModelFwBw::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

TagVector ModelFwBw::sample(const Sentence& seq) {
  TagVector vec;
  this->gradient(seq, &vec, false);
  return vec;
}
