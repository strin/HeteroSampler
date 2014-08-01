/* implementation of baseline sequence tagging models, including 
 * > independent logistic regression.
 * > CRF with Gibbs sampling. 
 */
#include "model.h"

using namespace std;
using namespace std::placeholders;

////////// Simple Model (Independent Logit) ////////////
ModelSimple::ModelSimple(const Corpus& corpus, int windowL, int T, int B, int Q, double eta)
:Model(corpus, T, B, Q, eta), windowL(windowL) {
  xmllog.begin("windowL"); xmllog << windowL << endl; xmllog.end();
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
	 << "-" << token << "-" << tag.tag[pos];
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
ModelCRFGibbs::ModelCRFGibbs(const Corpus& corpus, int windowL, int T, int B, int Q, double eta)
:ModelSimple(corpus, windowL, T, B, Q, eta) {
}

TagVector ModelCRFGibbs::sample(const Sentence& seq) { 
  TagVector vec;
  gradient(seq, &vec, false); 
  return vec;
}

FeaturePointer ModelCRFGibbs::extractFeatures(const Tag& tag, int pos) {
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  // extract word features. 
  FeaturePointer features = makeFeaturePointer();
  for(int l = max(0, pos - windowL); l <= min(pos + windowL, seqlen-1); l++) {
    StringVector nlp = NLPfunc(sen[l].word);
    for(const string& token : *nlp) {
      stringstream ss;
      ss << "w-" << to_string(l-pos) 
	 << "-" << token << "-" << tag.tag[pos];
      (*features)[ss.str()] = 1;
    }
  }
  // extract bigram features.
  if(pos >= 1) {
    stringstream ss;
    ss << "p-" << tag.tag[pos-1] << "-" << tag.tag[pos];
    (*features)[ss.str()] = 1;
  }
  if(pos < seqlen-1) {
    stringstream ss;
    ss << "p-" << tag.tag[pos] << "-" << tag.tag[pos+1];
    (*features)[ss.str()] = 1;
  }
  return features;
}

FeaturePointer ModelCRFGibbs::extractFeatures(const Tag& tag) {
  FeaturePointer features = makeFeaturePointer();
  size_t seqlen = tag.size();
  for(size_t t = 0; t < seqlen; t++) {
    FeaturePointer this_feat = extractFeatures(tag, t);
    mapCopy(*features, *this_feat);
  }
  return features;
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag truth(seq, corpus, &rngs[0], param);
  FeaturePointer feat = this->extractFeatures(truth);
  ParamPointer gradient = makeFeaturePointer();
  for(int t = 0; t < T; t++) {
    for(int i = 0; i < seq.tag.size(); i++) 
      tag.proposeGibbs(i, [&] (const Tag& tag) -> FeaturePointer {
			    return this->extractFeatures(tag, i); 
			  });
    if(t < B) continue;
    if(update_grad)
      mapUpdate<double, double>(*gradient, *this->extractFeatures(tag));
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
ModelIncrGibbs::ModelIncrGibbs(const Corpus& corpus, int windowL, int T, int B, int Q, double eta)
:ModelCRFGibbs(corpus, windowL, T, B, Q, eta) {
}

TagVector ModelIncrGibbs::sample(const Sentence& seq) {
  TagVector samples;
  gradient(seq, &samples, false);
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << samples.back()->str() << endl; xmllog.end();
  return samples;
}

ParamPointer ModelIncrGibbs::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, false);
}

ParamPointer ModelIncrGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag mytag(tag);
  Tag truth(seq, corpus, &rngs[0], param);
  FeaturePointer feat = this->extractFeatures(truth);
  ParamPointer gradient = makeFeaturePointer();
  for(int i = 0; i < seq.tag.size(); i++) {
    ParamPointer g = tag.proposeGibbs(i, [&] (const Tag& tag) -> FeaturePointer {
				      return this->extractFeatures(tag, i);  
				    }, true, false);
    mapUpdate<double, double>(*gradient, *g);
    mytag.tag[i] = tag.tag[i];
    tag.tag[i] = seq.tag[i];
    mapUpdate<double, double>(*gradient, *this->extractFeatures(tag, i)); 
  }
  if(samples)
    samples->push_back(shared_ptr<Tag>(new Tag(mytag)));
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << mytag.str() << endl; xmllog.end();
  return gradient;
}

///////// Forward-Backward Algorithm ////////////////////////////
ModelFwBw::ModelFwBw(const Corpus& corpus, int windowL, int T, int B, int Q, double eta) 
:ModelCRFGibbs(corpus, windowL, T, B, Q, eta) { 
}

ParamPointer ModelFwBw::gradient(const Sentence& seq, TagVector* vec, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  Tag truth(seq, corpus, &rngs[0], param);
  size_t seqlen = tag.size(), taglen = corpus.tags.size();
  // dp[i][j] is the multiplicative factor for the marginal 
  // of S_{i:end} starting with character j. (use log-space)
  double dp[seqlen][taglen];
  // forward.
  for(size_t c = 0; c < taglen; c++) dp[0][c] = 0.0;
  for(int i = 1; i < seqlen; i++) {
    for(size_t c = 0; c < taglen; c++) {
      dp[i][c] = -DBL_MAX;
      for(size_t s = 0; s < taglen; s++) {
	double uni = 0.0;
	for(int l = max(0, i-1-windowL); l <= min(i-1+windowL, (int)seqlen-1); l++) { 
	  StringVector nlp = NLPfunc(seq.seq[l].word);
	  for(const string& token : *nlp) {
	    stringstream ss;
	    ss << "w-" << to_string(l-i+1) 
		<< "-" << token << "-" << tag.tag[i-1];
	    uni += mapGet(*param, ss.str());
	  }
	}
	double bi = mapGet(*param, "p-"+to_string(s)+"-"+to_string(c));
	dp[i][c] = logAdd(dp[i][c], dp[i-1][c] + uni + bi);
      }
    }
  }
  // backward. 
  double sc[taglen];
  for(int i = seqlen-1; i >= 0; i--) {
    for(size_t c = 0; c < taglen; c++) {
      sc[c] = 0.0;
      for(int l = max(0, i-windowL); l <= min(i+windowL, (int)seqlen-1); l++) { 
	StringVector nlp = NLPfunc(seq.seq[l].word);
	for(const string& token : *nlp) {
	  stringstream ss;
	  ss << "w-" << to_string(l-i) 
	      << "-" << token << "-" << tag.tag[i];
	  sc[c] += mapGet(*param, ss.str());
	}
      }
      if(i < seqlen-1) 
	sc[c] += mapGet(*param, "p-"+to_string(c)+"-"+to_string(tag.tag[i+1]));
      sc[c] += dp[i][c];
    }
    logNormalize(sc, taglen);
    tag.tag[i] = rngs[0].sampleCategorical(sc, taglen);
  }
  if(vec) 
    vec->push_back(shared_ptr<Tag>(new Tag(tag)));
  ParamPointer gradient = makeParamPointer();
  mapUpdate(*gradient, *this->extractFeatures(tag), -1);
  mapUpdate(*gradient, *this->extractFeatures(truth));
  xmllog.begin("truth"); xmllog << truth.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
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
