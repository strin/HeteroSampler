/* implementation of baseline sequence tagging models, including 
 * > independent logistic regression.
 * > CRF with Gibbs sampling. 
 */
#include "model.h"

using namespace std;

//////// Model CRF Gibbs ///////////////////////////////
TagVector ModelCRFGibbs::sample(const Sentence& seq) { 
  TagVector vec;
  gradient(seq, &vec, false); 
  return vec;
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

ParamPointer ModelCRFGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  FeaturePointer feat = tag.extractFeatures(seq.tag);
  ParamPointer gradient(new map<string, double>()); 
  for(int t = 0; t < T; t++) {
    for(int i = 0; i < seq.tag.size(); i++) 
      tag.proposeGibbs(i);
    if(t < B) continue;
    if(update_grad)
      mapUpdate<double, double>(*gradient, *tag.extractFeatures(tag.tag));
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

////////// Simple Model (Independent Logit) ////////////
TagVector ModelSimple::sample(const Sentence& seq) {
  TagVector vec;
  gradient(seq, &vec, false);
  return vec;
}

ParamPointer ModelSimple::gradient(const Sentence& seq) {
  return this->gradient(seq, nullptr, true);
}

ParamPointer ModelSimple::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
  Tag tag(&seq, corpus, &rngs[0], param);
  ParamPointer gradient(new map<string, double>());
  for(size_t i = 0; i < tag.size(); i++) {
    ParamPointer g = tag.proposeSimple(i, true, false);
    if(update_grad) {
      mapUpdate<double, double>(*gradient, *g);
      mapUpdate<double, double>(*gradient, *tag.extractSimpleFeatures(seq.tag, i));
    }
  }
  if(samples)
    samples->push_back(shared_ptr<Tag>(new Tag(tag)));
  xmllog.begin("truth"); xmllog << seq.str() << endl; xmllog.end();
  xmllog.begin("tag"); xmllog << tag.str() << endl; xmllog.end();
  return gradient;
}

void ModelSimple::run(const Corpus& testCorpus, bool lets_test) {
  Corpus retagged(testCorpus);
  retagged.retag(this->corpus); // use training taggs. 
  xmllog.begin("train_simple");
  int numObservation = 0;
  for(int q = 0; q < Q0; q++) {
    for(const Sentence& seq : corpus.seqs) {
      xmllog.begin("example_"+to_string(numObservation));
      ParamPointer gradient = this->gradient(seq, nullptr, true);
      this->configStepsize(gradient, this->eta);
      this->adagrad(gradient);
      xmllog.end();
      numObservation++;
    }
  }
  xmllog.end();
  if(lets_test) {
    xmllog.begin("test");
    test(retagged);
    xmllog.end();
  }
}

////////// Incremental Gibbs Sampling /////////////////////////
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
  FeaturePointer feat = tag.extractFeatures(seq.tag);
  ParamPointer gradient(new map<string, double>());
  for(int i = 0; i < seq.tag.size(); i++) {
    ParamPointer g = tag.proposeGibbs(i, true);
    mapUpdate<double, double>(*gradient, *g);
    mapUpdate<double, double>(*gradient, *tag.extractFeatures(tag.tag), -1);
    mytag.tag[i] = tag.tag[i];
    tag.tag[i] = seq.tag[i];
    mapUpdate<double, double>(*gradient, *tag.extractFeatures(tag.tag)); 
  }
  if(samples)
    samples->push_back(shared_ptr<Tag>(new Tag(tag)));
  return gradient;
}
