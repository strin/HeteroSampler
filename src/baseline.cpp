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

namespace Tagging { 
  ////////// Simple Model (Independent Logit) ////////////
  ModelSimple::ModelSimple(ptr<Corpus> corpus, const po::variables_map& vm) 
  :Model(corpus, vm), windowL(vm["windowL"].as<int>()),
   depthL(vm["depthL"].as<int>()) {
    xmllog.begin("windowL"); xmllog << windowL << endl; xmllog.end();
  }

  void ModelSimple::sample(Tag& tag, int time, bool argmax) {
    for(int t = 0; t < time; t++) {
      for(size_t i = 0; i < tag.size(); i++) {
	auto featExtract = [&] (const Tag& tag) -> FeaturePointer {
			      return this->extractFeatures(tag, i); 
			    };
	tag.proposeGibbs(i, featExtract, false, false, argmax);
      }
    }
  }

  TagVector ModelSimple::sample(const Sentence& seq, bool argmax) {
    assert(argmax == true);
    TagVector vec;
    gradient(seq, &vec, false);
    return vec;
  }

  FeaturePointer ModelSimple::extractFeatures(const Tag& tag, int pos) {
    FeaturePointer features = makeFeaturePointer();
    const vector<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    // extract word features only.
    for(int l = max(0, pos - windowL); l <= min(pos + windowL, seqlen-1); l++) {
      StringVector nlp = NLPfunc(cast<TokenLiteral>(sen[l])->word);
      for(const string& token : *nlp) {
	stringstream ss;
	ss << "simple-w-" << to_string(l-pos) 
	   << "-" << token << "-" << corpus->invtag(tag.tag[pos]);
	insertFeature(features, ss.str());
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

  void ModelSimple::logArgs() {
    Model::logArgs();
    xmllog.begin("windowL"); xmllog << windowL << endl; xmllog.end();
    xmllog.begin("depthL"); xmllog << depthL << endl; xmllog.end();
  }

  //////// Model CRF Gibbs ///////////////////////////////
  ModelCRFGibbs::ModelCRFGibbs(ptr<Corpus> corpus, const po::variables_map& vm)
   :ModelSimple(corpus, vm), factorL(vm["factorL"].as<int>()), 
    extractFeatures([] (ptr<Model> model, const Tag& tag, int pos) {
      // default feature extraction, support literal sequence tagging.
      assert(isinstance<ModelCRFGibbs>(model));
      ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model);
      size_t windowL = this_model->windowL;
      size_t depthL = this_model->depthL;
      size_t factorL = this_model->factorL;
      assert(isinstance<CorpusLiteral>(tag.corpus));
      const vector<TokenPtr>& sen = tag.seq->seq;
      int seqlen = tag.size();
      // extract word features. 
      FeaturePointer features = makeFeaturePointer();
      extractUnigramFeature(tag, pos, windowL, depthL, features);
      // extract word bigram. (only for compatiblility, should be deleted).
      if(pos >= 1) {
        extractBigramFeature(tag, pos, features);
      }
      if(pos < seqlen-1) {
        extractBigramFeature(tag, pos+1, features);
      }
      // extract higher-order grams.
      for(int factor = 1; factor <= factorL; factor++) {
       for(int p = pos; p < pos+factor; p++) {
         if(p-factor+1 >= 0 && p < seqlen) {
           extractXgramFeature(tag, p, factor, features);
         }
       }
      }
      return features;
    }),
    extractFeatAll([] (ptr<Model> model, const Tag& tag) {
      // default feature extraction, support literal sequence tagging.
      assert(isinstance<ModelCRFGibbs>(model));
      ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model);
      size_t windowL = this_model->windowL;
      size_t depthL = this_model->depthL;
      size_t factorL = this_model->factorL;
      assert(isinstance<CorpusLiteral>(tag.corpus));
      const vector<TokenPtr>& sen = tag.seq->seq;
      int seqlen = tag.size();
      // extract word features. 
      FeaturePointer features = makeFeaturePointer();
      for(size_t pos = 0; pos < seqlen; pos++) {
        // extract ILR features.
        extractUnigramFeature(tag, pos, windowL, depthL, features);
        // extract higher-order grams.
        for(int factor = 1; factor <= factorL; factor++) {
           if(pos-factor+1 >= 0) {
             extractXgramFeature(tag, pos, factor, features);
           }
         }
      }
      return features;
   }) {
    if(isinstance<CorpusLiteral>(corpus))
      cast<CorpusLiteral>(corpus)->computeWordFeat();
   }

  void ModelCRFGibbs::sample(Tag& tag, int time, bool argmax) {
    for(int t = 0; t < time; t++) {
      this->sampleOneSweep(tag, argmax);  
    }
  }

  ParamPointer ModelCRFGibbs::sampleOne(Tag& tag, int choice) {
    if(choice >= tag.size())
      throw "kernel choice invalid (>= tag size)";
    return tag.proposeGibbs(choice, [&] (const Tag& tag) -> FeaturePointer {
			  return this->extractFeatures(shared_from_this(), tag, choice); 
			}, true, true);
  }

  TagVector ModelCRFGibbs::sample(const Sentence& seq, bool argmax) { 
    TagVector vec;
    TagPtr tag = makeTagPtr(&seq, corpus, &rngs[0], param);
    this->sample(*tag, T, argmax);
    vec.push_back(tag);
    return vec;
  }

  double ModelCRFGibbs::score(const Tag& tag) {
    FeaturePointer feat = this->extractFeaturesAll(tag);
    return Tagging::score(this->param, feat);
  }

  FeaturePointer ModelCRFGibbs::extractFeaturesAll(const Tag& tag) {
    extractFeatAll(shared_from_this(), tag);
  }

  void ModelCRFGibbs::sampleOneSweep(Tag& tag, bool argmax) {
    for(int i = 0; i < tag.tag.size(); i++) { 
      tag.proposeGibbs(i, [&] (const Tag& tag) -> FeaturePointer {
			    return this->extractFeatures(shared_from_this(), tag, i); 
			  }, false, false, argmax);
    }
  }

  void ModelCRFGibbs::logArgs() {
    ModelSimple::logArgs();
    xmllog.begin("factorL"); xmllog << factorL << endl; xmllog.end();
  }

  ParamPointer ModelCRFGibbs::gradient(const Sentence& seq) {
    return this->gradient(seq, nullptr, true);
  }

  ParamPointer ModelCRFGibbs::gradient(const Sentence& seq, TagVector* samples, bool update_grad) {
    Tag tag(&seq, corpus, &rngs[0], param);
    Tag truth(seq, corpus, &rngs[0], param);
    ParamPointer gradient = makeParamPointer();
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
      FeaturePointer feat = this->extractFeaturesAll(truth);
      mapDivide<double>(*gradient, -(double)(T-B));
      mapUpdate<double, double>(*gradient, *feat);
    }
    return gradient;
  }
}

