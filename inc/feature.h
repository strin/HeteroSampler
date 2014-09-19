#ifndef POS_FEAT_EXTRACT_H
#define POS_FEAT_EXTRACT_H

#include "utils.h"
#include "tag.h"
#include "model.h"
#include "tag.h"
#include "corpus.h"
#include "objcokus.h"

namespace Tagging {
  StringVector NLPfunc(const std::string word);
  void extractUnigramFeature(const Tag& tag, int pos, int breadth, int depth, FeaturePointer output);
  void extractBigramFeature(const Tag& tag, int pos, FeaturePointer output);
  // extract X-gram feature, i.e. factor connecting pos-factorL+1:pos.
  void extractXgramFeature(const Tag& tag, int pos, int factorL, FeaturePointer output);

  static auto extractOCR = [] (ptr<Model> model, const GraphicalModel& gm, int pos) {
    // default feature extraction, support literal sequence tagging.
    assert(isinstance<ModelCRFGibbs>(model));
    ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model);
    auto tag = dynamic_cast<const Tag&>(gm);
    size_t windowL = this_model->windowL;
    size_t depthL = this_model->depthL;
    size_t factorL = this_model->factorL;
    assert((isinstance<CorpusOCR<16, 8> >(tag.corpus)));
    const vec<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    FeaturePointer features = makeFeaturePointer();
    // extract unigram.
    for(int i = 0; i < 16; i++) {
      for(int j = 0; j < 8; j++) {
	string code = tag.corpus->invtags[tag.tag[pos]]+"  ";
	code[1] = i+'a';
	code[2] = j+'a';
	/*code += to_string(i);
	code += "-";
	code += to_string(j);
	code += "-";
	code += to_string(cast<TokenOCR<16, 8> >(sen[pos])->get(i, j));*/
	if(cast<TokenOCR<16, 8> >(sen[pos])->get(i, j) == 1) 
	  code[0] = code[0] - 'a' + 'A';
	insertFeature(features, code, 1); 
      }
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
  };
  static auto extractOCRAll = [] (ptr<Model> model, const GraphicalModel& gm) {
    // default feature extraction, support literal sequence tagging.
    assert(isinstance<ModelCRFGibbs>(model));
    ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model);
    auto tag = dynamic_cast<const Tag&>(gm);
    size_t windowL = this_model->windowL;
    size_t depthL = this_model->depthL;
    size_t factorL = this_model->factorL;
    assert((isinstance<CorpusOCR<16, 8> >(tag.corpus)));
    const vec<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    FeaturePointer features = makeFeaturePointer();
    for(int pos = 0; pos < seqlen; pos++) {
      // extract unigram.
      for(int i = 0; i < 16; i++) {
	for(int j = 0; j < 8; j++) {
	  string code = tag.corpus->invtags[tag.tag[pos]]+"  ";
	  code[1] = i+'a';
	  code[2] = j+'a';
	  if(cast<TokenOCR<16, 8> >(sen[pos])->get(i, j) == 1) 
	    code[0] = code[0] - 'a' + 'A';
	  insertFeature(features, code, 1); 
	}
      }
      // extract higher-order grams.
      for(int factor = 1; factor <= factorL; factor++) {
	 if(pos-factor+1 >= 0) {
	   extractXgramFeature(tag, pos, factor, features);
	 }
      }
    }
    return features;
  };
}
#endif
