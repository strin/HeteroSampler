#include "feature.h"
#include "boost/lexical_cast.hpp"

using namespace std;

namespace Tagging {
  StringVector NLPfunc(const string word) {
    StringVector nlp = makeStringVector();
    nlp->push_back(word);
    size_t wordlen = word.length();
    for(size_t k = 1; k <= 3; k++) {
      if(wordlen > k) {
	nlp->push_back("p"+to_string(k)+"-"+word.substr(0, k));
	nlp->push_back("s"+to_string(k)+"-"+word.substr(wordlen-k, k));
      }
    }
    if(std::find_if(word.begin(), word.end(), 
	    [](char c) { return std::isdigit(c); }) != word.end()) {
	nlp->push_back("00-");  // number
    }
    // word signature.
    stringstream sig0;
    string sig1(word);
    string lowercase = word;
    transform(lowercase.begin(), lowercase.end(), lowercase.begin(), ::tolower);
    nlp->push_back(lowercase);
    char prev = '0';
    bool capitalized = true;
    for(size_t i = 0; i < wordlen; i++) {
      if(word[i] <= 'Z' && word[i] >= 'A') {
	if(prev != 'A') 
	  sig0 << "A";
	sig1[i] = 'A';
	prev = 'A';
      }else if(word[i] <= 'z' && word[i] >= 'a') {
	if(prev != 'a')
	  sig0 << "a";
	sig1[i] = 'a';
	prev = 'a';
	capitalized = false;
      }else{
	sig1[i] = 'x';
	prev = 'x';
	capitalized = false;
      }
    }
    nlp->push_back("SG-"+sig0.str());
    nlp->push_back("sg-"+sig1);
    if(capitalized) 
      nlp->push_back("CAP-");
    // word_feat[word] = nlp;
    return nlp;
  }

  void extractUnigramFeature(const Tag& tag, int pos, int breadth, int depth, FeaturePointer output) {
    const vector<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    if(isinstance<ptr<TokenLiteral> >(sen[pos])) {
      // word-tag potential.
      list<pair<string, int>> vp;
      for(int l = max(0, pos - breadth); l <= min(pos + breadth, seqlen-1); l++) {
	ptr<TokenLiteral> token = dynamic_pointer_cast<TokenLiteral>(sen[l]);
	StringVector nlp = tag.corpus->getWordFeat(token->word);
	string lpos = boost::lexical_cast<string>(l-pos);
	for(const string& token : *nlp) {
	  string ss = "w-";
	  ss += lpos;
	  ss += "-";
	  ss += token;
	  ss += "-";
	  ss += tag.getTag(pos);
	  //(*output)[ss] = 1;
	  insertFeature(output, ss);
	}
	for(int d = 1; d <= depth; d++) {
	  if(d >= token->depth()) continue;
	  string ss = "t";
	  if(d > 1) ss += boost::lexical_cast<string>(d);
	  ss += "-";
	  ss += lpos;
	  ss += "-";
	  ss += token->token[d];
	  ss +="-";
	  ss += tag.getTag(pos);
	  insertFeature(output, ss);
	  //(*output)[ss] = 1;
	}
      }
    }
  }

  void extractBigramFeature(const Tag& tag, int pos, FeaturePointer output) {
    const vector<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    assert(pos >= 0 && pos < seqlen);
    string ss = "p-";
    ss += tag.getTag(pos-1);
    ss += "-";
    ss += tag.getTag(pos);
    /* ss << "p-" << tag.getTag(pos-1) << "-" 
	  << tag.getTag(pos);*/
    // (*output)[ss.str()] = 1;
    insertFeature(output, ss); 
  }

  void extractXgramFeature(const Tag& tag, int pos, int factor, FeaturePointer output) {
    const vector<TokenPtr>& sen = tag.seq->seq;
    int seqlen = tag.size();
    assert(pos >= 0 && pos < seqlen);
    assert(pos >= factor-1);
    assert(factor >= 1);
    string ss = "p";
    ss += boost::lexical_cast<string>(factor);
    for(int f = 1; f <= factor; f++) {
      ss += "-";
      ss += tag.getTag(pos-f+1);
    }
    insertFeature(output, ss);
  }
}
