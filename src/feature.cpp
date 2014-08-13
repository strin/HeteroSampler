#include "feature.h"

using namespace std;

StringVector NLPfunc(const string word) {
  StringVector nlp = makeStringVector();
  nlp->push_back(word);
  /* unordered_map<std::string, StringVector>::iterator it = word_feat.find(word);
  if(it != word_feat.end())
    return it->second; */
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
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  // word-tag potential.
  for(int l = max(0, pos - breadth); l <= min(pos + breadth, seqlen-1); l++) {
    StringVector nlp = tag.corpus->getWordFeat(sen[l].word);
    for(const string& token : *nlp) {
      stringstream ss;
      ss << "w-" << to_string(l-pos) 
	 << "-" << token << "-" << tag.getTag(pos);
      if(token == sen[l].word)
	(*output)[ss.str()] = 1;
      else
	(*output)[ss.str()] = 1;
    }
    for(int d = 1; d <= depth; d++) {
      if(d >= sen[l].depth()) continue;
      stringstream ss;
      string ds = "";
      if(d > 1) ds = to_string(d);
      ss << "t" << ds << "-" << l-pos 
	<< "-" << sen[l].token[d] << "-" << tag.getTag(pos);
      (*output)[ss.str()] = 1;
    }
  }
}

void extractBigramFeature(const Tag& tag, int pos, FeaturePointer output) {
  const vector<Token>& sen = tag.seq->seq;
  int seqlen = tag.size();
  stringstream ss;
  ss << "p-" << tag.getTag(pos-1) << "-" 
  	<< tag.getTag(pos);
  (*output)[ss.str()] = 1;
}
