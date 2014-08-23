#include "corpus.h"
#include "boost/algorithm/string.hpp"
#include "boost/foreach.hpp"
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>
#include "feature.h"

using namespace std;
using namespace boost;

namespace Tagging {
  // implement Token.
  Token::Token() {}
  Token::Token(const string& line) {
    this->parseline(line);
  }

  TokenLiteral::TokenLiteral() : is_doc_start(false) {}
  TokenLiteral::TokenLiteral(const string& line) : Token(line), is_doc_start(false) {
  }

  void TokenLiteral::parseline(const string& line) {
    vector<string> parts;
    split(parts, line, is_any_of(" "));
    // new convention.
    if(parts.size() < 2) throw ("invalid token - "+line).c_str();
    token.clear();
    word = parts[0];
    for(size_t d = 0; d < parts.size()-1; d++) 
      token.push_back(parts[d]);
    tag = parts[parts.size()-1];
    // old convention.
    pos = parts[1];
    if(parts.size() >= 3)
      pos2 = parts[2];
    if(parts.size() >= 4)
      ner = parts[3];
    if(word == "-DOCSTART-") 
      is_doc_start = true;
  }


  // implement Sentence.
  Sentence::Sentence(const Corpus* corpus) : corpus(corpus) {}
  Sentence::Sentence(const Corpus* corpus, const vector<string>& lines)
  :corpus(corpus) {
    this->parselines(lines);
  }

  SentenceLiteral::SentenceLiteral(const Corpus* corpus) : Sentence(corpus) {}
  SentenceLiteral::SentenceLiteral(const Corpus* corpus, const std::vector<std::string>& lines) : Sentence(corpus, lines) {
  }

  void SentenceLiteral::parselines(const vector<string>& lines) {
    this->seq.clear();
    BOOST_FOREACH(const string& line, lines) {
      ptr<TokenLiteral> token = ptr<TokenLiteral>(new TokenLiteral(line));
      if(!token->is_doc_start)
	this->seq.push_back(token); 
    }
  }

  string SentenceLiteral::str() const {
    stringstream ss;
    for(const TokenPtr tk : seq) {
      ptr<TokenLiteral> token = dynamic_pointer_cast<TokenLiteral>(tk);
      ss << token->word << "/" << token->tag << "\t";
    }
    return ss.str();
  }

  // implement Corpus.
  Corpus::Corpus() 
  :is_word_feat_computed(false) {
    word_feat.clear();
  }


  void Corpus::retag(const Corpus& corpus) {
    this->tags = corpus.tags;
    this->invtags = corpus.invtags;
    for(SentencePtr sen : seqs) {
      for(int i = 0; i < sen->size(); i++) {
	sen->tag[i] = this->tags[sen->seq[i]->tag];
      }
    }
  }

  void Corpus::computeWordFeat() {
    for(const pair<string, int>& p : this->dic) {
      word_feat[p.first] = NLPfunc(p.first);    
    }
    is_word_feat_computed = true;
  }

  StringVector Corpus::getWordFeat(string word) const {
    if(is_word_feat_computed and word_feat.find(word) != word_feat.end())
      return word_feat.find(word)->second;
    return NLPfunc(word);
  }
}
