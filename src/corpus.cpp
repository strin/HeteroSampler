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
  }

  TokenLiteral::TokenLiteral() : is_doc_start(false) {}
  TokenLiteral::TokenLiteral(const string& line) : Token(line), is_doc_start(false) {
    this->parseline(line);
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

  Sentence::Sentence(const Corpus* corpus) : corpus(corpus) {}
  Sentence::Sentence(const Corpus* corpus, const vector<string>& lines)
  :corpus(corpus) {
  }

  SentenceLiteral::SentenceLiteral(const Corpus* corpus) : Sentence(corpus) {}
  SentenceLiteral::SentenceLiteral(const Corpus* corpus, const std::vector<std::string>& lines) : Sentence(corpus, lines) {
    this->parselines(lines);
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

  /////////// Corpus ///////////////////////////////////////////
  Corpus::Corpus() { 
  }

  void Corpus::retag(ptr<Corpus> corpus) {
    this->tags = corpus->tags;
    this->invtags = corpus->invtags;
    for(SentencePtr sen : seqs) {
      for(int i = 0; i < sen->size(); i++) {
	sen->tag[i] = this->tags[sen->seq[i]->tag];
      }
    }
  }



  //////////// CorpusLiteral /////////////////////////////////
  CorpusLiteral::CorpusLiteral() 
    :is_word_feat_computed(false) {
    word_feat.clear();
  }

  void CorpusLiteral::computeWordFeat() {
    for(const pair<string, int>& p : this->dic) {
      word_feat[p.first] = NLPfunc(p.first);    
    }
    is_word_feat_computed = true;
  }

  StringVector CorpusLiteral::getWordFeat(string word) const {
    if(is_word_feat_computed and word_feat.find(word) != word_feat.end())
      return word_feat.find(word)->second;
    return NLPfunc(word);
  }

  void CorpusLiteral::read(const std::string& filename, bool lets_shuffle) {
    std::ifstream file;
    file.open(filename);
    if(!file.is_open()) 
      throw "failed to read corpus file. ";
    std::string line;
    std::vector<std::string> lines;
    int tagid = 0;
    total_words = 0;
    this->seqs.clear();
    this->tags.clear();
    this->invtags.clear();
    while(!file.eof()) {
      getline(file, line);
      if(line == "") {
	SentencePtr sen = SentencePtr(new SentenceLiteral(this, lines));
	lines.clear();
	if(sen->seq.size() > 0) {
	  seqs.push_back(sen);
	}else continue;
	for(const TokenPtr token : seqs.back()->seq) {
	  string tg =  token->tag;
	  if(not tags.contains(tg)) {
	    tags[tg] =  tagid;
	    invtags.push_back(tg);
	    tagid++;
	    tagcounts[tg] = 0;
	  }
	  tagcounts[tg]++;
	  ptr<TokenLiteral> token_literal = cast<TokenLiteral>(token);
	  if(dic_counts.find(token_literal->word) == dic_counts.end()) { 
	    total_words++;
	    dic_counts[token_literal->word] = 0;
	  }
	  dic_counts[token_literal->word]++;
	}
      }else
	lines.push_back(line);
    }
    // convert raw tag into integer tag.
    word_tag_count.clear();
    aveT = 0;
    invdic.clear();
    dic.clear();
    total_sig = 0;
    auto mapNewToken = [&] (string token) {
	dic[token] = total_sig;
	invdic.push_back(token);
	total_sig++;
    };
    for(SentencePtr seq : seqs) {
      aveT += seq->size();
      for(const TokenPtr token : seq->seq) {
	string tg;
	tg = token->tag;
	int itg = tags[tg];
	seq->tag.push_back(itg); 
	ptr<TokenLiteral> token_literal = cast<TokenLiteral>(token);
	// map tokens to int.
	string word = token_literal->word;
	if(not dic.contains("w-"+word))  
	  mapNewToken("w-"+word);
	token_literal->itoken.push_back(dic["w-"+word]);
	for(size_t t = 1; t < token_literal->token.size(); t++) {
	  string tk = token_literal->token[t];
	  if(not dic.contains("t"+to_string(t)+"-"+tk))
	    mapNewToken("t"+to_string(t)+"-"+tk);
	  token_literal->itoken.push_back(dic["t"+to_string(t)+"-"+tk]);
	}
	StringVector nlp = NLPfunc(token_literal->word);
	for(auto sig : *nlp) {
	  if(not dic.contains(sig)) 
	    mapNewToken(sig);
	  token_literal->itoken.push_back(dic[sig]);
	}
	if(word_tag_count.find(token_literal->word) == word_tag_count.end()) {
	  word_tag_count[token_literal->word].resize(tagid, 0.0);
	}
	word_tag_count[token_literal->word][itg]++;
      }
    }
    aveT /= (double)seqs.size();
    // shuffle corpus.
    if(lets_shuffle)
      shuffle<SentencePtr>(seqs, cokus);
  }

  void CorpusLiteral::retag(ptr<Corpus> corpus) {
    if(isinstance<CorpusLiteral>(corpus)) {
      ptr<CorpusLiteral> corpus_literal = cast<CorpusLiteral>(corpus);
      this->dic = corpus_literal->dic;
      this->invdic = corpus_literal->invdic;
      this->dic_counts = corpus_literal->dic_counts;
      this->total_sig = corpus_literal->total_sig;
      this->total_words = corpus_literal->total_words;
    }
  }

  tuple<ParamPointer, double> CorpusLiteral::tagEntropySimple() const {
    ParamPointer feat = makeParamPointer();
    const size_t taglen = this->tags.size();
    double logweights[taglen];
    // compute raw entropy.
    for(const pair<string, int>& p : this->dic) {
      auto count = this->word_tag_count.find(p.first);
      if(count == this->word_tag_count.end()) {
	cerr << "tagEntropy: word not found." << endl;
	(*feat)[p.first] = log(taglen);
	continue;
      }
      for(size_t t = 0; t < taglen; t++) {
	if(count->second[t] == 0)
	  logweights[t] = -DBL_MAX;
	else
	  logweights[t] = log(count->second[t]);
      }
      logNormalize(logweights, taglen);
      double entropy = 0.0;
      for(size_t t = 0; t < taglen; t++) {
	entropy -= logweights[t] * exp(logweights[t]);
      }
      (*feat)[p.first] = entropy;
    }
    // compute feature mean.
    double mean_ent = 0.0, count = 0;
    for(const SentencePtr seq : this->seqs) {
      for(size_t i = 0; i < seq->seq.size(); i++) {
	ptr<TokenLiteral> token = dynamic_pointer_cast<TokenLiteral>(seq->seq[i]);
	mean_ent += (*feat)[token->word]; 
	count++;
      }
    }
    mean_ent /= count;
    // substract the mean.
    for(const pair<string, int>& p : this->dic) {
      (*feat)[p.first] -= mean_ent; 
    }
    return make_tuple(feat, mean_ent);
  }

  tuple<ParamPointer, double> CorpusLiteral::wordFrequencies() const {
    assert(this->seqs.size() > 0 and isinstance<SentenceLiteral>(this->seqs[0]));
    ParamPointer feat = makeParamPointer();
    // compute raw feature.
    for(const pair<string, int>& p : this->dic_counts) {
      (*feat)[p.first] = log(this->total_words)-log(p.second);
    }
    // compute feature mean.
    double mean_fre = 0, count = 0;
    for(const SentencePtr seq : this->seqs) {
      for(size_t i = 0; i < seq->seq.size(); i++) {
	ptr<TokenLiteral> token = dynamic_pointer_cast<TokenLiteral>(seq->seq[i]);
	mean_fre += (*feat)[token->word];
	count++;
      }
    }
    mean_fre /= count;
    // substract the mean.
    for(const pair<string, int>& p : this->dic) {
      (*feat)[p.first] -= mean_fre;
    }
    return make_tuple(feat, mean_fre);
  }

  pair<Vector2d, vector<double> > CorpusLiteral::tagBigram() const {
    size_t taglen = this->tags.size();
    Vector2d mat = makeVector2d(taglen, taglen, 1.0);
    vector<double> vec(taglen, 1.0);
    for(const SentencePtr seq : this->seqs) {
      vec[seq->tag[0]]++;
      for(size_t t = 1; t < seq->size(); t++) {
	mat[seq->tag[t-1]][seq->tag[t]]++; 
      }
    }
    for(size_t i = 0; i < taglen; i++) {
      vec[i] = log(vec[i])-log(taglen+this->seqs.size());
      double sum_i = 0.0;
      for(size_t j = 0; j < taglen; j++) {
	sum_i += mat[i][j];
      }
      for(size_t j = 0; j < taglen; j++) {
	mat[i][j] = log(mat[i][j])-log(sum_i);
      }
    }
    return make_pair(mat, vec);
  }
  //////////// CorpusOCR /////////////////////////////////
  template<size_t height, size_t width>
  CorpusOCR<height, width>::CorpusOCR() {
  }

  template<size_t height, size_t width>
  void CorpusOCR<height, width>::read(const std::string& filename, bool lets_shuffle) {
    std::ifstream file;
    file.open(filename);
    if(!file.is_open()) 
      throw "failed to read corpus file. ";
    std::string line;
    std::vector<std::string> lines;
    int tagid = 0;
    vec<int> tokenid;
    this->seqs.clear();
    this->tags.clear();
    while(!file.eof()) {
      getline(file, line);
      if(line == "") {
	SentencePtr sen = SentencePtr(new SentenceOCR<height, width>(this, lines));
	lines.clear();
	if(sen->seq.size() > 0) {
	  seqs.push_back(sen);
	}else continue;
	for(const TokenPtr token : seqs.back()->seq) {
	  string tg = token->tag;
	  if(not tags.contains(tg)) 
	    tags[tg] =  tagid++;
	}
	continue;
      }else
	lines.push_back(line);
    }
    // encoding tag / tokens.
    invtags.clear();
    aveT = 0;
    for(SentencePtr seq : seqs) {
      aveT += seq->size();
      for(const TokenPtr token : seq->seq) {
	string tg = tg = token->tag;
	int itg = tags[tg];
	seq->tag.push_back(itg); 
	invtags[itg] = tg;
      }
    }
    aveT /= (double)seqs.size();
    // shuffle corpus.
    if(lets_shuffle)
      shuffle<SentencePtr>(seqs, cokus);
  }
}
