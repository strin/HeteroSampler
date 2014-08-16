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

// implement Token.
Token::Token() : is_doc_start(false) {}
Token::Token(const string& line) : is_doc_start(false) {
  this->parseline(line);
}

void Token::parseline(const string& line) {
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

void Sentence::parselines(const vector<string>& lines) {
  this->seq.clear();
  BOOST_FOREACH(const string& line, lines) {
    Token token(line);
    if(!token.is_doc_start)
      this->seq.push_back(token); 
  }
}

string Sentence::str() const {
  stringstream ss;
  for(const Token& token : seq) {
    ss << token.word << "/" << token.tag << "\t";
  }
  return ss.str();
}

// implement Corpus.
Corpus::Corpus() 
:is_word_feat_computed(false) {
  word_feat.clear();
}

Corpus::Corpus(const string& filename)
:is_word_feat_computed(false) {
  this->read(filename);
  word_feat.clear();
}

void Corpus::read(const string& filename, bool lets_shuffle) {
  ifstream file;
  file.open(filename);
  if(!file.is_open()) 
    throw "failed to read corpus file. ";
  string line;
  vector<string> lines;
  int tagid = 0;
  total_words = 0;
  this->seqs.clear();
  this->tags.clear();
  while(!file.eof()) {
    getline(file, line);
    if(line == "") {
      Sentence sen(this, lines);
      lines.clear();
      if(sen.seq.size() > 0) {
	seqs.push_back(sen);
      }else continue;
      BOOST_FOREACH(const Token& token, seqs.back().seq) {
	string tg;
	tg = token.tag;
	if(tags.find(tg) == tags.end()) {
	  tags[tg] =  tagid++;
	  tagcounts[tg] = 0;
	}
	tagcounts[tg]++;
	dic[token.word] = true;
	if(dic_counts.find(token.word) == dic_counts.end()) 
	  dic_counts[token.word] = 0;
	dic_counts[token.word]++;
	total_words++;
      }
      continue;
    }else
      lines.push_back(line);
  }
  // convert raw tag into integer tag.
  invtags.clear();
  word_tag_count.clear();
  aveT = 0;
  for(Sentence& seq : seqs) {
    aveT += seq.size();
    for(const Token& token : seq.seq) {
      string tg;
      tg = token.tag;
      int itg = tags[tg];
      seq.tag.push_back(itg); 
      invtags[itg] = tg;
      if(word_tag_count.find(token.word) == word_tag_count.end()) {
	word_tag_count[token.word].resize(tagid, 0.0);
      }
      word_tag_count[token.word][itg]++;
    }
  }
  aveT /= (double)seqs.size();
  // shuffle corpus.
  if(lets_shuffle)
    shuffle<Sentence>(seqs, cokus);
}

void Corpus::retag(const Corpus& corpus) {
  this->tags = corpus.tags;
  this->invtags = corpus.invtags;
  for(Sentence& sen : seqs) {
    for(int i = 0; i < sen.size(); i++) {
      sen.tag[i] = this->tags[sen.seq[i].tag];
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

