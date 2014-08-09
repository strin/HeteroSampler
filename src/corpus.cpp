#include "corpus.h"
#include "boost/algorithm/string.hpp"
#include "boost/foreach.hpp"
#include <map>
#include <fstream>
#include <iostream>
#include <sstream>

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
  word = parts[0];
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
    if(corpus->mode == Corpus::MODE_POS)
      ss << token.word << "/" << token.pos << "\t";
    else if(corpus->mode == Corpus::MODE_NER)
      ss << token.word << "/" << token.ner << "\t";
  }
  return ss.str();
}

// implement Corpus.
Corpus::Corpus(Mode mode) : mode(mode) {}

Corpus::Corpus(const string& filename, Mode mode)
:mode(mode) {
  this->read(filename);
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
	if(mode == MODE_POS) tg = token.pos;
	else if(mode == MODE_NER) tg = token.ner;
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
  // count tags.
  if(mode == MODE_POS) total_tags = total_words; 
  else{
    total_tags = 0;
    for(const pair<string, int>& p : tagcounts) {
      if(p.first != "O") total_tags += p.second; 
    }
  }
  // convert raw tag into integer tag.
  invtags.clear();
  word_tag_count.clear();
  for(Sentence& seq : seqs) {
    for(const Token& token : seq.seq) {
      string tg;
      if(mode == MODE_POS) tg = token.pos;
      else if(mode == MODE_NER) tg = token.ner;
      int itg = tags[tg];
      seq.tag.push_back(itg); 
      invtags[itg] = tg;
      if(word_tag_count.find(token.word) == word_tag_count.end()) {
	word_tag_count[token.word].resize(tagid, 0.0);
      }
      word_tag_count[token.word][itg]++;
    }
  }
  // shuffle corpus.
  if(lets_shuffle)
    shuffle<Sentence>(seqs, cokus);
}

void Corpus::retag(const Corpus& corpus) {
  this->tags = corpus.tags;
  this->invtags = corpus.invtags;
  for(Sentence& sen : seqs) {
    for(int i = 0; i < sen.size(); i++) {
      if(mode == MODE_POS)
	sen.tag[i] = this->tags[sen.seq[i].pos];
      else if(mode == MODE_NER)
	sen.tag[i] = this->tags[sen.seq[i].ner];
    }
  }
}
