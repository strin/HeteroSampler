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
  pos2 = parts[2];
  ner = parts[3];
  if(word == "-DOCSTART-") 
    is_doc_start = true;
}


// implement Sentence.
Sentence::Sentence() {}
Sentence::Sentence(const vector<string>& lines) {
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
    ss << token.word << "/" << token.pos << "\t";
  }
  return ss.str();
}

// implement Corpus.
Corpus::Corpus() {}

Corpus::Corpus(const string& filename) {
  this->read(filename);
}

void Corpus::read(const string& filename) {
  ifstream file;
  file.open(filename);
  if(!file.is_open()) 
    throw "failed to read corpus file. ";
  string line;
  vector<string> lines;
  int tagid = 0;
  this->seqs.clear();
  this->tags.clear();
  while(!file.eof()) {
    getline(file, line);
    if(line == "") {
      Sentence sen(lines);
      lines.clear();
      if(sen.seq.size() > 0) {
	seqs.push_back(sen);
      }else continue;
      BOOST_FOREACH(const Token& token, seqs.back().seq) {
	if(tags.find(token.pos) == tags.end()) {
	  tags[token.pos] =  tagid++;
	  tagcounts[token.pos] = 0;
	}else
	  tagcounts[token.pos]++;
      }
      continue;
    }else
      lines.push_back(line);
  }
  // convert raw tag into integer tag.
  invtags.clear();
  for(Sentence& seq : seqs) {
    for(const Token& token : seq.seq) {
      seq.tag.push_back(tags[token.pos]); 
      invtags[tags[token.pos]] = token.pos;
    }
  }
}

void Corpus::retag(const Corpus& corpus) {
  this->tags = corpus.tags;
  this->invtags = corpus.invtags;
  for(Sentence& sen : seqs) {
    for(int i = 0; i < sen.size(); i++) {
      sen.tag[i] = this->tags[sen.seq[i].pos];
    }
  }
}
