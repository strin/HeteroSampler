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

namespace HeteroSampler {

  template<size_t height, size_t width>
  TokenOCR<height, width>::TokenOCR() {}

  template<size_t height, size_t width>
  TokenOCR<height, width>::TokenOCR(const string& line) : Token(line) {
    this->parseline(line);
  }

  template<size_t height, size_t width>
  void TokenOCR<height, width>::parseline(const string& line) {
    vector<string> parts;
    split(parts, line, is_any_of("\t"));
    tag = parts[1];
    fold = int(parts[5][0]);
    for(size_t i = 0; i < height; i++) {
      for(size_t j = 0; j < width; j++) {
        img[i][j] = int(parts[6 + i * width + j][0]-'0');
      }
    }
  }

  template<size_t height, size_t width>
  string TokenOCR<height, width>::str() const {
    string str = "";
    for(size_t i = 0; i < height; i++) {
      for(size_t j = 0; j < width; j++) {
        str += '0'+img[i][j];
      }
    }
    return str;
  }

  template<size_t height, size_t width>
  void SentenceOCR<height, width>::parselines(const vector<string>& lines) {
    this->seq.clear();
    for(const string& line : lines) {
      ptr<TokenOCR<height, width> > token = ptr<TokenOCR<height, width> >(new TokenOCR<height, width>(line));
      this->seq.push_back(token);
    }
  }

  template<size_t height, size_t width> 
  string SentenceOCR<height, width>::str() const {
    stringstream ss;
    for(const TokenPtr token : seq) {
      ss << token->str() << " / " << token->tag << "\t";
    }
    return ss.str();
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
    invtags.resize(tagid);
    aveT = 0;
    for(SentencePtr seq : seqs) {
      aveT += seq->size();
      for(const TokenPtr token : seq->seq) {
        string tg = token->tag;
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

  // explicitly init 16 x 8 template.
  template class TokenOCR<16, 8>;
  template class SentenceOCR<16, 8>;
  template class CorpusOCR<16, 8>;
}

