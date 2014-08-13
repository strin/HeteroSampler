#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "utils.h"

#ifndef POS_SENTENCE_H
#define POS_SENTENCE_H

struct Token {
public:
  // old convention, kept for compatibility.
  std::string word,  // word. use lower case, B-S means beginning of sentence. 
	      pos, pos2, 
	      ner;   // name entity.
  // new, more general convention.
  std::vector<std::string> token; // input.
  std::string tag;                // output.
  size_t depth() const {return token.size(); }    // return input depth.
  bool is_doc_start;
  Token();
  Token(const std::string& line);
  void parseline(const std::string& line);
};

struct Corpus;

struct Sentence {
public:
  Sentence(const Corpus* const corpus);
  Sentence(const Corpus* const corpus, const std::vector<std::string>& lines);
  std::vector<Token> seq;
  std::vector<int> tag;
  const Corpus* corpus;
  void parselines(const std::vector<std::string>& lines);
  std::string str() const;
  size_t size() const {return this->seq.size(); }
};

typedef std::shared_ptr<std::vector<std::string> > StringVector;
static StringVector makeStringVector() {
  return StringVector(new std::vector<std::string>());
}

struct Corpus {
public:
  enum Mode {MODE_POS, MODE_NER};
  Mode mode;
  Corpus(Mode mode = MODE_POS);
  Corpus(const std::string& filename, Mode mode = MODE_POS);
  std::vector<Sentence> seqs;
  objcokus cokus;
  std::map<std::string, int> tags, tagcounts;
  std::map<std::string, int> dic, dic_counts;
  std::map<std::string, std::vector<int> > word_tag_count;
  size_t total_words, total_tags;
  std::map<int, std::string> invtags;
  std::string invtag(int id) const {return invtags.find(id)->second; }
  void read(const std::string& filename, bool lets_shuffle = true);
  void retag(const Corpus& corpus); // retag using corpus' tag.
  int size() const {return seqs.size(); }
  void computeWordFeat();  // compute and cache word features.
  StringVector getWordFeat(std::string word) const;

private:
  std::unordered_map<std::string, StringVector> word_feat;
  bool is_word_feat_computed;
};

#endif
