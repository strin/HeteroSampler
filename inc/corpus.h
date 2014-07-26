#ifndef POS_SENTENCE_H
#define POS_SENTENCE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

struct Token {
public:
  std::string word,  // word. use lower case, B-S means beginning of sentence. 
	      pos, pos2, 
	      ner;   // name entity.
  bool is_doc_start = false;
  Token();
  Token(const std::string& line);
  void parseline(const std::string& line);
};

struct Sentence {
public:
  std::vector<Token> seq;
  std::vector<int> tag;
  Sentence();
  Sentence(const std::vector<std::string>& lines);
  void parselines(const std::vector<std::string>& lines);
  std::string str() const;
};

struct Corpus {
public:
  std::vector<Sentence> seqs;
  std::map<std::string, int> tags, tagcounts;
  std::map<int, std::string> invtags;
  Corpus();
  Corpus(const std::string& filename);
  void read(const std::string& filename);
};

#endif
