#pragma once
#include "corpus.h"
#include "model.h"
#include "boost/algorithm/string.hpp"
#include <boost/lexical_cast.hpp>

namespace Tagging {

struct TokenIsing : public Token {
public:
  TokenIsing(int val, int target)
  : val(val) {
    this->tag = std::to_string(target);
  };

  int val;

  virtual void 
  parseline(const std::string& line) {};

  virtual string 
  str() const {
    return boost::lexical_cast<string>(val);
  };
};
  


struct ImageIsing : public Sentence {
public:
  ImageIsing(const Corpus* corpus) : Sentence(corpus) {};
  ImageIsing(const Corpus* corpus, const vec<string>& lines,
              const vec<string>& lines_gt)
  : Sentence(corpus) {
    this->parselines(lines, lines_gt);
  }

  virtual void parselines(const vec<string>& lines) {}
  
  virtual void
  parselines(const vec<string>& lines, const vec<string>& lines_gt);

  virtual string str() const;

  struct Pt {
  public:
    int h, w;
    Pt() {}
    Pt(int h, int w): h(h), w(w) {}
  };

  int H, W;
  vec2d<int> img, img_gt;

  Pt posToPt(int pos) const {
    Pt pt; 
    pt.w = pos / H;
    pt.h = pos % H;
    return pt;
  }
  int ptToPos(const Pt& pt) const {
    return pt.w * H + pt.h;
  }
};

struct CorpusIsing : public Corpus {
public:
  CorpusIsing() {}

  void read(const std::string& filename, bool lets_shuffle = true);
};
  
void ImageIsing::parselines(const vec<string>& lines,
                       const vec<string>& lines_gt) {
  auto linesToImg = [&] (const vec<string>& lines, 
                         vec2d<int>& img) {
    this->H = lines.size();
    img.resize(H);
    this->W = 0;
    vec<string> parts;
    int h = 0;
    for(const string& line : lines) {
      boost::split(parts, line, boost::is_any_of(" "));
      if(parts[parts.size()-1] == "")
        parts.resize(parts.size()-1);
      if(W == 0) {
        W = parts.size();
      }else{
        assert(W == parts.size());
      }
      img[h].resize(W);
      for(int w = 0; w < W; w++) {
        img[h][w] = boost::lexical_cast<int>(parts[w]);
      }
      h++;
    }
  };
  linesToImg(lines, img);
  linesToImg(lines_gt, img_gt);

  for(int w = 0; w < W; w++) { // fortran style.
    for(int h = 0; h < H; h++) {
      seq.push_back(std::make_shared<TokenIsing>(img[h][w], img_gt[h][w]));
    }
  }
}
  
string ImageIsing::str() const {
  string ret = "";
  ret += std::to_string(H) + " " + std::to_string(W) + "\n";
  for(const TokenPtr tk : seq) {
    ptr<TokenIsing> token = std::dynamic_pointer_cast<TokenIsing>(tk);
    ret += std::to_string(token->val) +  " / " + token->tag + "\t";
  }
  return ret;
}

void CorpusIsing::read(const std::string& filename, bool lets_shuffle) {
  std::ifstream file;
  file.open(filename);
  if(!file.is_open()) 
    throw "failed to read corpus file. ";
  std::string line;
  std::vector<std::string> lines, lines_basic;
  int tagid = 0;
  vec<int> tokenid;
  this->seqs.clear();
  this->tags.clear();
  while(!file.eof()) {
    getline(file, line);
    if(line == "" and lines_basic.size() == 0) {
      lines_basic = lines;
      lines.clear();
    }else if(line == "" and lines_basic.size() > 0) {
      SentencePtr sen = SentencePtr(new ImageIsing(this, lines_basic, lines));
      lines.clear();
      lines_basic.clear();
      if(sen->seq.size() > 0) {
        seqs.push_back(sen);
      }else continue;
      for(const TokenPtr token : seqs.back()->seq) {
        string tg = token->tag;
        if(not tags.contains(tg))  {
          tags[tg] =  tagid++;
          invtags.push_back(tg);
        }
        seqs.back()->tag.push_back(tags[tg]);
      }
      continue;
    }else
      lines.push_back(line);
  }
}
  
static auto extractIsingUnigram = 
[&] (FeaturePointer features, const ImageIsing* image, const Tag& tag, int pos) {
  insertFeature(features, "u-"+image->seq[pos]->str()+"-"+tag.getTag(pos));
};

static auto extractIsingBigram = 
[&] (FeaturePointer features, const ImageIsing* image, const Tag& tag, int pos1, int pos2) {
  const string token1 = tag.getTag(pos1);
  const string token2 = tag.getTag(pos2);
  insertFeature(features, "w-"+token1+"-"+token2);
};

// feature extraction at *pos* for ising-like model.
static auto extractIsing = [] (ptr<Model> model, const Tag& tag, int pos) {
  ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model); 
  auto image = dynamic_cast<const ImageIsing*>(tag.seq);
  assert(image != NULL);
  FeaturePointer features = makeFeaturePointer();
  extractIsingUnigram(features, image, tag, pos);
  const int H = image->H, W = image->W;
  ImageIsing::Pt pt = image->posToPt(pos);
  const int shift[4][2] = {{1, 0}, {0, 1}, {-1, 0}, {0, -1}};
  auto extractByShift = [&] (int shiftX, int shiftY) {
    ImageIsing::Pt pt2(pt);
    pt2.h += shiftY;
    pt2.w += shiftX;
    if(pt2.h < H and pt2.h >= 0 and pt2.w < W and pt2.w >= 0) {
      int pos2 = image->ptToPos(pt2);
      extractIsingBigram(features, image, tag, pos, pos2); 
    }
  };
  for(int i = 0; i < 4; i++) 
    extractByShift(shift[i][0], shift[i][1]);
  return features;
};

// extract all features for ising-like model.
static auto extractIsingAll = [] (ptr<Model> model, const Tag& tag) {
  ptr<ModelCRFGibbs> this_model = cast<ModelCRFGibbs>(model); 
  auto image = dynamic_cast<const ImageIsing*>(tag.seq);
  assert(image != NULL);
  FeaturePointer features = makeFeaturePointer();
  for(int w = 0; w < image->W; w++) {
    for(int h = 0; h < image->H; h++) {
      extractIsingUnigram(features, image, tag, image->ptToPos(ImageIsing::Pt(h, w)));
      if(w < image->W-1) {
        extractIsingBigram(features, image, tag, 
                  image->ptToPos(ImageIsing::Pt(h, w)),
                  image->ptToPos(ImageIsing::Pt(h, w+1)));
      }
      if(h < image->H-1) {
        extractIsingBigram(features, image, tag, 
                  image->ptToPos(ImageIsing::Pt(h, w)),
                  image->ptToPos(ImageIsing::Pt(h+1, w)));
      }
//      if(h < image->H-1 and w < image->W-1) {
//        extractIsingBigram(features, image, tag, 
//                  image->ptToPos(ImageIsing::Pt(h, w)),
//                  image->ptToPos(ImageIsing::Pt(h+1, w+1)));
//      }
    }
  }
  return features;
};

} // namespace Tagging.
