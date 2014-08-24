package space;

import java.io.*;
import java.util.*;
import java.util.regex.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

interface Performance {
  public void add(int[] truey, int[] predy);
  public double getQuality();
  public void log();
}

class TaggingPerformance implements Performance {
  Indexer<String> tagIndexer;
  int T; // Number of tags (B-PER, I-PER, B-LOC, I-LOC, O, ...)
  double[][] tagCounts; // Confusion matrix

  public TaggingPerformance(Indexer<String> tagIndexer) {
    this.tagIndexer = tagIndexer;
    this.T = tagIndexer.size();
    this.tagCounts = new double[T+1][T+1];
  }

  public void add(int[] truey, int[] predy) {
    assert truey.length == predy.length;
    //logs("true: " + StrUtils.join(ListUtils.subArray(tagIndexer.getObjects(), truey), " "));
    //logs("pred: " + StrUtils.join(ListUtils.subArray(tagIndexer.getObjects(), predy), " "));
    for(int i = 0; i < truey.length; i++) {
      tagCounts[truey[i] == -1 ? T : truey[i]]
               [predy[i] == -1 ? T : predy[i]]++;
    }
  }

  public double getQuality() { return getTagAccuracy(); }
  public double getTagAccuracy() {
    double numCorrect = 0, numTotal = 0;
    for(int s = 0; s < T; s++) {
      for(int t = 0; t < T; t++) {
        if(s == t) numCorrect += tagCounts[s][t];
        numTotal += tagCounts[s][t];
      }
    }
    return numCorrect / numTotal;
  }

  public void log() {
    String[] names = ListUtils.concat(tagIndexer.getObjects().toArray(new String[0]),
        new String[] {"-"});
    logs(StrUtils.join(ListUtils.toObjArray(NumUtils.toInt(tagCounts)),
          names, names));
  }
}

// A bit more pessimistic than the conlleval script
class SegmentingPerformance extends TaggingPerformance {
  boolean[] isSeg, isBegin; // tag -> segment properties
  int[] segMap;
  Indexer<String> segIndexer;
  int S; // Number of segment names (PER, LOC)
  double[][] segCounts; // Confusion matrix on entity types
  double precision, recall, f1;

  public SegmentingPerformance(Indexer<String> tagIndexer) {
    super(tagIndexer);
    this.segIndexer = new Indexer();
    this.isSeg = new boolean[T];
    this.isBegin = new boolean[T];
    this.segMap = new int[T];
    
    for(int t = 0; t < T; t++) { // For each tag...
      String tag = tagIndexer.getObject(t);
      if(tag.length() >= 3 && tag.charAt(1) == '-') { // If represents [BI]-...
        isSeg[t] = true;
        isBegin[t] = tag.charAt(0) == 'B';
        segMap[t] = segIndexer.getIndex(tag.substring(2));
      }
    }

    this.S = segIndexer.size();
    this.segCounts = new double[S+1][S+1];
  }

  // Return a pair of
  //  - location i -> segment at that location (S if none)
  //  - location i -> length of the entity (-1 if none)
  Pair<int[],int[]> tags2segs(int[] tags) {
    int N = tags.length;
    int[] segments = new int[N];
    int[] lengths = new int[N];
    int lastBeginI = -1; // Last position that was a B-...
    for(int i = 0; i < N; i++) { // For each position...
      int t = tags[i];
      if(t != -1 && isSeg[t]) { // Is a segment
        if(lastBeginI != -1 && !isBegin[t] && segments[lastBeginI] == segMap[t]) {
          // Continuation of previous segment
          segments[i] = S;
          lengths[i] = -1;
        }
        else {
          // New segment (swallows inconsistencies too, e.g. I-PER I-LOC)
          if(lastBeginI != -1) // Finish off old segment
            lengths[lastBeginI] = i-lastBeginI;
          segments[i] = segMap[t];
          // Length will be filled in later
          lastBeginI = i;
        }
      }
      else { // Not a segment
        if(lastBeginI != -1) // Finish off old segment
          lengths[lastBeginI] = i-lastBeginI;
        segments[i] = S;
        lengths[i] = -1;
        lastBeginI = -1;
      }
    }
    if(lastBeginI != -1) // Finish off old segment
      lengths[lastBeginI] = N-lastBeginI;

    return new Pair(segments, lengths);
  }

  public void add(int[] truey, int[] predy) {
    super.add(truey, predy);
    assert truey.length == predy.length;
    Pair<int[],int[]> trueSegs = tags2segs(truey);
    Pair<int[],int[]> predSegs = tags2segs(predy);
    for(int i = 0; i < truey.length; i++) {
      int trueSeg = trueSegs.getFirst()[i];
      int predSeg = predSegs.getFirst()[i];
      int trueLen = trueSegs.getSecond()[i];
      int predLen = predSegs.getSecond()[i];
      if(trueSeg != S && predSeg != S) { // Both proposed an entity
        if(trueLen == predLen) // Lengths have to match
          segCounts[trueSeg][predSeg]++;
        else { // Lengths don't match, so consider as two entities
          segCounts[trueSeg][S]++;
          segCounts[S][predSeg]++;
        }
      }
      else if(trueSeg != S || predSeg != S) // Only one proposed an entity
        segCounts[trueSeg][predSeg]++;
    }
  }

  public void computePrecisionRecall() {
    double numCorrect = 0, numTrue = 0, numPred = 0;
    for(int trues = 0; trues <= S; trues++) {
      for(int preds = 0; preds <= S; preds++) {
        if(trues == preds) numCorrect += segCounts[trues][preds];
        if(trues < S) numTrue += segCounts[trues][preds];
        if(preds < S) numPred += segCounts[trues][preds];
      }
    }
    this.precision = numCorrect / numPred;
    this.recall = numCorrect / numTrue;
    this.f1 = 2 * precision*recall / (precision + recall);
  }

  public double getQuality() {
    computePrecisionRecall();
    return f1;
  }

  public void log() {
    computePrecisionRecall();
    logs("tag accuracy = %s, precision = %s, recall = %s, F1 = %s",
        Fmt.D(getTagAccuracy()), Fmt.D(precision), Fmt.D(recall), Fmt.D(f1));
    String[] names = ListUtils.concat(segIndexer.getObjects().toArray(new String[0]),
        new String[] {"-"});
    logs(StrUtils.join(ListUtils.toObjArray(NumUtils.toInt(segCounts)),
          names, names));
  }
}
