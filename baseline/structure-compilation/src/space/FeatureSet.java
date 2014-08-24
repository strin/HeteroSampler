package space;

import java.io.*;
import java.util.*;
import java.util.concurrent.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

/**
 * Usage:
 * extractPredicates() on training examples
 * cachePredicates() on all examples
 */
public class FeatureSet {
  int P; // Number of input predicates
  Predicate[] predicates; // List of predicates

  // Computed based on examples
  int[] offsets; // predicate p -> starting position in parameter vector
  int T; // Number of output tags

  // Map to strings
  Indexer[] predIndexer;
  Indexer<String> tagIndexer;

  public FeatureSet(List<Predicate> predicates) {
    this.predicates = predicates.toArray(new Predicate[0]);
    this.P = this.predicates.length;
  }

  public void outputParams(String path, double[] params, double threshold) {
    begin_track("outputParams(%s): %d parameters", path, params.length);
    PrintWriter out = IOUtils.openOutEasy(path);
    if(out == null) return;
    // Node features
    for(int p = 0; p < P; p++) {
      for(int k = 0; k < predIndexer[p].size(); k++) {
        for(int t = 0; t < T; t++) {
          double v = params[offsets[p] + k * T + t];
          if(Math.abs(v) < threshold) continue;
          out.printf("%s|%s|%s\t%s\n",
              predicates[p].name(),
              predIndexer[p].getObject(k), tagIndexer.getObject(t),
              Fmt.D(v));
        }
      }
    }
    // Edge features
    for(int s = 0; s < T; s++) {
      for(int t = 0; t < T; t++) {
        double v = params[offsets[P] + s * T + t];
        if(v < threshold) continue;
        out.printf("EDGE|%s|%s\t%s\n",
          tagIndexer.getObject(s), tagIndexer.getObject(t),
          Fmt.D(v));
      }
    }
    out.close();
    end_track();
  }

  public void extractPredicates(List<Example> examples) {
    begin_track("extractPredicates("+examples.size()+" examples)", true);
    // FUTURE: threshold features that are too rare
    // Extract predicates and throw out ones that don't occur enough times
    this.predIndexer = new Indexer[P];
    for(int p = 0; p < P; p++)
      predIndexer[p] = new Indexer();
    this.tagIndexer = new Indexer();
    begin_track("Examples");
    for(int e = 0; e < examples.size(); e++) {
      Example ex = examples.get(e);
      begin_track("Example %d/%d", e, examples.size());
      for(int i = 0; i < ex.N; i++) {
        for(int p = 0; p < P; p++)
          predIndexer[p].getIndex(predicates[p].eval(ex, i));
        tagIndexer.getIndex(ex.data[i][ex.data[i].length-1]);
      }
      end_track();
    }
    end_track();

    // Setup offsets
    this.T = tagIndexer.size();
    this.offsets = new int[P+1];
    for(int p = 0; p < P; p++)
      offsets[p+1] += offsets[p] + predIndexer[p].size()*T;

    // Log predicates
    begin_track(P+ " predicates", true);
    for(int p = 0; p < P; p++)
      logs("%s: %d values", predicates[p].name(), predIndexer[p].size());
    Execution.putLogRec("numTags", T);
    Execution.putLogRec("numFeatures", size());
    end_track();

    end_track();
  }

  public void cachePredicates(List<Example> examples, boolean getLabels) {
    begin_track("cachePredicates(%d examples)", examples.size());
    for(int i = 0; i < examples.size(); i++) {
      Example ex = examples.get(i);
      begin_track("Example %d/%d", i, examples.size());
      cachePredicates(ex, getLabels);
      examples.set(i, ex); // if SerializedList
      end_track();
    }
    end_track();
  }

  public void cachePredicates(Example ex, boolean getLabels) {
    // Precompute all predicates, use integers and store in the example
    ex.xIndices = new int[ex.N][P];
    if(getLabels)
      ex.y = new int[ex.N];
    for(int i = 0; i < ex.N; i++) {
      for(int p = 0; p < P; p++)
        ex.xIndices[i][p] = predIndexer[p].indexOf(predicates[p].eval(ex, i));
      // Assume last column is label
      if(getLabels)
        ex.y[i] = tagIndexer.indexOf(ex.data[i][ex.data[i].length-1]);
    }
  }

  // Number of features
  public int size() { return offsets[P] + T*T; }

  public double nodeScore(Example ex, int i, int y, double[] params) {
    if(y == -1) return 0;
    double v = 0;
    for(int p = 0; p < P; p++) {
      int k = ex.xIndices[i][p];
      if(k == -1) continue;
      v += params[offsets[p] + k * T + y]; 
    }
    return v;
  }

  public double edgeScore(int y1, int y2, double[] params) {
    if(y1 == -1 || y2 == -1) return 0;
    return params[offsets[P] + y1 * T + y2];
  }

  public void nodeUpdate(Example ex, int i, int y, double incr, double[] counts) {
    assert y != -1;
    for(int p = 0; p < P; p++) {
      int k = ex.xIndices[i][p];
      if(k == -1) continue;
      counts[offsets[p] + k * T + y] += incr;
    }
  }

  public void edgeUpdate(int y1, int y2, double incr, double[] counts) {
    assert y1 != -1 && y2 != -1;
    counts[offsets[P] + y1 * T + y2] += incr;
  }
}
