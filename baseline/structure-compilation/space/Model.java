package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

// Summary of the parameters params needed to do inference,
// i.e., define p(y | x)
class ExamplePotentials implements Serializable {
  double[][] edgePotentials; // tag, tag
  double[][] nodePotentials; // position, tag
  double extraLogZ; // Doesn't work for piecewise training

  public ExamplePotentials(Example ex, FeatureSet features, boolean includeEdges) {
    int T = features.T;
    int N = ex.N;
    // Uniform potentials
    if(includeEdges)
      this.edgePotentials = ListUtils.newDouble(T, T, 1);
    this.nodePotentials = ListUtils.newDouble(N, T, 1);
    this.extraLogZ = 0;
  }

  public ExamplePotentials(Example ex, FeatureSet features, double[] params, boolean includeEdges) {
    int T = features.T;
    int N = ex.N;

    // We scale the potentials to a reasonable range,
    // and dump the log scale into extraLogZ
    this.extraLogZ = 0;

    // Edge potentials
    if(includeEdges) {
      this.edgePotentials = new double[T][T];
      for(int t1 = 0; t1 < T; t1++)
        for(int t2 = 0; t2 < T; t2++)
          edgePotentials[t1][t2] = features.edgeScore(t1, t2, params);
      double max = ListUtils.max(edgePotentials);
      for(int t1 = 0; t1 < T; t1++)
        for(int t2 = 0; t2 < T; t2++)
          edgePotentials[t1][t2] = Math.exp(edgePotentials[t1][t2] - max);
      extraLogZ += max * (N-1);
    }

    // Node potentials
    this.nodePotentials = new double[N][T];
    for(int i = 0; i < N; i++) {
      // Compute the scores (log potentials) for each current tag
      for(int t = 0; t < T; t++)
        nodePotentials[i][t] = features.nodeScore(ex, i, t, params);

      // Exp and fill in the scores (which depend on states)
      // Actually, this is probably not necessary
      double max = ListUtils.max(nodePotentials[i]);
      extraLogZ += max;
      for(int t = 0; t < T; t++)
        nodePotentials[i][t] = Math.exp(nodePotentials[i][t] - max);
    }
  }

  // Make sure potentials aren't too big or small to prevent floating point issues
  public void normalizeRange() {
    int N = nodePotentials.length;
    int T = nodePotentials[0].length;
    double max;

    // Edge potentials
    if(edgePotentials != null) {
      max = 0;
      for(int s = 0; s < T; s++)
        for(int t = 0; t < T; t++)
          max = Math.max(max, edgePotentials[s][t]);
      if(max > 0) {
        extraLogZ += Math.log(max);
        for(int s = 0; s < T; s++)
          for(int t = 0; t < T; t++)
            edgePotentials[s][t] /= max;
      }
      //NumUtils.assertIsFinite(edgePotentials);
    }

    // Node potentials
    for(int i = 0; i < N; i++) {
      max = 0;
      for(int t = 0; t < T; t++)
        max = Math.max(max, nodePotentials[i][t]);
      if(max > 0) {
        extraLogZ += Math.log(max);
        for(int t = 0; t < T; t++)
          nodePotentials[i][t] /= max;
      }
    }
    //NumUtils.assertIsFinite(nodePotentials);
  }
}

/**
 * Implementation of first-order chain CRFs.
 */
public class Model {
  Options opts;
  FeatureSet features;
  int T; // Number of tags

  public Model(Options opts, FeatureSet features) {
    this.opts = opts;
    this.features = features;
    this.T = features.T;
  }

  // Return the logLikelihood
  // Update the counts if necessary
  public InferState getInferState(Example ex, double[] params) {
    return new InferState(ex, params);
  }

  /**
   * Usage:
   *  - infer() to get logLikelihood
   *  - updateCounts(...) if computing gradient
   *  - findBest()
   */
  public class InferState {
    // Input to inference
    Example ex;
    int N;
    double[] params;
    double sumEdgeMutualInfo; // Sum of mutual information on edges

    // Intermediate step
    ExamplePotentials potentials;
    ForwardBackward FB;

    public InferState(Example ex, double[] params) {
      this.ex = ex;
      this.N = ex.N;
      this.params = params;
    }

    // Return logLikelihood
    public double infer() {
      double logLikelihood = 0;
      logLikelihood += inferNumerator(+1);
      NumUtils.assertIsFinite(logLikelihood);
      logLikelihood += inferDenominator(-1);
      NumUtils.assertIsFinite(logLikelihood);
      return logLikelihood;
    }
    public double subInfer(int start, int end) {
      double logLikelihood = 0;
      logLikelihood += subInferNumerator(+1, start, end);
      NumUtils.assertIsFinite(logLikelihood);
      logLikelihood += subInferDenominator(-1, start, end);
      NumUtils.assertIsFinite(logLikelihood);
      return logLikelihood;
    }

    public void updateCounts(double scale, double[] counts) {
      synchronized(counts) {
        updateCountsNumerator(+scale, counts);
        updateCountsDenominator(-scale, counts);
      }
    }
    public void subUpdateCounts(double scale, double[] counts, int start, int end) {
      NumUtils.assertIsFinite(scale);
      synchronized(counts) {
        subUpdateCountsNumerator(+scale, counts, start, end);
        subUpdateCountsDenominator(-scale, counts, start, end);
      }
    }

    private double inferNumerator(double sign) {
      double logZ = 0;
      for(int i = 0; i < N; i++) {
        logZ += sign*features.nodeScore(ex, i, ex.y[i], params);
        if(opts.includeEdges && i+1 < N)
          logZ += sign*features.edgeScore(ex.y[i], ex.y[i+1], params);
      }
      return logZ;
    }
    private double subInferNumerator(double sign, int start, int end) {
      double logZ = 0;
      for(int i = start; i <= end; i++) {
        logZ += sign*features.nodeScore(ex, i, ex.y[i], params);
        if(opts.includeEdges && i+1 <= end)
          logZ += sign*features.edgeScore(ex.y[i], ex.y[i+1], params);
      }
      return logZ;
    }

    // If have potentials to use for inference not based on params
    public void initFB(ExamplePotentials potentials) {
      if(this.FB != null) return;
      if(potentials == null)
        potentials = new ExamplePotentials(ex, features, params, opts.includeEdges);
      this.potentials = potentials;
      this.FB = new ForwardBackward(potentials.edgePotentials, potentials.nodePotentials, null, null);
    }

    private double inferDenominator(double sign) {
      initFB(null);
      FB.infer();
      return sign * (FB.getLogZ() + potentials.extraLogZ);
    }
    private double subInferDenominator(double sign, int start, int end) {
      initFB(null);
      FB.subInfer(start, end);
      return sign * (FB.getLogZ() + potentials.extraLogZ);
    }

    public void updateCountsNumerator(double sign, double[] counts) {
      for(int i = 0; i < N; i++) {
        features.nodeUpdate(ex, i, ex.y[i], sign, counts);
        if(opts.includeEdges && i+1 < N)
          features.edgeUpdate(ex.y[i], ex.y[i+1], sign, counts);
      }
    }
    public void subUpdateCountsNumerator(double sign, double[] counts, int start, int end) {
      for(int i = start; i <= end; i++) {
        features.nodeUpdate(ex, i, ex.y[i], sign, counts);
        if(opts.includeEdges && i+1 <= end)
          features.edgeUpdate(ex.y[i], ex.y[i+1], sign, counts);
      }
    }

    public void updateCountsUniform(double sign, double[] counts) {
      for(int i = 0; i < N; i++) { // For each position
        // Update node features
        for(int s = 0; s < T; s++)
          features.nodeUpdate(ex, i, s, sign/T, counts);
        // Update edge features
        if(opts.includeEdges && i+1 < N) {
          for(int s = 0; s < T; s++)
            for(int t = 0; t < T; t++)
              features.edgeUpdate(s, t, sign/(T*T), counts);
        }
      }
    }

    public void updateCountsDenominator(double sign, double[] counts) {
      double[] nodePosteriors = new double[T];
      double[][] edgePosteriors = new double[T][T];
      for(int i = 0; i < N; i++) { // For each position
        // Update node features
        FB.getNodePosteriors(i, nodePosteriors);
        for(int s = 0; s < T; s++)
          features.nodeUpdate(ex, i, s, sign*nodePosteriors[s], counts);
        // Update edge features
        if(opts.includeEdges && i+1 < N) {
          FB.getEdgePosteriors(i, edgePosteriors);
          for(int s = 0; s < T; s++)
            for(int t = 0; t < T; t++)
              features.edgeUpdate(s, t, sign*edgePosteriors[s][t], counts);
        }
      }
    }
    public void subUpdateCountsDenominator(double sign, double[] counts, int start, int end) {
      double[] nodePosteriors = new double[T];
      double[][] edgePosteriors = new double[T][T];
      for(int i = start; i <= end; i++) { // For each position
        // Update node features
        FB.getNodePosteriors(i, nodePosteriors);
        for(int s = 0; s < T; s++)
          features.nodeUpdate(ex, i, s, sign*nodePosteriors[s], counts);
        // Update edge features
        if(opts.includeEdges && i+1 <= end) {
          FB.getEdgePosteriors(i, edgePosteriors);
          for(int s = 0; s < T; s++)
            for(int t = 0; t < T; t++)
              features.edgeUpdate(s, t, sign*edgePosteriors[s][t], counts);
        }
      }
    }

    public int piecewiseStart(int i) { return Math.max(i-opts.piecewiseWindowSize, 0); }
    public int piecewiseEnd(int i) { return Math.min(i+opts.piecewiseWindowSize, N-1); }

    public double[] computeEdgeMutualInfo() {
      if(!opts.includeEdges) return new double[0];

      // Compute mutual information
      double[] mutualInfos = new double[N-1];
      double[] nodePosteriors1 = new double[T];
      double[] nodePosteriors2 = new double[T];
      double[][] edgePosteriors = new double[T][T];
      for(int i = 0; i < N-1; i++) { // For each position
        FB.getNodePosteriors(i, nodePosteriors1);
        FB.getNodePosteriors(i+1, nodePosteriors2);
        FB.getEdgePosteriors(i, edgePosteriors);
        double mi = 0;
        for(int s = 0; s < T; s++) {
          for(int t = 0; t < T; t++) {
            if(edgePosteriors[s][t] == 0) continue;
            mi += edgePosteriors[s][t] *
              Math.log(edgePosteriors[s][t] / nodePosteriors1[s] / nodePosteriors2[t]);
          }
        }
        mutualInfos[i] = mi;
      }
      return mutualInfos;
    }

    public int[] getBestOutput() {
      initFB(null);
      if(opts.piecewise) {
        int W = opts.piecewiseWindowSize;
        double[] posteriors = new double[T];
        int[] tag = new int[N];
        for(int i = 0; i < N; i++) {
          FB.subInfer(piecewiseStart(i), piecewiseEnd(i));
          FB.getNodePosteriors(i, posteriors);
          tag[i] = ListUtils.maxIndex(posteriors);
        }
        return tag;
      }
      else if(opts.usePosteriorDecoding) {
        double[] posteriors = new double[T];
        int[] tag = new int[N];
        FB.infer();
        for(int i = 0; i < N; i++) { // For each position
          FB.getNodePosteriors(i, posteriors);
          tag[i] = ListUtils.maxIndex(posteriors);
        }
        return tag;
      }
      else
        return FB.getViterbi();
    }

  }

  public Performance newPerformance() {
    if(opts.segmenting)
      return new SegmentingPerformance(features.tagIndexer);
    else
      return new TaggingPerformance(features.tagIndexer);
  }

  ////////////////////////////////////////////////////////////

  public static class Options {
    @Option(gloss="Include edges (CRF) or not (ILR)")
      public boolean includeEdges = true;
    @Option(gloss="Is this a segmenting task")
      public boolean segmenting = false;
    @Option(gloss="Use posterior decoding?")
      public boolean usePosteriorDecoding = false;
    @Option(gloss="Use piecewise training and posterior decoding on the piece at test time")
      public boolean piecewise = false;
    @Option public int piecewiseWindowSize = 3;
  }
}
