package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

public class ForwardBackward {
  // Input
  private double[][] edgeWeights, nodeWeights;
  private int[][] prevStates, nextStates; // optional (if transitions are sparse)
  private int S; // Number of states
  private int N; // Length of sequence
  
  // Intermediate
  private double[][] inside; // Sum over all weights after node (including node potential)
  private double[][] outside; // Sum over all weights before a node
  private double logZ;

  // For exponentiated gradient: want to be able to keep track of
  // inferences with two parameters values at the same time
  // Usage: call infer(), saveState(), change the weights, infer(),
  // fetch the new marginals; restoreState(), fetch the old marginals
  private double[][] saveInside;
  private double[][] saveOutside;
  private double saveLogZ;
  public void saveState() {
    saveInside = ListUtils.shallowClone(inside);
    saveOutside = ListUtils.shallowClone(outside);
    saveLogZ = logZ;
  }
  public void restoreState() {
    inside = saveInside;
    outside = saveOutside;
    logZ = saveLogZ;
  }

  // Return output
  public void getNodePosteriors(int i, double[] posteriors) {
    for(int s = 0; s < S; s++)
      posteriors[s] = inside[i][s] * outside[i][s];
    //dbg(Fmt.D(inside[i]) + " || " + Fmt.D(outside[i]) + " || " + Fmt.D(posteriors));
    NumUtils.normalize(posteriors);
  }
  public void getEdgePosteriors(int i, double[][] posteriors) {
    assert i+1 < N;
    for(int s = 0; s < S; s++) {
      double A = outside[i][s] * nodeWeights[i][s];
      for(int t = 0; t < S; t++)
        posteriors[s][t] = A * inside[i+1][t] * edgeWeights[s][t];
    }
    NumUtils.normalize(posteriors);
  }
  public double getLogZ() { return logZ; }

  public ForwardBackward(double[][] edgeWeights, double[][] nodeWeights) {
    this(edgeWeights, nodeWeights, null, null);
  }
  public ForwardBackward(double[][] edgeWeights, double[][] nodeWeights,
      int[][] nextStates, int[][] prevStates) {
    this.edgeWeights = edgeWeights;
    this.nodeWeights = nodeWeights;
    this.nextStates = nextStates;
    this.prevStates = prevStates;
    this.N = nodeWeights.length;
    this.S = nodeWeights[0].length;
  }

  public int[] getViterbi() {
    // Special case: no edges, all nodes are independent
    if(edgeWeights == null) {
      int[] seq = new int[N];
      for(int i = 0; i < N; i++)
        seq[i] = ListUtils.maxIndex(nodeWeights[i]);
      return seq;
    }

    // Like inside score, but instead of summing, compute max
    double[][] maxScore = new double[N][S];

    // Compute max probabilities
    for(int i = N-1; i >= 0; i--) { // For each current position...
      for(int s = 0; s < S; s++) { // For each current state...
        double A = nodeWeights[i][s]; // Node
        if(A == 0) { maxScore[i][s] = 0; continue; }

        double B; // Edge
        if(i == N-1)
          B = 1;
        else {
          B = 0;
          // For each next state...
          if(nextStates != null) {
            for(int t : nextStates[s])
              B = Math.max(B, edgeWeights[s][t] * maxScore[i+1][t]);
          }
          else {
            for(int t = 0; t < S; t++)
              B = Math.max(B, edgeWeights[s][t] * maxScore[i+1][t]);
          }
        }

        //NumUtils.assertIsFinite(A);
        //NumUtils.assertIsFinite(B);
        maxScore[i][s] = A*B;
      }

      // Scale
      double max = ListUtils.max(maxScore[i]);
      assert max > 0 : i + " " + max + " ||| " + Fmt.D(maxScore[i]);
      ListUtils.multMut(maxScore[i], 1.0/max);
    }

    // Now do greedy on maxScore to find best sequence
    int[] seq = new int[N];
    int prevs = seq[0] = ListUtils.maxIndex(maxScore[0]); // Find best starting position
    for(int i = 1; i < N; i++) { // For each remaining position...
      int bests = -1;
      double bestScore = Double.NEGATIVE_INFINITY;
      for(int s = 0; s < S; s++) { // For each possible state
        double score = edgeWeights[prevs][s] * maxScore[i][s];
        if(score > bestScore) {
          bestScore = score;
          bests = s;
        }
      }
      prevs = seq[i] = bests;
    }
    return seq;
  }

  // Return posteriors
  public void infer() {
    this.inside = new double[N][S];
    this.outside = new double[N][S];
    this.logZ = 0;

    if(edgeWeights != null)
      NumUtils.assertIsFinite(edgeWeights);
    NumUtils.assertIsFinite(nodeWeights);

    // Special case: no edges, all nodes are independent
    if(edgeWeights == null) {
      ListUtils.set(inside, nodeWeights);
      ListUtils.set(outside, 1);
      for(int i = 0; i < N; i++)
        logZ += Math.log(ListUtils.sum(inside[i]));
      return;
    }

    // Compute inside probabilities
    for(int i = N-1; i >= 0; i--) { // For each current position...
      for(int s = 0; s < S; s++) { // For each current state...
        double A = nodeWeights[i][s]; // Node
        if(A == 0) { inside[i][s] = 0; continue; }

        double B; // Edge
        if(i == N-1)
          B = 1;
        else {
          B = 0;
          // For each next state...
          if(nextStates != null) {
            for(int t : nextStates[s])
              B += edgeWeights[s][t] * inside[i+1][t];
          }
          else {
            for(int t = 0; t < S; t++)
              B += edgeWeights[s][t] * inside[i+1][t];
          }
        }

        //NumUtils.assertIsFinite(A);
        //NumUtils.assertIsFinite(B);
        inside[i][s] = A*B;
      }

      // Scale
      double sum = ListUtils.sum(inside[i]);
      assert sum > 0 : i + " " + sum + " ||| " + Fmt.D(inside[i]);
      logZ += Math.log(sum);
      ListUtils.multMut(inside[i], 1.0/sum);
    }

    // Compute outside probabilities
    for(int i = 0; i < N; i++) { // For each current position...
      for(int s = 0; s < S; s++) { // For each current state...
        double A; // Edge
        if(i == 0)
          A = 1;
        else {
          A = 0;
          // For each previous state
          if(prevStates != null) {
            for(int r : prevStates[s])
              A += outside[i-1][r] * nodeWeights[i-1][r] * edgeWeights[r][s];
          }
          else {
            for(int r = 0; r < S; r++)
              A += outside[i-1][r] * nodeWeights[i-1][r] * edgeWeights[r][s];
          }
        }
        outside[i][s] = A;
      }

      // Scale
      double sum = ListUtils.sum(outside[i]);
      assert sum > 0 : i + " " + Fmt.D(outside);
      ListUtils.multMut(outside[i], 1.0/sum);
    }

    NumUtils.assertIsFinite(logZ);
  }

  // Do inference from start to end alone instead of 0..N-1
  // For implementing piecewise training.
  // Inefficient: Allocates O(N) memory.
  public void subInfer(int start, int end) {
    this.inside = new double[N][S];
    this.outside = new double[N][S];
    this.logZ = 0;

    if(edgeWeights != null)
      NumUtils.assertIsFinite(edgeWeights);
    NumUtils.assertIsFinite(nodeWeights);

    // Compute inside probabilities
    for(int i = end; i >= start; i--) { // For each current position...
      for(int s = 0; s < S; s++) { // For each current state...
        double A = nodeWeights[i][s]; // Node
        if(A == 0) { inside[i][s] = 0; continue; }

        double B; // Edge
        if(i == end)
          B = 1;
        else {
          B = 0;
          // For each next state...
          if(nextStates != null) {
            for(int t : nextStates[s])
              B += edgeWeights[s][t] * inside[i+1][t];
          }
          else {
            for(int t = 0; t < S; t++)
              B += edgeWeights[s][t] * inside[i+1][t];
          }
        }

        //NumUtils.assertIsFinite(A);
        //NumUtils.assertIsFinite(B);
        inside[i][s] = A*B;
      }

      // Scale
      double sum = ListUtils.sum(inside[i]);
      assert sum > 0 : i + " " + sum + " ||| " + Fmt.D(inside[i]);
      logZ += Math.log(sum);
      ListUtils.multMut(inside[i], 1.0/sum);
    }

    // Compute outside probabilities
    for(int i = start; i <= end; i++) { // For each current position...
      for(int s = 0; s < S; s++) { // For each current state...
        double A; // Edge
        if(i == start)
          A = 1;
        else {
          A = 0;
          // For each previous state
          if(prevStates != null) {
            for(int r : prevStates[s])
              A += outside[i-1][r] * nodeWeights[i-1][r] * edgeWeights[r][s];
          }
          else {
            for(int r = 0; r < S; r++)
              A += outside[i-1][r] * nodeWeights[i-1][r] * edgeWeights[r][s];
          }
        }
        outside[i][s] = A;
      }

      // Scale
      double sum = ListUtils.sum(outside[i]);
      assert sum > 0 : i + " " + Fmt.D(outside);
      ListUtils.multMut(outside[i], 1.0/sum);
    }
  }
}
