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
