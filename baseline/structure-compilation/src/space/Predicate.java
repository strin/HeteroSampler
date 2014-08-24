package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

/**
 * A predicate maps a position of a sequence to an object, which will be mapped into an integer.
 */
interface Predicate {
  public String name(); // Generic name of the predicate 
  public Object eval(Example ex, int i);
}

class OffsetPredicate implements Predicate {
  final String boundaryStr = "###";
  Predicate pred;
  int offset;

  public OffsetPredicate(Predicate pred, int offset) {
    this.pred = pred;
    this.offset = offset;
  }

  public String name() { return pred.name() + (offset >= 0 ? "+"+offset : ""+offset); }
  public Object eval(Example ex, int i) {
    i += offset;
    if(i < 0 || i >= ex.N) return boundaryStr;
    return pred.eval(ex, i);
  }
}

// Invokes a predicate and then wraps it in a function
class FuncPredicate implements Predicate {
  Predicate pred;
  Func func;
  public FuncPredicate(Predicate pred, Func func) {
    this.pred = pred;
    this.func = func;
  }

  public String name() { return String.format("%s(%s)", func.name(), pred.name()); }
  public Object eval(Example ex, int i) {
    return func.eval(pred.eval(ex, i));
  }
}

// Take the cross product of a bunch of predicates
class ProductPredicate implements Predicate {
  Predicate[] predicates;
  public ProductPredicate(Predicate[] predicates) {
    this.predicates = predicates;
  }
  public String name() {
    List<String> names = new ArrayList();
    for(Predicate pred : predicates) names.add(pred.name());
    return "("+StrUtils.join(names, " x ")+")";
  }
  public Object eval(Example ex, int i) {
    List l = new ArrayList(predicates.length);
    for(Predicate pred : predicates)
      l.add(pred.eval(ex, i));
    return l;
  }
}

// Given a list of (offset,track) pairs,
// concatenate the data from those pairs.
class ConcatPredicate implements Predicate {
  final String boundaryStr = "###";
  String delim = " ";
  List<Pair<Integer,Integer>> offsetTracks;
  public ConcatPredicate(List<Pair<Integer,Integer>> offsetTracks) {
    this.offsetTracks = offsetTracks;
  }
  public String name() {
    List<String> names = new ArrayList();
    for(Pair<Integer,Integer> ot : offsetTracks)
      names.add("("+ot.getFirst()+","+ot.getSecond()+")");
    return StrUtils.join(names, "+");
  }
  public Object eval(Example ex, int i) {
    StringBuilder buf = new StringBuilder();
    for(Pair<Integer,Integer> ot : offsetTracks) {
      if(buf.length() > 0) buf.append(delim);
      int ii = i+ot.getFirst();
      if(ii < 0 || ii >= ex.N) buf.append(boundaryStr);
      else buf.append(ex.data[ii][ot.getSecond()]);
    }
    return buf.toString();
  }
}

// Return data[i+offset][track]
// Special case of ConcatPredicate
class RawPredicate implements Predicate {
  final String boundaryStr = "###";
  int offset, track;
  public RawPredicate(int offset, int track) {
    this.offset = offset;
    this.track = track;
  }
  public String name() {
    return "("+offset+","+track+")";
  }
  public Object eval(Example ex, int i) {
    i += offset;
    if(i < 0 || i >= ex.N) return boundaryStr;
    return ex.data[i][track];
  }
}

// 05/29/08: for letter OCR 
class SignaturePredicate implements Predicate {
  int nr = 16;
  int nc = 8;
  int nsigs = 1 << 9;
  int r0, c0, r1, c1, sig;
  public SignaturePredicate(int r0, int c0, int r1, int c1, int sig) {
    this.r0 = r0;
    this.c0 = c0;
    this.r1 = r1;
    this.c1 = c1;
    this.sig = sig;
  }
  public String name() {
    return String.format("%d{(%d,%d)-(%d,%d)}", sig, r0, c1, r1, c1);
  }

  int getPixel(String[] x, int r, int c) {
    if(r < 0 || r >= nr || c < 0 || c >= nc) return 0;
    return x[r*nc+c].charAt(0) == '1' ? 1 : 0;
  }

  int sig(String[] x, int r, int c) {
    int i = 0;
    int a = 0;
    a |= (getPixel(x, r-1, c-1) << i++);
    a |= (getPixel(x, r-1, c) << i++);
    a |= (getPixel(x, r-1, c+1) << i++);
    a |= (getPixel(x, r, c-1) << i++);
    a |= (getPixel(x, r, c) << i++);
    a |= (getPixel(x, r, c+1) << i++);
    a |= (getPixel(x, r+1, c-1) << i++);
    a |= (getPixel(x, r+1, c) << i++);
    a |= (getPixel(x, r+1, c+1) << i++);
    return a;
  }
  public Object eval(Example ex, int j) {
    if(ex.sigs == null) {
      ex.sigs = new int[ex.N][ex.data[0].length];
      for(int i = 0; i < ex.N; i++)
        for(int r = 0; r < nr; r++)
          for(int c = 0; c < nc; c++)
            ex.sigs[i][r*nc+c] = sig(ex.data[i], r, c);
    }
    for(int r = r0; r < r1; r++) {
      for(int c = c0; c < c1; c++) {
        if(ex.sigs[j][r*nc+c] == sig) return true;
      }
    }
    return false;
  }
}
