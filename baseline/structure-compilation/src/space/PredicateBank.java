package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

public class PredicateBank {
  public enum FuncType { identity, suffix, prefix, capitalization,
    form, match, separate };

  Options opts;

  public PredicateBank(Options opts) {
    this.opts = opts;
  }

  private Predicate subseqPredicate(int leftOffset, int rightOffset, int track) {
    List<Pair<Integer,Integer>> l = new ArrayList();
    for(int offset = leftOffset; offset <= rightOffset; offset++)
      l.add(new Pair(offset, track));
    return new ConcatPredicate(l);
  }

  public List<Predicate> getPredicates() {
    List<Predicate> predicates = new ArrayList();
    // Specialized case
    if(opts.useNumTracks > 0) {
      int nr = 16, nc = 8, nsigs = 1 << 9;
      for(int i = 0; i < opts.useNumTracks; i++)
        predicates.add(new RawPredicate(0, i));
      if(opts.useSig) {
        for(int size = 6; size <= 12; size += 6)
          for(int r = 0; r < nr; r += size)
            for(int c = 0; c < nc; c += size)
              for(int sig = 0; sig < nsigs; sig++)
                predicates.add(new SignaturePredicate(r, c, Math.min(r+size, nr), Math.min(c+size, nc), sig));
      }
      return predicates; 
    }

    for(int offset = -opts.windowSize; offset <= +opts.windowSize; offset++)
      for(Predicate pred : getProcessPredicates(opts.predFuncTypes, new RawPredicate(0, 0)))
        predicates.add(new OffsetPredicate(pred, offset));
    // Bigram
    predicates.addAll(getProcessPredicates(opts.bigramPredFuncTypes,
                      subseqPredicate(-1, 0, 0)));
    predicates.addAll(getProcessPredicates(opts.bigramPredFuncTypes,
                      subseqPredicate(0, +1, 0)));
    // Trigram
    predicates.addAll(getProcessPredicates(opts.trigramPredFuncTypes,
                      subseqPredicate(-1, +1, 0)));
    return predicates;
  }

  public List<Predicate> getProcessPredicates(List<FuncType> funcTypes, Predicate pred) {
    List<Predicate> predicates = new ArrayList();

    // Construct predicates
    for(FuncType funcType : funcTypes) {
      switch(funcType) {
        case identity:
          addFunc(predicates, pred, new IdentityFunc());
          break;
        case prefix:
          for(int size = opts.minPrefixLength; size <= opts.maxPrefixLength; size++)
            addFunc(predicates, pred, new PrefixFunc(size));
          break;
        case suffix:
          for(int size = opts.minSuffixLength; size <= opts.maxSuffixLength; size++)
            addFunc(predicates, pred, new SuffixFunc(size));
          break;
        case capitalization:
          addFunc(predicates, pred, new IsCapitalizedFunc());
          break;
        case form:
          addFunc(predicates, pred, new FormFunc(false));
          addFunc(predicates, pred, new FormFunc(true));
          break;
        case match:
          for(String s : opts.matchPredicates)
            addFunc(predicates, pred, new MatchRegExpFunc(s));
          break;
        case separate:
          for(String s : opts.separatePredicates)
            addFunc(predicates, pred, new SeparateRegExpFunc(s));
          break;
        default:
          throw Exceptions.unknownCase;
      }
    }
    return predicates;
  }

  private void addFunc(List<Predicate> predicates, Predicate pred, Func func) {
    add(predicates, new FuncPredicate(pred, func));
  }
  private void add(List<Predicate> predicates, Predicate pred) {
    predicates.add(pred);
  }

  public static class Options {
    @Option public int windowSize = 1;

    @Option(gloss="A list of functions to apply on the current word")
      public ArrayList<FuncType> predFuncTypes =
        ListUtils.newList(FuncType.identity);

    @Option(gloss="A list of functions to apply on bigrams")
      public ArrayList<FuncType> bigramPredFuncTypes = new ArrayList();
        //ListUtils.newList(FuncType.identity, FuncType.form);
    @Option(gloss="A list of functions to apply on trigrams")
      public ArrayList<FuncType> trigramPredFuncTypes = new ArrayList();
        //ListUtils.newList(FuncType.form);

    // Thresholding predicates (applies to some predicates)
    @Option(gloss="Keep predicate only if it occurs at least this many times")
      public int minOccur = 0;
    @Option(gloss="Keep predicate only if it occurs at most this many times")
      public int maxOccur = Integer.MAX_VALUE;

    @Option(gloss="A list of match predicates")
      public ArrayList<String> matchPredicates = new ArrayList();
    @Option(gloss="A list of match predicates (e.g., the||no)")
      public ArrayList<String> separatePredicates = new ArrayList();

    @Option(gloss="Minimum prefix length") public int minPrefixLength = 1;
    @Option(gloss="Maximum prefix length") public int maxPrefixLength = 3;
    @Option(gloss="Minimum suffix length") public int minSuffixLength = 1;
    @Option(gloss="Maximum suffix length") public int maxSuffixLength = 3;

    @Option(gloss="Predicates are data[current position][all tracks]")
      public int useNumTracks = 0;
    @Option(gloss="Use signature predicates")
      public boolean useSig = false;
  }
}
