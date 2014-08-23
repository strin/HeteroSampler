package space;

import java.io.*;
import java.util.*;
import fig.basic.*;
import fig.exec.*;
import fig.record.*;
import fig.prob.*;
import static fig.basic.LogInfo.*;

interface Func<X,Y> {
  public String name();
  public Y eval(X x);
}

class IdentityFunc<T> implements Func<T,T> {
  public String name() { return "identity"; }
  public T eval(T x) { return x; }
}

class PrefixFunc implements Func<String,String> {
  int length;
  public PrefixFunc(int length) { this.length = length; }
  public String name() { return "prefix"+length; }
  public String eval(String x) {
    return x.substring(0, Math.min(x.length(), length));
  }
}

class SuffixFunc implements Func<String,String> {
  int length;
  public SuffixFunc(int length) { this.length = length; }
  public String name() { return "suffix"+length; }
  public String eval(String x) {
    return x.substring(Math.max(x.length()-length, 0));
  }
}

// Inc. => Aa. (if contract)
class FormFunc implements Func<String,String> {
  private boolean contract;

  public FormFunc(boolean contract) { this.contract = contract; }
  public String name() { return contract ? "form" : "Form"; }
  public String eval(String x) {
    StringBuilder buf = new StringBuilder();
    char lastc = 0;
    for(int i = 0; i < x.length(); i++) {
      char c = x.charAt(i);
      if(Character.isDigit(c)) c = '0';
      else if(Character.isLetter(c)) c = Character.isLowerCase(c) ? 'a' : 'A';
      if(!contract || c != lastc) buf.append(c);
      lastc = c;
    }
    return buf.toString();
  } 
}

// Does x begin with a capital letter?
class IsCapitalizedFunc implements Func<String,Boolean> {
  public String name() { return "isCapitalized"; }
  public Boolean eval(String x) { return Character.isUpperCase(x.charAt(0)); }
}

// x =~ pattern
class MatchRegExpFunc implements Func<String,Boolean> {
  String pattern;
  public MatchRegExpFunc(String pattern) { this.pattern = pattern; }
  public String name() { return "match("+pattern+")"; }
  public Boolean eval(String x) { return x.matches(pattern); }
}

// x -> first pattern that matches; otherwise return null
class SeparateRegExpFunc implements Func<String,String> {
  final String delim = "\\|\\|"; // Delimits patterns
  String[] patterns;
  public SeparateRegExpFunc(String s) {
    //this.patterns = ListUtils.toArray(StrUtils.splitByStr(s, delim));
    this.patterns = s.split(delim);
  }
  public SeparateRegExpFunc(String[] patterns) { this.patterns = patterns; }
  public String name() { return "separate("+StrUtils.join(patterns, delim)+")"; }
  public String eval(String x) {
    for(String pattern : patterns)
      if(x.matches(pattern)) return pattern;
    return null;
  }
}
