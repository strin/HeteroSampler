#pragma once
#ifndef OPENGM_ADAGIBBS_HXX
#define OPENGM_ADAGIBBS_HXX

#include <vector>
#include <string>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <typeinfo>

#include "opengm/opengm.hxx"
#include "opengm/utilities/random.hxx"
#include "opengm/inference/inference.hxx"
#include "opengm/inference/movemaker.hxx"
#include "opengm/operations/minimizer.hxx"
#include "opengm/operations/maximizer.hxx"
#include "opengm/operations/adder.hxx"
#include "opengm/operations/multiplier.hxx"
#include "opengm/operations/integrator.hxx"
#include "opengm/inference/visitors/visitors.hxx"
#include "opengm/inference/gibbs.hxx"

namespace opengm {
  
/// \brief AdaGibbs sampling
template<class GM, class ACC>
class AdaGibbs 
: public Inference<GM, ACC> {
public:
  typedef ACC AccumulationType;
  typedef GM GraphicalModelType;
  OPENGM_GM_TYPE_TYPEDEFS;
  typedef Movemaker<GraphicalModelType> MovemakerType;
  typedef visitors::VerboseVisitor<AdaGibbs<GM, ACC> > VerboseVisitorType;
  typedef visitors::EmptyVisitor<AdaGibbs<GM, ACC> > EmptyVisitorType;
  typedef visitors::TimingVisitor<AdaGibbs<GM, ACC> > TimingVisitorType;
  typedef double ProbabilityType;

  class Parameter {
  public:
    enum VariableProposal {RANDOM, CYCLIC};

    Parameter(
      const size_t maxNumberOfSamplingSteps = 1e5,
      const size_t numberOfBurnInSteps = 1e5,
      const bool useTemp=false,
      const ValueType tmin=0.0001,
      const ValueType tmax=1,
      const IndexType periods=10,
      const VariableProposal variableProposal = RANDOM,
      const std::vector<size_t>& startPoint = std::vector<size_t>()
    )
    :  maxNumberOfSamplingSteps_(maxNumberOfSamplingSteps), 
      numberOfBurnInSteps_(numberOfBurnInSteps), 
      variableProposal_(variableProposal),
      startPoint_(startPoint),
      useTemp_(useTemp),
      tempMin_(tmin),
      tempMax_(tmax),
      periods_(periods){
      p_=static_cast<ValueType>(maxNumberOfSamplingSteps_/periods_);
    }
    bool useTemp_;
    ValueType tempMin_;
    ValueType tempMax_;
    size_t periods_;
    ValueType p_;
    size_t maxNumberOfSamplingSteps_;
    size_t numberOfBurnInSteps_;
    VariableProposal variableProposal_;
    std::vector<size_t> startPoint_;
  };

  AdaGibbs(const GraphicalModelType&, const Parameter& param = Parameter());
  std::string name() const;
  const GraphicalModelType& graphicalModel() const;
  void reset();
  InferenceTermination infer();
  template<class VISITOR>
    InferenceTermination infer(VISITOR&);
  void setStartingPoint(typename std::vector<LabelType>::const_iterator);
  virtual InferenceTermination arg(std::vector<LabelType>&, const size_t = 1) const;

  LabelType markovState(const size_t) const;
  ValueType markovValue() const;
  LabelType currentBestState(const size_t) const;
  ValueType currentBestValue() const;

private:
  ValueType cosTemp(const ValueType arg,const ValueType periode,const ValueType min,const ValueType max)const{
    return static_cast<ValueType>(((std::cos(arg/periode)+1.0)/2.0)*(max-min))+min;
    //if(v<
  }

  ValueType getTemperature(const size_t step)const{
    return cosTemp( 
      static_cast<ValueType>(step),
      parameter_.p_,
      parameter_.tempMin_,
      parameter_.tempMax_
    );
  }
  Parameter parameter_;
  const GraphicalModelType& gm_;
  MovemakerType movemaker_;
  std::vector<size_t> currentBestState_;
  ValueType currentBestValue_;
  bool inInference_;
};

template<class GM, class ACC>
inline
AdaGibbs<GM, ACC>::AdaGibbs
(
  const GraphicalModelType& gm, 
  const Parameter& parameter
)
:  parameter_(parameter), 
  gm_(gm), 
  movemaker_(gm), 
  currentBestState_(gm.numberOfVariables()),
  currentBestValue_()
{
  inInference_=false;
  ACC::ineutral(currentBestValue_);
  if(parameter.startPoint_.size() != 0) {
    if(parameter.startPoint_.size() == gm.numberOfVariables()) {
      movemaker_.initialize(parameter.startPoint_.begin());
      currentBestState_ = parameter.startPoint_;
      currentBestValue_ = movemaker_.value();
    }
    else {
      throw RuntimeError("parameter.startPoint_.size() is neither zero nor equal to the number of variables.");
    }
  }
}

template<class GM, class ACC>
inline void
AdaGibbs<GM, ACC>::reset() {
  if(parameter_.startPoint_.size() != 0) {
    if(parameter_.startPoint_.size() == gm_.numberOfVariables()) {
      movemaker_.initialize(parameter_.startPoint_.begin());
      currentBestState_ = parameter_.startPoint_;
      currentBestValue_ = movemaker_.value();
    }
    else {
      throw RuntimeError("parameter.startPoint_.size() is neither zero nor equal to the number of variables.");
    }
  }
  else {
    movemaker_.reset();
    std::fill(currentBestState_.begin(), currentBestState_.end(), 0);
  }
}

template<class GM, class ACC>
inline void
AdaGibbs<GM, ACC>::setStartingPoint
(
  typename std::vector<typename AdaGibbs<GM, ACC>::LabelType>::const_iterator begin
) {
  try{
    movemaker_.initialize(begin);

    for(IndexType vi=0;vi<static_cast<IndexType>(gm_.numberOfVariables());++vi ){
      currentBestState_[vi]=movemaker_.state(vi);
    }
    currentBestValue_ = movemaker_.value();

  }
  catch(...) {
    throw RuntimeError("unsuitable starting point");
  }
}

template<class GM, class ACC>
inline std::string
AdaGibbs<GM, ACC>::name() const
{
  return "AdaGibbs";
}

template<class GM, class ACC>
inline const typename AdaGibbs<GM, ACC>::GraphicalModelType&
AdaGibbs<GM, ACC>::graphicalModel() const
{
  return gm_;
}

template<class GM, class ACC>
inline InferenceTermination
AdaGibbs<GM, ACC>::infer()
{
  EmptyVisitorType visitor;
  return infer(visitor);
}

template<class GM, class ACC>
template<class VISITOR>
InferenceTermination AdaGibbs<GM, ACC>::infer(
  VISITOR& visitor
) {
  inInference_=true;
  visitor.begin(*this);
  opengm::RandomUniform<size_t> randomVariable(0, gm_.numberOfVariables());
  opengm::RandomUniform<ProbabilityType> randomProb(0, 1);
  
  if(parameter_.useTemp_==false) {
    for(size_t iteration = 0; iteration < parameter_.maxNumberOfSamplingSteps_ + parameter_.numberOfBurnInSteps_; ++iteration) {
      // select variable
      size_t variableIndex = 0;
      if(this->parameter_.variableProposal_ == Parameter::RANDOM) {
        variableIndex = randomVariable();
      }
      else if(this->parameter_.variableProposal_ == Parameter::CYCLIC) {
        variableIndex < gm_.numberOfVariables() - 1 ? ++variableIndex : variableIndex = 0;
      }
      const bool burningIn = (iteration < parameter_.numberOfBurnInSteps_);

      const ValueType oldValue = movemaker_.value();
      // draw label
      opengm::RandomUniform<size_t> randomLabel(0, gm_.numberOfLabels(variableIndex));
      const size_t label = randomLabel();
      for(size_t label = 0; label < gm_.numberOfLabels(variableIndex); label++) {
	const ValueType newValue = movemaker_.valueAfterMove(&variableIndex, &variableIndex + 1, &label);
      }
      if(AccumulationType::bop(newValue, oldValue)) {
        movemaker_.move(&variableIndex, &variableIndex + 1, &label);
        if(AccumulationType::bop(newValue, currentBestValue_) && newValue != currentBestValue_) {
          currentBestValue_ = newValue;
          for(size_t k = 0; k < currentBestState_.size(); ++k) {
            currentBestState_[k] = movemaker_.state(k);
          }
        }
        visitor(*this);
        //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
      }
      else {
        const ProbabilityType pFlip =
          detail_gibbs::ValuePairToProbability<
            OperatorType, AccumulationType, ProbabilityType
          >::convert(newValue, oldValue);
        if(randomProb() < pFlip) {
          movemaker_.move(&variableIndex, &variableIndex + 1, &label); 
          visitor(*this);
          //visitor(*this, newValue, currentBestValue_, iteration, true, burningIn);
        }
        else {
          visitor(*this);
         // visitor(*this, newValue, currentBestValue_, iteration, false, burningIn);
        }
      }
      ++iteration;
     }
    
  }
  //visitor.end(*this, currentBestValue_, currentBestValue_);
  visitor.end(*this);
  inInference_=false;
  return NORMAL;
}

template<class GM, class ACC>
inline InferenceTermination
AdaGibbs<GM, ACC>::arg
(
  std::vector<LabelType>& x, 
  const size_t N
) const {
  if(N == 1) {
    x.resize(gm_.numberOfVariables());
    for(size_t j = 0; j < x.size(); ++j) {
      if(!inInference_)
        x[j] = currentBestState_[j];
      else{
        x[j] = movemaker_.state(j);
      }
    }
    return NORMAL;
  }
  else {
    return UNKNOWN;
  }
}

template<class GM, class ACC>
inline typename AdaGibbs<GM, ACC>::LabelType
AdaGibbs<GM, ACC>::markovState
(
  const size_t j
) const
{
  OPENGM_ASSERT(j < gm_.numberOfVariables());
  return movemaker_.state(j);
}

template<class GM, class ACC>
inline typename AdaGibbs<GM, ACC>::ValueType
AdaGibbs<GM, ACC>::markovValue() const
{
  return movemaker_.value();
}

template<class GM, class ACC>
inline typename AdaGibbs<GM, ACC>::LabelType
AdaGibbs<GM, ACC>::currentBestState
(
  const size_t j
) const
{
  OPENGM_ASSERT(j < gm_.numberOfVariables());
  return currentBestState_[j];
}

template<class GM, class ACC>
inline typename AdaGibbs<GM, ACC>::ValueType
AdaGibbs<GM, ACC>::currentBestValue() const
{
  return currentBestValue_;
}

} // namespace opengm

#endif // #ifndef OPENGM_GIBBS_HXX
