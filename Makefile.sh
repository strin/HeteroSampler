run () {
  echo $1
  $1
}

cd scilog
COMPILE_SCILOG="g++ -std=c++11 --shared -o libscilog.so src/*.cpp -I./inc -fPIC"
LINK_SCILOG="g++ -std=c++11 -o testlog testlog.cpp -L./ -lscilog -I./inc"
run "$COMPILE_SCILOG"
run "$LINK_SCILOG"
cd ..
COMPILE_HS="g++ --shared -o libheterosampler.so -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual src/*.cpp
-Iinc/ -Iscilog/inc -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options -fPIC"
COMPILE_TAGGING="g++ -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual model_tagging.cpp -L./ -lheterosampler -Iinc/ -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options -o bin/tagging"
COMPILE_POLICY="g++ -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual test_policy.cpp -L./ -lheterosampler -Iinc/ -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options -o bin/policy -lhdf5 "
run "$COMPILE_HS"
run "$COMPILE_TAGGING"
run "$COMPILE_POLICY"