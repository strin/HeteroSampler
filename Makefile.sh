cd scilog
g++ --shared -o libscilog.dylib src/*.cpp -I./inc
g++ -o testlog testlog.cpp -L./ -lscilog -I./inc
cd ..
g++ --shared -o libheterosampler.so -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual src/*.cpp -Iinc/ -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options
g++ -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual -o bin/tagging model_tagging.cpp src/*.cpp -Iinc/ -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options
g++ -std=c++11 -Wno-deprecated-register -Wno-overloaded-virtual test_policy.cpp -L./ -lheterosampler -Iinc/ -Iscilog/inc/ -Lscilog/ -lscilog -lboost_program_options -o bin/policy -lhdf5
