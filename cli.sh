ifind () {
  grep -nr --color "$1" *.cpp  inc/*.h src/*.cpp
}

ireplace () {
  grep -lr --color "$1" *.cpp inc/*.h src/*.cpp | xargs sed -i.bu "s/$1/$2/g"
}

tags () {
  ctags -R -f .tags
}
