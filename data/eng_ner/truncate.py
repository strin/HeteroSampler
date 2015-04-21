import sys

input_f = sys.argv[1]
output_f = sys.argv[2]
allcount = int(sys.argv[3])

lines = file(input_f).readlines()
f = open(output_f, 'w')
count = 0

for line in lines:
  if line == '\n':
    count += 1
  if count >= allcount+1:
    break
  else:
    f.write(line)

f.close()

