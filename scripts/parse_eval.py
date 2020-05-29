acc=0.0
n=0
for x in range(8):
  try:
    f=open("/tmp/eval_results%d.txt" % x, "r").readlines()
  except:
    continue
  f=f[1].strip().split(' ')
  if f[0]!='loss':
   print("Couldn't parse", x)
  print("File ", x, f[2])
  acc+=float(f[2])
  n+=1

if n>0:
  print(n, acc/n)
