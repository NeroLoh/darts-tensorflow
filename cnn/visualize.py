import sys
import genotypes
from graphviz import Digraph


def plot(genotype, filename):
  g = Digraph(
      format='png',
      edge_attr=dict(fontsize='20', fontname="times"),
      node_attr=dict(style='filled', shape='rect', align='center', fontsize='20', height='0.5', width='0.5', penwidth='2', fontname="times"),
      engine='dot')
  g.body.extend(['rankdir=LR'])

  g.node("c_{k-2}", fillcolor='darkseagreen2')
  g.node("c_{k-1}", fillcolor='darkseagreen2')
  assert len(genotype) % 2 == 0
  steps = len(genotype) // 2
  for i in range(steps):
    g.node(str(i), fillcolor='lightblue')

  for i in range(steps):
    for k in [2*i, 2*i + 1]:
      op, j = genotype[k]
      if j == 0:
        u = "c_{k-2}"
      elif j == 1:
        u = "c_{k-1}"
      else:
        u = str(j-2)
      v = str(i)
      g.edge(u, v, label=op, fillcolor="gray")

  g.node("c_{k}", fillcolor='palegoldenrod')
  for i in range(steps):
    g.edge(str(i), "c_{k}", fillcolor="gray")

  g.render(filename)



if __name__ == '__main__':
  # if len(sys.argv) != 2:
  #   print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
  #   sys.exit(1)

  # genotype_name = sys.argv[1]
  # try:
  #   genotype = eval('genotypes.{}'.format(genotype_name))
  # except AttributeError:
  #   print("{} is not specified in genotypes.py".format(genotype_name)) 
  #   sys.exit(1)
  f= open("outputs/train_model/genotype_record_file.txt",'r')
  lines=f.readlines()

  f.close()
  line=lines[-1]
  genotype=line[line.find("reduce=")+7:(line.find("],",line.find("reduce="))+1)]
  genotype_new=[]
  currnt_index=0
  while not genotype.find("(",currnt_index)==-1:
    left=genotype.find("(",currnt_index)
    right=genotype.find(")",currnt_index)
    sub=genotype[left+1:right]
    genotype_new.append((sub.split(',')[0][1:-1],int(sub.split(',')[1][1])))
    currnt_index=right+1
  print(genotype_new)

  # plot(genotype_new, "normal")
  plot(genotype_new, "reduction")


