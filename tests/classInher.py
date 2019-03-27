from models.Alex.KernelBig import KernelBig
from models.ConvNetKernel import ConvNetKernel

clazzes = [ConvNetKernel, KernelBig]

# for clazz in clazzes:
#     clazz.load_model('')

entry = {"a": clazzes[0].load_model}
entry2 = {"b": clazzes[1].load_model}
d = {}


d.update(entry)
d.update(entry2)
print(d['a'], d['b'])