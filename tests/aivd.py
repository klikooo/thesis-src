

alfa = [chr(i) for i in range(0x41, 0x5B)]
m = "wnbvruhxzpfdjlkegsitaocmqy".upper()
print(len(m))

text = """vr piba kshwlzirrst owl tcrrrltczltzh padz tkt rl jrt tcrr wahaitai tcrrvazyrlvlrhrltzrl rrl lzracr rvztr owl vr iajjrsibxkkd ke rrl hrxrzjr dkbwtzr zl lrvrsdwlv. Adi pr czdt vrrdlrjrl itaas vwl okks rrlrlvrstzh jrz tcrrvazyrlvlrhrltzrl pr bo rl jktzowtzrnszru lwws iajjrsibxkkd@jzlnyf.ld
""".upper()

total = []
for c in text:
    if 0x41 <= ord(c) <= 0x5B:
        i = m.index(c)
        # print(alfa[i])
        total.append(alfa[i])

    else:
        print(c)
        total.append(c)

print("".join(total))
