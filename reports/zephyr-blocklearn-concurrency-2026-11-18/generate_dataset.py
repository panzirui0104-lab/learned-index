from pathlib import Path

# 100,000 sorted unique 12-byte keys: user00000000 .. user00099999
out = Path(__file__).with_name("zephyr_keys_100k.txt")
with out.open("w", encoding="ascii", newline="\n") as stream:
    for i in range(100_000):
        stream.write(f"user{i:08d}\n")
print(out, out.stat().st_size)
