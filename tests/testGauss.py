from scipy.stats import norm


spread_factor = 6.0
step = float(1/spread_factor)
threshold = 0.000001
print(f"step: {step}")
left = 0.0
right = step
for i in range(int(spread_factor)):
    l2 = left if left != 0.0 else left + threshold
    r2 = right if right != 1.0 else right - threshold

    x = norm.ppf([l2, r2])
    # print(x)
    print(norm.ppf(r2))

    # print(f"l: {l2}, r: {r2}")

    left += step
    right += step



