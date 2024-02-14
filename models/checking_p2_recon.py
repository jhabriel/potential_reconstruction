import sympy as sym

a, b, c = sym.symbols("a b c")  # coefficients of RT0 velocity field in 2d
Kxx, Kxy, Kyy = sym.symbols("Kxx Kxy Kyy")  # Coefficients of symmetric perm tensor
x, y = sym.symbols("x y")  # spatial variables

# Found coefficients
c0 = (a * Kyy) / (2 * (Kxy ** 2 - Kxx * Kyy))
c1 = (a * Kxy) / (Kxx * Kyy - Kxy ** 2)
c2 = (Kxy * c - Kyy * b) / (Kxx * Kyy - Kxy ** 2)
c3 = (a * Kxx) / (2 * (Kxy ** 2 - Kxx * Kyy))
c4 = (Kxx * c - Kxy * b) / (Kxy ** 2 - Kxx * Kyy)
c5 = 1  # We don't really care about c5, this is resolved with the other eq.

# Potential
s = c0 * x ** 2 + c1 * x * y + c2 * x + c3 * y ** 2 + c4 * y + c5

# Gradient of potential
grad_s = [sym.diff(s, x), sym.diff(s, y)]

# RT0 velocity
v = [a * x + b, a * y + c]

# Checking
lhs_x = - Kxx * grad_s[0] - Kxy * grad_s[1]
lhs_y = - Kxy * grad_s[0] - Kyy * grad_s[1]
assert v[0] == sym.simplify(lhs_x)
assert v[1] == sym.simplify(lhs_y)

