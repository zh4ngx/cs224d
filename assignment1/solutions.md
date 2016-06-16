# Derive Sigmoid

$sig(x) = 1 / (1 + e^-x)$

d sig(x) = dz 1 / (z) * dx/dz (z)

= 1 / z^2 * dx/dz (1 + e^-x)
= 1 / (1 + e^-x)^2 * -e^-x
= -e^-x / (1 + e^-x)^2
= (1 / (1 + e^-x)) * (-e^-x / (1 + e^-x))
= sig(x) * ((-e^-x + 1 - 1) / (1 + e^-x))
= sig(x) * ((-1 - e^-x) / (1 + e^-x) + 1 / (1 + e^-x))
= sig(x) * ((1 + e^-x) / (1 + e^-x) - 1 / (1 + e^-x))
= sig(x) * (1 - sig(x))