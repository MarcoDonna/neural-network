const relu = {
    f: x => x > 0 ? x : 0,
    d: x => x > 0 ? 1 : 0
}

const sigmoid = {
    f: x => 1 / (1 + Math.exp(-x)),
    d: x => sigmoid.f(x) * (1 - sigmoid.f(x))
}

const linear = {
    f: x => x,
    d: x => 1
}