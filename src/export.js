var exportObj = {
    Tensor: Tensor,
    Variable: Variable,
    pool: pool,
    scale: scale,
    add: add,
    sub: sub,
    mul: mul,
    div: div,
    reshape: reshape,
    repeated: repeated,
    sumOuter: sumOuter,
    matmul: matmul,
    padImages: padImages,
    imagePatches: imagePatches,
    conv2d: conv2d,
    relu: relu
};

if ('undefined' !== typeof window) {
    window.jsnet = exportObj;
} else {
    module.exports = exportObj;
}
