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
    broadcast: broadcast,
    canBroadcast: canBroadcast,
    matmul: matmul,
    conv2d: conv2d,
    addBiases: addBiases,
    padImages: padImages,
    imagePatches: imagePatches,
    relu: relu
};

if ('undefined' !== typeof window) {
    window.jsnet = exportObj;
} else {
    module.exports = exportObj;
}
