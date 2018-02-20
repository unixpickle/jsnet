var exportObj = {
    Tensor: Tensor,
    Variable: Variable,
    pool: pool,
    scale: scale,
    addScalar: addScalar,
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
    padImages: padImages,
    imagePatches: imagePatches,
    relu: relu,
    rsqrt: rsqrt,
    square: square,
    pow: pow,
    normalizeChannels: normalizeChannels,
    channelMean: channelMean,
    logSoftmax: logSoftmax
};

if ('undefined' !== typeof window) {
    window.jsnet = exportObj;
} else if ('undefined' !== typeof self) {
    self.jsnet = exportObj;
} else {
    module.exports = exportObj;
}
