var DEFAULT_EPSILON = 0.001;

function normalizeChannels(input, epsilon) {
    epsilon = epsilon || DEFAULT_EPSILON;
    var centered = sub(input, broadcast(channelMean(input), input.value.shape));
    var variance = addScalar(channelMean(square(centered)), epsilon);
    return mul(centered, broadcast(rsqrt(variance), input.value.shape));
}

function channelMean(input) {
    var count = 1;
    while (input.value.shape.length > 1) {
        count *= input.value.shape[0];
        input = sumOuter(input);
    }
    return scale(input, 1 / count);
}

function logSoftmax(input) {
    var batchSize = shapeProduct(input.value.shape.slice(0, input.value.shape.length - 1));
    var channels = input.value.shape[input.value.shape.length - 1];
    var logSumExps = [];
    var result = input.value.copy();
    for (var i = 0; i < batchSize; ++i) {
        var max = -Infinity;
        var start = channels * i;
        for (var j = 0; j < channels; ++j) {
            max = Math.max(max, input.value.data[start + j]);
        }
        var logSumExp = 0;
        for (var j = 0; j < channels; ++j) {
            logSumExp += Math.exp(input.value.data[start + j] - max);
        }
        logSumExp = Math.log(logSumExp) + max;
        logSumExps[i] = logSumExp;
        for (var j = 0; j < channels; ++j) {
            result.data[start + j] -= logSumExp;
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < batchSize; ++i) {
                var start = channels * i;
                var gradSum = 0;
                for (var j = 0; j < channels; ++j) {
                    gradSum += outGrad.data[start + j];
                }
                for (var j = 0; j < channels; ++j) {
                    var normalizeGrad = Math.exp(input.value.data[start + j] - logSumExps[i]);
                    inGrad.data[start + j] = outGrad.data[start + j] - normalizeGrad * gradSum;
                }
            }
            input.backward(inGrad);
        }
    };
}
