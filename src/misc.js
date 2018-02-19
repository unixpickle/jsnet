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
