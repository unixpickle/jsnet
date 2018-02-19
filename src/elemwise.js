function relu(input) {
    var result = input.value.copy();
    for (var i = 0; i < result.data.length; ++i) {
        if (result.data[i] < 0) {
            result.data[i] = 0;
        }
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < outGrad.data.length; ++i) {
                if (input.value.data[i] < 0) {
                    inGrad.data[i] = 0;
                }
            }
            input.backward(inGrad);
        }
    };
}

function rsqrt(input) {
    return pow(input, -0.5);
}

function square(input) {
    return pow(input, 2);
}

function pow(input, power) {
    var result = input.value.copy();
    for (var i = 0; i < result.data.length; ++i) {
        result.data[i] = Math.pow(result.data[i], power);
    }
    return {
        value: result,
        backward: function(outGrad) {
            var inGrad = outGrad.copy();
            for (var i = 0; i < inGrad.data.length; ++i) {
                inGrad.data[i] *= power * Math.pow(input.value.data[i], power - 1);
            }
            input.backward(inGrad);
        }
    }
}
