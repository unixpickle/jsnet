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
