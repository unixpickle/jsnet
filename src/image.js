function padImages(images, top, right, bottom, left) {
    if (images.value.shape.length !== 4) {
        throw Error('invalid image batch');
    }
    var padded = new Tensor([images.value.shape[0],
                             images.value.shape[1] + top + bottom,
                             images.value.shape[2] + left + right,
                             images.value.shape[3]]);
    for (var i = 0; i < images.value.shape[0]; ++i) {
        var srcIdx = i * shapeProduct(images.value.shape.slice(1));
        var dstIdx = i * shapeProduct(padded.shape.slice(1));
        for (var j = 0; j < images.value.shape[1]; ++j) {
            var jRowSize = shapeProduct(images.value.shape.slice(2));
            var srcStart = srcIdx + j * jRowSize;
            var dstStart = (dstIdx + (j + top) * shapeProduct(padded.shape.slice(2)) +
                left * padded.shape[3]);
            for (var k = 0; k < jRowSize; ++k) {
                padded.data[dstStart + k] = images.value.data[srcStart + k];
            }
        }
    }
    return {
        value: padded,
        backward: function(outGrad) {
            var unpadded = new Tensor(images.value.shape);
            for (var i = 0; i < unpadded.shape[0]; ++i) {
                var srcIdx = i * shapeProduct(outGrad.shape.slice(1));
                var dstIdx = i * shapeProduct(unpadded.shape.slice(1));
                for (var j = 0; j < unpadded.shape[1]; ++j) {
                    var jRowSize = shapeProduct(unpadded.shape.slice(2));
                    var dstStart = dstIdx + j * jRowSize;
                    var srcStart = (srcIdx + (j + top) * shapeProduct(outGrad.shape.slice(2)) +
                        left * outGrad.shape[3]);
                    for (var k = 0; k < jRowSize; ++k) {
                        unpadded.data[dstStart + k] = outGrad.data[srcStart + k];
                    }
                }
            }
            images.backward(unpadded);
        }
    };
}
